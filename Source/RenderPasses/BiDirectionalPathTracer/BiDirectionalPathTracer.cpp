/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "BiDirectionalPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"

namespace
{
const std::string kShaderGeneratePhotons = "RenderPasses/BiDirectionalPathTracer/CreateLightPaths.rt.slang";
const std::string kShaderGenerateCameraPaths = "RenderPasses/BiDirectionalPathTracer/CreateCameraPaths.rt.slang";
const std::string kShaderCollectPhotons = "RenderPasses/BiDirectionalPathTracer/CollectBackProject.rt.slang";
const std::string kCombinePaths = "RenderPasses/BiDirectionalPathTracer/CombinePaths.rt.slang";

const std::string kShaderModel = "6_5";
const uint kMaxPayloadBytes = 96u;

const std::string kOutputColor = "color";

const Falcor::ChannelList kOutputChannels{
    {kOutputColor, "gOutColor", "Output Color (linear)", false /*optional*/, ResourceFormat::RGBA32Float},
};
const Gui::DropdownList kModes{{0, "Reprojection"}, {1, "BDPT"}};

} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, BiDirectionalPathTracer>();
}

BiDirectionalPathTracer::BiDirectionalPathTracer(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(Device::ShaderModel::SM6_5))
    {
        throw RuntimeError("ReSTIR_FG: Shader Model 6.5 is not supported by the current device");
    }
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("ReSTIR_FG: Raytracing Tier 1.1 is not supported by the current device");
    }

    // TODO Handle Properties

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
}

Properties BiDirectionalPathTracer::getProperties() const
{
    return {};
}

RenderPassReflection BiDirectionalPathTracer::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void BiDirectionalPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) // Return on empty scene
        return;

    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // Prepare used Datas and Buffers
    prepareLighting(pRenderContext);
    prepareBuffers(pRenderContext, renderData);

    switch (mMode)
    {
    case 0:
        prepareLinkedListResources(pRenderContext, renderData); 
        break;
    default:
        break;
    }

    // RenderPasses
    handlePhotonCounter(pRenderContext);
    preparePhotonsPass(pRenderContext, renderData);
    prepareCameraPathPass(pRenderContext, renderData);
    prepareCombinePathsPass(pRenderContext, renderData);

    if (mpScene->useEmissiveLights())
    {
        generateEmissivePhotonsPass(pRenderContext, renderData);
    }
    
    if (mpScene->useAnalyticLights())
    {
        generateAnalyticPhotonsPass(pRenderContext, renderData);
    }

    generateCameraPathPass(pRenderContext, renderData);

    switch (mMode)
    {
    case 0:
        collectPhotons(pRenderContext, renderData);
        break;
    case 1:
        combinePaths(pRenderContext, renderData);
        break;
    default:
        break;
    }
    mFrameCount++;
}

void BiDirectionalPathTracer::renderUI(Gui::Widgets& widget)
{
    bool changed = false;
    uint dispatchedPhotons = mNumDispatchedPhotons;
    bool disPhotonChanged = widget.var("Dispatched Light Paths", dispatchedPhotons, mPhotonYExtent, 9984000u, (float)mPhotonYExtent);
    if (disPhotonChanged)
        mNumDispatchedPhotons = (uint)(dispatchedPhotons / mPhotonYExtent) * mPhotonYExtent;

    widget.text("Light Points: " + std::to_string(mCurrentPhotonCount) + " / " + std::to_string(mNumMaxPhotons));

    widget.var("Light Point Buffer Size", mNumMaxPhotonsUI, 100u, 100000000u, 100);
    mChangePhotonLightBufferSize = widget.button("Apply", true);

    changed |= widget.var("Max Bounces", mLightMaxBounces, 0u, 32u);
    if (mpScene && mpScene->useAnalyticLights() && mpScene->useEmissiveLights())
    {
        widget.text("Emissive Lights Samples: " + std::to_string((uint) ceil(mNumDispatchedPhotons * mEmissivePercentage)));
        widget.var("Emissive Lights Percentage: ", mEmissivePercentage, 0.f, 1.f, 0.1f);
        mAnalyticPercentage = 1 - mEmissivePercentage;
        widget.text("Analytic Lights Samples: " + std::to_string((uint) floor(mNumDispatchedPhotons * mAnalyticPercentage)));
        widget.var("Analytic Lights Percentage: ", mAnalyticPercentage, 0.f, 1.f, 0.1f);
        mEmissivePercentage = 1 - mAnalyticPercentage;
    }
    changed |= widget.dropdown("Current Mode", kModes, mMode);
    mOptionsChanged |= changed;
}

void BiDirectionalPathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Reset Scene
    mpScene = pScene;

    mGeneratePhotonPass = RayTraceProgramHelper::create();
    mGenerateCameraPathPass = RayTraceProgramHelper::create();
    mCollectPhotonPass = RayTraceProgramHelper::create();
    mCombinePathsPass= RayTraceProgramHelper::create();
    mpEmissiveLightSampler.reset();

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        prepareRayTracingShaders(pRenderContext);
    }
}

bool BiDirectionalPathTracer::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;
    // Make sure that the emissive light is up to date
    auto& pLights = mpScene->getLightCollection(pRenderContext);

    if (mpScene->useEmissiveLights())
    {
        // Init light sampler if not set
        if (!mpEmissiveLightSampler)
        {
            // Ensure that emissive light struct is build by falcor
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            // TODO: Support different types of sampler
            mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene);
            lightingChanged = true;
        }
    }
    else
    {
        if (mpEmissiveLightSampler)
        {
            mpEmissiveLightSampler = nullptr;
            lightingChanged = true;
            mGeneratePhotonPass.pVars.reset();
        }
    }

    // Update Emissive light sampler
    if (mpEmissiveLightSampler)
    {
        lightingChanged |= mpEmissiveLightSampler->update(pRenderContext);
    }

    return lightingChanged;
}

void BiDirectionalPathTracer::prepareBuffers(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint2 resolution = renderData.getDefaultTextureDims();
    uint numPixel = resolution.x * resolution.y;
    if (!mpLightPaths)
    {
        mPathsBufferSize = numPixel * mLightMaxBounces;
        //TODO: adapt size to the size of the packed hit info with HitInfo::kDefaultFormat
        mpLightPaths = Buffer::createStructured(
            mpDevice, sizeof(float3) + sizeof(uint4) + sizeof(float3), mPathsBufferSize, 
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            Buffer::CpuAccess::None,
            nullptr,
            false
            );
        mpLightPaths->setName("BDPT::LightPaths");
    }
    if (!mpCameraPaths)
    {
        mPathsBufferSize =numPixel * mLightMaxBounces;
        mpCameraPaths = Buffer::createStructured(
            mpDevice, sizeof(uint4) + sizeof(float3), mPathsBufferSize, 
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            Buffer::CpuAccess::None,
            nullptr,
            false
            );
        mpCameraPaths->setName("BDPT::CameraPaths");
    }
    if (mChangePhotonLightBufferSize)
    {
        mNumMaxPhotons = mNumMaxPhotonsUI;
        mpLightTraceData.reset();
    }
    // Photon
    if (!mpPhotonCounter)
    {
        mpPhotonCounter = Buffer::create(mpDevice, sizeof(uint) * 4);
        mpPhotonCounter->setName("PM::PhotonCounterGPU");
    }
    if (!mpPhotonCounterCPU)
    {
        mpPhotonCounterCPU = Buffer::create(mpDevice, sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpPhotonCounterCPU->setName("PM::PhotonCounterCPU");
    }
    if (!mpLightTraceData)
    {
        mpLightTraceData = Buffer::createStructured(mpDevice, sizeof(uint) * 4, mNumMaxPhotons);
        mpLightTraceData->setName("PM::PhotonData");
    }
}

void BiDirectionalPathTracer::prepareLinkedListResources(RenderContext* pRenderContext, const RenderData& renderData) {
    if (!mpLinkedList)
    {
        mpLinkedList = Buffer::createStructured(mpDevice, sizeof(float4), mNumMaxPhotons);
    }
    if (!mpHeadCounter)
    {
        uint2 frameDim = renderData.getDefaultTextureDims();
        mpHeadCounter = Texture::create2D(mpDevice, frameDim.x , frameDim.y, ResourceFormat::R32Uint, 1U, 1, nullptr, ResourceBindFlags::AllColorViews);
    }
    pRenderContext->clearUAV(mpHeadCounter->getUAV().get(), uint4(-1));
}

void BiDirectionalPathTracer::prepareRayTracingShaders(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();

    // TODO specify the payload bytes for each pass
    mGeneratePhotonPass.initRTProgram(mpDevice, mpScene, kShaderGeneratePhotons, kMaxPayloadBytes, globalTypeConformances);
    mGenerateCameraPathPass.initRTProgram(mpDevice, mpScene, kShaderGenerateCameraPaths, kMaxPayloadBytes, globalTypeConformances);
    mCombinePathsPass.initRTProgram(mpDevice, mpScene, kCombinePaths , kMaxPayloadBytes, globalTypeConformances);

    // Special Program for the Photon Collection as the photon acceleration structure is used
    mCollectPhotonPass.initRTCollectionProgram(mpDevice, mpScene, kShaderCollectPhotons, kMaxPayloadBytes, globalTypeConformances);
}

float getNormalizedPixelSize(uint2 frameDim, float fovY, float aspect)
{
    float h = tan(fovY / 2.f) * 2.f;
    float w = h * aspect;
    float wPix = w / frameDim.x;
    float hPix = h / frameDim.y;
    return wPix * hPix;
}
float2 getPixelWidthHeight(uint2 frameDim, float fovY, float aspect)
{
    float h = tan(fovY / 2.f) * 2.f;
    float w = h * aspect;
    float wPix = w / frameDim.x;
    float hPix = h / frameDim.y;
    return float2(wPix, hPix);
}

void BiDirectionalPathTracer::preparePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData, bool clearBuffers)
{
    FALCOR_PROFILE(pRenderContext, "PhotonGeneration");

    // TODO Clear via Compute pass?
    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));
    // Defines
    mGeneratePhotonPass.pProgram->addDefine("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");
    mGeneratePhotonPass.pProgram->addDefine("PHOTON_BUFFER_SIZE_GLOBAL", std::to_string(mNumMaxPhotons));
    mGeneratePhotonPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mGeneratePhotonPass.pVars)
    {
        FALCOR_ASSERT(mGeneratePhotonPass.pProgram);
        if (mpEmissiveLightSampler)
            mGeneratePhotonPass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());

        mGeneratePhotonPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    };

    FALCOR_ASSERT(mGeneratePhotonPass.pVars);

    auto var = mGeneratePhotonPass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    // Set constants (uniforms).
    //
    // PerFrame Constant Buffer
    uint2 frameDim = renderData.getDefaultTextureDims();
    std::string nameBuf = "PerFrame";
    var[nameBuf]["gFrameCount"] = mFrameCount;
    const auto& cameraData = mpScene->getCamera()->getData();
    var[nameBuf]["gNormalizedPixelArea"] = getNormalizedPixelSize(frameDim, focalLengthToFovY(cameraData.focalLength, cameraData.frameHeight), cameraData.aspectRatio);
    var[nameBuf]["gPixelWidthHeight"] = getPixelWidthHeight(frameDim, focalLengthToFovY(cameraData.focalLength, cameraData.frameHeight), cameraData.aspectRatio);
    var[nameBuf]["gProjectedFocalLength"] = math::mul(cameraData.projMat, float4(0.f, 0.f, cameraData.focalLength, 1.f)).w;
    var[nameBuf]["gFrameDim"] = frameDim; 
    var["gLightPaths"] = mpLightPaths;
    // Upload constant buffer only if options changed

    // Fill flags
    uint flags = 0;

    nameBuf = "CB";
    var[nameBuf]["gMaxRecursion"] = mLightMaxBounces;
    var[nameBuf]["gFlags"] = flags;

    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->setShaderData(var["Light"]["gEmissiveSampler"]);

    // Set the photon buffers
    
    var["gLightTraceData"] = mpLightTraceData;
    var["gLinkedList"] = mpLinkedList;
    var["gPhotonCounter"] = mpPhotonCounter;
    var["gHeadCounter"] = mpHeadCounter;
}


void BiDirectionalPathTracer::generateEmissivePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "generateEmissivePhotons");
    // Get dimensions of ray dispatch.
    uint dispatchedPhotons = ceil(mNumDispatchedPhotons * mEmissivePercentage);
    const uint2 targetDim = uint2(std::max(1u, dispatchedPhotons / mPhotonYExtent), mPhotonYExtent);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Trace the photons
    mpScene->raytrace(pRenderContext, mGeneratePhotonPass.pProgram.get(), mGeneratePhotonPass.pVars, uint3(targetDim, 1));

}

void BiDirectionalPathTracer::generateAnalyticPhotonsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "generateAnalyticPhotons");
    // Fill flags
    uint flags = 0;
    flags |= 0x20; // Analytic lights collect flag

    FALCOR_ASSERT(mGeneratePhotonPass.pVars);

    auto var = mGeneratePhotonPass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    var["CB"]["gFlags"] = flags;

    // Get dimensions of ray dispatch.
    uint dispatchedPhotons = floor(mNumDispatchedPhotons * mAnalyticPercentage);
    const uint2 targetDim = uint2(std::max(1u, dispatchedPhotons / mPhotonYExtent), mPhotonYExtent);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Trace the photons
    mpScene->raytrace(pRenderContext, mGeneratePhotonPass.pProgram.get(), mGeneratePhotonPass.pVars, uint3(targetDim, 1));

}

void BiDirectionalPathTracer::prepareCameraPathPass(RenderContext* pRenderContext, const RenderData& renderData, bool clearBuffers)
{
    FALCOR_PROFILE(pRenderContext, "CameraPathGeneration");

    // Defines
    mGenerateCameraPathPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mGenerateCameraPathPass.pVars)
    {
        FALCOR_ASSERT(mGenerateCameraPathPass.pProgram);
        mGenerateCameraPathPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    };

    FALCOR_ASSERT(mGenerateCameraPathPass.pVars);

    auto var = mGenerateCameraPathPass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    // Set constants (uniforms).
    //
    // PerFrame Constant Buffer
    uint2 frameDim = renderData.getDefaultTextureDims();
    std::string nameBuf = "PerFrame";
    var[nameBuf]["gFrameCount"] = mFrameCount;
    const auto& cameraData = mpScene->getCamera()->getData();
    var[nameBuf]["gFrameDim"] = frameDim; 
    // CB vars
    uint flags = 0; // Fill flags
    nameBuf = "CB";
    var[nameBuf]["gMaxRecursion"] = mLightMaxBounces;
    var[nameBuf]["gFlags"] = flags;
    //vars 
    var["gCameraPaths"] = mpCameraPaths;
}

void BiDirectionalPathTracer::generateCameraPathPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "GenerateCameraPaths");
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    // Trace the photons
    mpScene->raytrace(pRenderContext, mGenerateCameraPathPass.pProgram.get(), mGenerateCameraPathPass.pVars, uint3(targetDim, 1));
}

void BiDirectionalPathTracer::prepareCombinePathsPass(RenderContext* pRenderContext, const RenderData& renderData, bool clearBuffers)
{
    FALCOR_PROFILE(pRenderContext, "CreateCombinePathsPass");
    // TODO Clear via Compute pass?
    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));
    // Defines
    mCombinePathsPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mCombinePathsPass.pVars)
    {
        FALCOR_ASSERT(mCombinePathsPass.pProgram);
        mCombinePathsPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    };

    FALCOR_ASSERT(mCombinePathsPass.pVars);

    auto var = mCombinePathsPass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    // Set constants (uniforms).
    //
    // PerFrame Constant Buffer
    uint2 frameDim = renderData.getDefaultTextureDims();
    std::string nameBuf = "PerFrame";
    var[nameBuf]["gFrameCount"] = mFrameCount;
    const auto& cameraData = mpScene->getCamera()->getData();
    var[nameBuf]["gFrameDim"] = frameDim; 
    var["gCameraPaths"] = mpCameraPaths;
    var["gLightPaths"] = mpLightPaths;
    var["gColor"] = renderData[kOutputColor]->asTexture();

    // Fill flags
    uint flags = 0;

    nameBuf = "CB";
    var[nameBuf]["gMaxRecursion"] = mLightMaxBounces;
    var[nameBuf]["gFlags"] = flags;
}

void BiDirectionalPathTracer::combinePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "CombinePaths");
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    // Trace the photons
    mpScene->raytrace(pRenderContext, mCombinePathsPass.pProgram.get(), mCombinePathsPass.pVars, uint3(targetDim, 1));
}

void BiDirectionalPathTracer::handlePhotonCounter(RenderContext* pRenderContext)
{
    // Copy the photonCounter to a CPU Buffer
    pRenderContext->copyBufferRegion(mpPhotonCounterCPU.get(), 0, mpPhotonCounter.get(), 0, sizeof(uint32_t));

    void* data = mpPhotonCounterCPU->map(Buffer::MapType::Read);
    std::memcpy(&mCurrentPhotonCount, data, sizeof(uint));
    mpPhotonCounterCPU->unmap();
}

void BiDirectionalPathTracer::collectPhotons(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "CollectPhotons");
    //Defines
    mCollectPhotonPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mCollectPhotonPass.pVars)
        mCollectPhotonPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    FALCOR_ASSERT(mCollectPhotonPass.pVars);

    auto var = mCollectPhotonPass.pVars->getRootVar();

    // Set Constant Buffers
    std::string nameBuf = "PerFrame";
    var[nameBuf]["gFrameCount"] = mFrameCount;

    var["gLightTraceData"] = mpLightTraceData;
    var["gLinkedList"] = mpLinkedList;
    var["gLightCounter"] = mpPhotonCounter;
    var["gHeadCounter"] = mpHeadCounter;
    var["gLightPaths"] = mpLightPaths;
    var["gCameraPaths"] = mpCameraPaths;

    var["gColor"] = renderData[kOutputColor]->asTexture();

    // Create dimensions based on the number of VPLs
    uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Trace the photons
    mpScene->raytrace(pRenderContext, mCollectPhotonPass.pProgram.get(), mCollectPhotonPass.pVars, uint3(targetDim, 1));
}

void BiDirectionalPathTracer::RayTraceProgramHelper::initRTProgram(
    ref<Device> device,
    ref<Scene> scene,
    const std::string& shaderName,
    uint maxPayloadBytes,
    const Program::TypeConformanceList& globalTypeConformances
)
{
    RtProgram::Desc desc;
    desc.addShaderModules(scene->getShaderModules());
    desc.addShaderLibrary(shaderName);
    desc.setMaxPayloadSize(maxPayloadBytes);
    desc.setMaxAttributeSize(scene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);
    if (!scene->hasProceduralGeometry())
        desc.setPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    pBindingTable = RtBindingTable::create(2, 2, scene->getGeometryCount());
    auto& sbt = pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    sbt->setMiss(0, desc.addMiss("miss"));
    sbt->setMiss(1, desc.addMiss("shadowMiss"));

    // TODO: Support more geometry types and more material conformances
    if (scene->hasGeometryType(Scene::GeometryType::TriangleMesh))
    {
        sbt->setHitGroup(0, scene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        sbt->setHitGroup(1, scene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));
    }

    pProgram = RtProgram::create(device, desc, scene->getSceneDefines());
}

void BiDirectionalPathTracer::RayTraceProgramHelper::initRTCollectionProgram(
    ref<Device> device,
    ref<Scene> scene,
    const std::string& shaderName,
    uint maxPayloadBytes,
    const Program::TypeConformanceList& globalTypeConformances
)
{
    RtProgram::Desc desc;
    desc.addShaderModules(scene->getShaderModules());
    desc.addShaderLibrary(shaderName);
    desc.setMaxPayloadSize(maxPayloadBytes);
    desc.setMaxAttributeSize(scene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);

    pBindingTable = RtBindingTable::create(1, 1, scene->getGeometryCount()); // Geometry Count is still needed as the scenes AS is still
                                                                             // bound
    auto& sbt = pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen", globalTypeConformances)); // Type conformances for material model
    sbt->setMiss(0, desc.addMiss("miss"));
    sbt->setHitGroup(0, 0, desc.addHitGroup("", "anyHit", "intersection", globalTypeConformances));

    pProgram = RtProgram::create(device, desc, scene->getSceneDefines());
}

void BiDirectionalPathTracer::RayTraceProgramHelper::initProgramVars(ref<Device> pDevice, ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);

    // Configure program.
    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances());
    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);

    // Bind utility classes into shared data.
    auto var = pVars->getRootVar();
    pSampleGenerator->setShaderData(var);
}
