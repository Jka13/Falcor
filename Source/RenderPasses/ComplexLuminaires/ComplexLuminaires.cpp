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
#include "ComplexLuminaires.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"

namespace
{
    const std::string kGenerateSamplesShader = "RenderPasses/ComplexLuminaires/Shader/GenerateSamples.rt.slang";
    const std::string kDirectIlluminationPass = "RenderPasses/ComplexLuminaires/Shader/DirectIllumination.cs.slang";
    const std::string kDirectIlluminationReferencePass = "RenderPasses/ComplexLuminaires/Shader/DirectIlluminationReference.rt.slang";
    const std::string kDebugPass = "RenderPasses/ComplexLuminaires/Shader/Debug.rt.slang";
    const std::string kShaderModel = "6_5";
    const uint kMaxPayloadBytes = 96u;

    const std::string kOutputColor = "color";
    const std::string kInputVBuffer= "vBuffer";
    const std::string kInputView= "view";

    const Falcor::ChannelList kInputChannels{
        {kInputVBuffer, "gVBuffer", "vBuffer", false /*optional*/},
        {kInputView, "gView", "view", false /*optional*/},
    };

    const Falcor::ChannelList kOutputChannels{
        {kOutputColor, "gOutColor", "Output Color (linear)", false /*optional*/, ResourceFormat::RGBA32Float},
    };
    const Gui::DropdownList kModes{{0, "VPL"}, {1, "Reference"}};
    } // namespace


extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ComplexLuminaires>();
}

ComplexLuminaires::ComplexLuminaires(ref<Device> pDevice, const Properties& props)
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

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
}

Properties ComplexLuminaires::getProperties() const
{
    return {};
}

RenderPassReflection ComplexLuminaires::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ComplexLuminaires::prepareLight(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pLights = mpScene->getLightCollection(pRenderContext);
    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveLightSampler)
        {
            switch (mEmissiveLightSamplerType)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveLightSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveLightSampler = std::make_unique<LightBVHSampler>(pRenderContext, mpScene, mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::Null:
            default:
                throw RuntimeError("Invalid Emissive Sampler type");
            }
        }
        if (mpEmissiveLightSampler)
            mpEmissiveLightSampler->update(pRenderContext); //return boolean
    }
}

void ComplexLuminaires::preparePhotonBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPhotonBuffer || mChangedPhotonBufferSize)
    {
        mpPhotonBuffer.reset();
        mpPhotonBuffer = Buffer::createStructured(
            mpDevice, 3 * sizeof(float3), mDispatchedPhotonsPerIteration,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
        );
        mpPhotonBuffer->setName("ComplexLuminaires::PhotonBuffer");
    }
}

void ComplexLuminaires::preparePhotonAABBBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPhotonAABBs || mChangedPhotonBufferSize)
    {
        mpPhotonAABBs.reset();
        mpPhotonAABBs = Buffer::createStructured(mpDevice, sizeof(AABB), mDispatchedPhotonsPerIteration);
        mpPhotonAABBs->setName("PM::PhotonAABB");
    }
}

void ComplexLuminaires::prepareAccelerationStructure()
{
    // Delete the Photon AS if max Buffer size changes
    if (mChangedPhotonBufferSize)
    {
        mpPhotonAS.reset();
    }

    // Create the Photon AS
    if (!mpPhotonAS)
    {
        std::vector<uint64_t> aabbCount = {mMaxPhotonCount};
        std::vector<uint64_t> aabbGPUAddress = {mpPhotonAABBs->getGpuAddress()};
        mpPhotonAS = std::make_unique<CustomAccelerationStructure>(mpDevice, aabbCount, aabbGPUAddress);
    }
}

void ComplexLuminaires::buildAccelerationStructure(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->uavBarrier(mpPhotonAABBs.get());
    std::vector<uint64_t> photonBuildSize = {mMaxPhotonCount};
    mpPhotonAS->update(pRenderContext, photonBuildSize);
}

void ComplexLuminaires::prepareRayTracingShader(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    mGenerateSamplesPass.initRTProgram(mpDevice, mpScene, kGenerateSamplesShader, kMaxPayloadBytes, globalTypeConformances);
    mDebugPass.initRTCollectionProgram(mpDevice, mpScene, kDebugPass, kMaxPayloadBytes, globalTypeConformances);
    mDirectIlluminationReferencePass.initRTCollectionProgram(mpDevice, mpScene, kDirectIlluminationReferencePass, kMaxPayloadBytes, globalTypeConformances);
}

void ComplexLuminaires::setSceneData(const RenderData& renderData, const ShaderVar& var)
{
    auto sceneDataVar = var["sdh"];
    sceneDataVar["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    sceneDataVar["gView"] = renderData[kInputView]->asTexture();
}

void ComplexLuminaires::prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareGenerateSamplesPass");

    mGenerateSamplesPass.pProgram->addDefine("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");
    mGenerateSamplesPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mGenerateSamplesPass.pVars)
    {
        if (mpEmissiveLightSampler)
            mGenerateSamplesPass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());
        mGenerateSamplesPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mGenerateSamplesPass.pVars->getRootVar();
    mpEmissiveLightSampler->setShaderData(var["LightCB"]["gEmissiveLightSampler"]);
    mpSampleGenerator->setShaderData(var);
    uint flags = 0;

    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["CB"]["gFlags"] = flags;
    var["CB"]["gMaxRecursion"] = mMaxRecursion;
    var["CB"]["gAABBSize"] = mAABBSize;
    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gPhotonAABBs"] = mpPhotonAABBs;
}

void ComplexLuminaires::prepareDirectIlluminationPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareDirectIllumination");
    if (!mpDirectIlluminationPass || mChangedPhotonBufferSize)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kDirectIlluminationPass).csEntry("main").setShaderModel("6_6");
        desc.addTypeConformances(mpScene->getTypeConformances());
        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        mpDirectIlluminationPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    auto var = mpDirectIlluminationPass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var, 1);
    mpSampleGenerator->setShaderData(var);
    setSceneData(renderData, var);
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPhotonCount"] = mDispatchedPhotonsPerIteration;
    var["CB"]["gConeExponent"] = mCosConeExponent;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    var["gPhotonBuffer"] = mpPhotonBuffer;
}

void ComplexLuminaires::prepareDirectIlluminationReferencePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareReferencePass");
    if (!mDirectIlluminationReferencePass.pVars)
    {
        mDirectIlluminationReferencePass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mDirectIlluminationReferencePass.pVars->getRootVar();

    uint flags = 0;
    var["CB"]["gFlags"] = flags;
    var["CB"]["gPhotonCount"] = mDispatchedPhotonsPerIteration;
    var["gPhotonAABBs"] = mpPhotonAABBs;
    var["gPhotonBuffer"] = mpPhotonBuffer;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    setSceneData(renderData, var);

    mpPhotonAS->bindTlas(var, "gPhotonAccelerationStructure");
}

void ComplexLuminaires::prepareDebugPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareDebugPass");

    if (!mDebugPass.pVars)
    {
        mDebugPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mDebugPass.pVars->getRootVar();

    uint flags = 0;
    var["CB"]["gFlags"] = flags;
    var["gPhotonAABBs"] = mpPhotonAABBs;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    setSceneData(renderData, var);

    mpPhotonAS->bindTlas(var, "gPhotonAccelerationStructure");
}

void ComplexLuminaires::directIllumiantionVPL(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIllumination");
    mpDirectIlluminationPass->execute(pRenderContext, launchDim.x, launchDim.y);
}

void ComplexLuminaires::directIllumiantionReference(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIlluminationReference");
    mpScene->raytrace(pRenderContext, mDirectIlluminationReferencePass.pProgram.get(), mDirectIlluminationReferencePass.pVars, uint3(launchDim, 1));
}


void ComplexLuminaires::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ++mFrameCount;
    if (!mpScene)
        return;
    //prepareResources
    preparePhotonBuffer(pRenderContext, renderData);
    prepareLight(pRenderContext, renderData);
    preparePhotonAABBBuffer(pRenderContext, renderData);
    prepareAccelerationStructure();

    //prepareShaders
    prepareGenerateSamplesPass(pRenderContext, renderData);
    prepareDebugPass(pRenderContext, renderData);
    mChangedPhotonBufferSize = false;

    //generateSamples
    mpScene->raytrace(pRenderContext,mGenerateSamplesPass.pProgram.get(),mGenerateSamplesPass.pVars, uint3(mDispatchedPhotonsPerIteration, 1, 1));
    buildAccelerationStructure(pRenderContext, renderData);

    uint2 launchDim = renderData.getDefaultTextureDims();
    //Pass for simple direct illumination
    switch (mMode)
    {
    case 0:
        prepareDirectIlluminationPass(pRenderContext, renderData);
        directIllumiantionVPL(pRenderContext, renderData, launchDim);
        break;
    case 1:
        prepareDirectIlluminationReferencePass(pRenderContext, renderData);
        directIllumiantionReference(pRenderContext, renderData, launchDim);
        break;
    default:
        break;
    }
    //Debugpass for displaying dispatched photons
    if (mShowDebug)
    {
        FALCOR_PROFILE(pRenderContext, "DebugPass");
        mpScene->raytrace(pRenderContext, mDebugPass.pProgram.get(),mDebugPass.pVars, uint3(launchDim, 1));
    }
}

void ComplexLuminaires::renderUI(Gui::Widgets& widget)
{
    mChangedPhotonBufferSize |= widget.var("Number of Photons", mDispatchedPhotonsPerIteration, 1u, 10000000u);
    if (mChangedPhotonBufferSize)
        mMaxPhotonCount = mDispatchedPhotonsPerIteration;
    widget.var("Recursion Depth", mMaxRecursion, 0u, 50u);
    widget.var("Cone Exponent", mCosConeExponent, 0.f, 100000.f);
    widget.var("Photon AABB Size", mAABBSize, 0.f, 1.f);
    widget.checkbox("Show Debug View", mShowDebug);
    widget.dropdown("Mode", kModes, mMode);
}

void ComplexLuminaires::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    //Reset Scene
    mpScene = pScene;

    mGenerateSamplesPass = RayTraceProgramHelper::create();
    mDebugPass = RayTraceProgramHelper::create();
    mDirectIlluminationReferencePass = RayTraceProgramHelper::create();
    mpDirectIlluminationPass.reset();
    mpEmissiveLightSampler.reset();

    if (mpScene)
    {
        const auto& bounds= mpScene->getSceneBounds();
        const float sceneExtend = math::length(bounds.extent());

        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }
        prepareRayTracingShader(pRenderContext);
    }
}

void ComplexLuminaires::RayTraceProgramHelper::initRTProgram(ref<Device> device,ref<Scene> scene,const std::string& shaderName,
                                                     uint maxPayloadBytes,const Program::TypeConformanceList& globalTypeConformances)
{
    RtProgram::Desc desc;
    desc.addShaderModules(scene->getShaderModules());
    desc.addShaderLibrary(shaderName);
    desc.setMaxPayloadSize(maxPayloadBytes);
    desc.setMaxAttributeSize(scene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);
    if (!scene->hasProceduralGeometry())
        desc.setPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    pBindingTable = RtBindingTable::create(1, 1, scene->getGeometryCount());
    auto& sbt = pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    sbt->setMiss(0, desc.addMiss("miss"));

    //TODO: Support more geometry types and more material conformances
    if (scene->hasGeometryType(Scene::GeometryType::TriangleMesh))
    {
        sbt->setHitGroup(
            0, scene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit") );
    }

    pProgram = RtProgram::create(device, desc, scene->getSceneDefines());
}

void ComplexLuminaires::RayTraceProgramHelper::initRTCollectionProgram(
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

void ComplexLuminaires::RayTraceProgramHelper::initProgramVars(ref<Device> pDevice,ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);

    // Configure program.
    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances());
    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    pVars = RtProgramVars::create(pDevice ,pProgram, pBindingTable);

    // Bind utility classes into shared data.
    auto var = pVars->getRootVar();
    pSampleGenerator->setShaderData(var);
}
