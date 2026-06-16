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
#include "ComplexLuminairesReSTIR_PT.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"

namespace
{
    //CL
    const std::string kGenerateSamplesShader = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/GenerateSamples.rt.slang";
    const std::string kDirectIlluminationPass = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/DirectIllumination.cs.slang";
    const std::string kDirectIlluminationReferencePass = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/DirectIlluminationReference.rt.slang";
    const std::string kDebugPass = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/Debug.rt.slang";

    //PT
    const std::string kPathTracingPass = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/PathTracing.rt.slang";
    const std::string kPathResamplePass = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/PathResample.rt.slang";
    const char kMaxBounces[] = "maxBounces";
    const char kComputeDirect[] = "computeDirect";
    const char kUseImportanceSampling[] = "useImportanceSampling";

    //ReSTIR
    const std::string kSampleShader = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/Sample.rt.slang";
    const std::string kResampleShader = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/Resample.cs.slang";
    const std::string kCombineShader = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/Combine.cs.slang";

    //Splatting
    const std::string kShaderTemporalSplatReservoirs = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/TemporalSplatReservoir.cs.slang";
    const std::string kShaderSortSplatReservoirs = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/SortSplatReservoirs.cs.slang";
    const std::string kShaderSplatResample = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/SplattingResample.cs.slang";
    const std::string kShaderSplatCombine = "RenderPasses/ComplexLuminairesReSTIR_PT/Shader/SplattingCombine.cs.slang";

    const std::string kShaderModel = "6_5";
    const uint kMaxPayloadBytes = 96u;

    const std::string kOutputColor = "color";
    const std::string kOutputDebug = "debug";
    const std::string kOutputDebug1 = "debug1";
    const std::string kInputVBuffer= "vBuffer";
    const std::string kInputView= "view";
    const std::string kInputMVec= "motionVector";

    const Falcor::ChannelList kInputChannels{
        {kInputVBuffer, "gVBuffer", "vBuffer", false /*optional*/},
        {kInputMVec, "gMVec", "motionVector", false /*optional*/},
        {kInputView, "gView", "view", false /*optional*/},
    };

    const Falcor::ChannelList kOutputChannels{
        {kOutputColor, "gOutColor", "Output Color (linear)", false /*optional*/, ResourceFormat::RGBA32Float},
        {kOutputDebug, "gOutDebug", "Output Debug", false /*optional*/, ResourceFormat::RGBA32Float},
        {kOutputDebug1, "gOutDebug1", "Output Debug1", false /*optional*/, ResourceFormat::RGBA32Float},
    };
    const Gui::DropdownList kModes{{0, "Path Tracer"}, {1, "ReSTIR PT"}};
    } // namespace


extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ComplexLuminairesReSTIR_PT>();
}

ComplexLuminairesReSTIR_PT::ComplexLuminairesReSTIR_PT(ref<Device> pDevice, const Properties& props)
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

RenderPassReflection ComplexLuminairesReSTIR_PT::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ComplexLuminairesReSTIR_PT::prepareLight(RenderContext* pRenderContext, const RenderData& renderData)
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
        else
            mpEmissiveLightSampler->update(pRenderContext); //return boolean
    }
}

void ComplexLuminairesReSTIR_PT::prepareDirectVPLBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpDirectVPLBuffer || mChangedPhotonBufferSize)
    {
        mpDirectVPLBuffer.reset();
        mpDirectVPLBuffer = Buffer::createStructured(
            mpDevice, 12 * sizeof(float), mMaxPhotonCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
        );
        mpDirectVPLBuffer->setName("ComplexLuminairesReSTIR_PT::DirectVPLBuffer");
    }
}

void ComplexLuminairesReSTIR_PT::prepareIndirectVPLBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpIndirectVPLBuffer || mChangedVPLBufferSize)
    {
        mpIndirectVPLBuffer.reset();
        mpIndirectVPLBuffer = Buffer::createStructured(
            mpDevice, 3 * sizeof(float3) + sizeof(uint4), mMaxPhotonCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
        );
        mpIndirectVPLBuffer->setName("ComplexLuminairesReSTIR_PT::VPLBuffer");
    }
}

void ComplexLuminairesReSTIR_PT::preparePhotonAABBBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPhotonAABBs || mChangedPhotonBufferSize)
    {
        mpPhotonAABBs.reset();
        mpPhotonAABBs = Buffer::createStructured(mpDevice, sizeof(AABB), mMaxPhotonCount);
        mpPhotonAABBs->setName("PM::PhotonAABB");
    }
}

void ComplexLuminairesReSTIR_PT::prepareCounter(RenderContext* pRenderContext, const RenderData& renderData)
{
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
    if (!mpLinkedListCounter)
    {
        mpLinkedListCounter = Buffer::create(mpDevice, sizeof(uint) * 4);
        mpLinkedListCounter->setName("PM::LinkedListCounterGPU");
    }
    if (!mpLinkedListCounterCPU)
    {
        mpLinkedListCounterCPU = Buffer::create(mpDevice, sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpLinkedListCounterCPU->setName("PM::LinkedListCounterCPU");
    }
}

void ComplexLuminairesReSTIR_PT::prepareLinkedList(RenderContext* renderContext, const RenderData& renderData)
{
    if (!mpReprojectionLinkedList)
    {
        mpReprojectionLinkedList = Buffer::createStructured(mpDevice, sizeof(float4) + sizeof(int) + sizeof(uint), mMaxPhotonCount);
        mpReprojectionLinkedList->setName("PM::LinkedList");
    }
    if (!mpHeadCounter)
    {
        uint2 frameDim = renderData.getDefaultTextureDims();
        mpHeadCounter = Texture::create2D(mpDevice, frameDim.x , frameDim.y, ResourceFormat::R32Uint, 1U, 1, nullptr, ResourceBindFlags::AllColorViews);
        mpHeadCounter->setName("PM::HeadCounter");
    }
    renderContext->clearUAV(mpHeadCounter->getUAV().get(), uint4(-1));
}

void ComplexLuminairesReSTIR_PT::prepareAccelerationStructure()
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

void ComplexLuminairesReSTIR_PT::buildAccelerationStructure(RenderContext* pRenderContext, const RenderData& renderData)
{
    pRenderContext->uavBarrier(mpPhotonCounter.get());
    pRenderContext->uavBarrier(mpPhotonAABBs.get());
    pRenderContext->uavBarrier(mpDirectVPLBuffer.get());
    uint currentPhotons = mFrameCount > 0 ? uint(float(mDispatchedPhotons) * 1.15f) : mMaxPhotonCount;
    std::vector<uint64_t> photonBuildSize = {std::min(mMaxPhotonCount, currentPhotons)};
    mpPhotonAS->update(pRenderContext, photonBuildSize);
}

void ComplexLuminairesReSTIR_PT::prepareRayTracingShader(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    //CL
    mGenerateSamplesPass.initRTProgram(mpDevice, mpScene, kGenerateSamplesShader, kMaxPayloadBytes, globalTypeConformances);
    mDebugPass.initRTCollectionProgram(mpDevice, mpScene, kDebugPass, kMaxPayloadBytes, globalTypeConformances);
    mDirectIlluminationReferencePass.initRTCollectionProgram(mpDevice, mpScene, kDirectIlluminationReferencePass, kMaxPayloadBytes, globalTypeConformances);
    //PT
    mPathTracingPass.initRTProgram(mpDevice, mpScene, kPathTracingPass, kMaxPayloadBytes, globalTypeConformances);
    //ReSTIR
    mSamplePass.initRTProgram(mpDevice, mpScene, kSampleShader, kMaxPayloadBytes, globalTypeConformances);
    mPathResamplePass.initRTProgram(mpDevice, mpScene, kPathResamplePass, kMaxPayloadBytes, globalTypeConformances);
}

void ComplexLuminairesReSTIR_PT::setSceneData(const RenderData& renderData, const ShaderVar& var)
{
    auto sceneDataVar = var["sdh"];
    sceneDataVar["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    sceneDataVar["gView"] = renderData[kInputView]->asTexture();
    sceneDataVar["gMVec"] = renderData[kInputMVec]->asTexture();
    sceneDataVar["gPrevVBufferWrite"] = mpPrevVBuffers[mFrameCount % 2];
    sceneDataVar["gPrevViewWrite"] = mpPrevViews[mFrameCount % 2];
    sceneDataVar["gPrevVBuffer"] = mpPrevVBuffers[(mFrameCount + 1) % 2];
    sceneDataVar["gPrevView"] = mpPrevViews[(mFrameCount + 1) % 2];
}

void ComplexLuminairesReSTIR_PT::setSampleData(const RenderData& renderData, const ShaderVar& var)
{
    auto samplerVar = var["vplSampler"];
    samplerVar["SampleBuffer"]["gCosOpeningAngle"] = mCosOpeningAngle;
    samplerVar["SampleBuffer"]["gPenumbraAngle"] = mPenumbraAngle;
    samplerVar["SampleBuffer"]["gPhotonCount"] = mDispatchedPhotons;
    samplerVar["SampleBuffer"]["gPointLightRadius"] = mPointLightRadius;
    samplerVar["gDirectVPLBuffer"] = mpDirectVPLBuffer;
    samplerVar["gPhotonCounter"] = mpPhotonCounter;
    samplerVar["gIndirectVPLBuffer"] = mpIndirectVPLBuffer;
    samplerVar["gOutDebug"] = renderData[kOutputDebug]->asTexture();
}

void ComplexLuminairesReSTIR_PT::getPhotonCount(RenderContext* pRenderContext)
{
    // Copy the Counters to a CPU Buffer
    pRenderContext->copyBufferRegion(mpPhotonCounterCPU.get(), 0, mpPhotonCounter.get(), 0, sizeof(uint32_t));
    void* data = mpPhotonCounterCPU->map(Buffer::MapType::Read);
    std::memcpy(&mStoredPhotons, data, sizeof(uint));
    mpPhotonCounterCPU->unmap();

    pRenderContext->copyBufferRegion(mpLinkedListCounterCPU.get(), 0, mpLinkedListCounter.get(), 0, sizeof(uint32_t));
    data = mpLinkedListCounterCPU->map(Buffer::MapType::Read);
    std::memcpy(&mLinkedListEntries, data, sizeof(uint));
    mpLinkedListCounterCPU->unmap();
}

void ComplexLuminairesReSTIR_PT::preparePathDebugBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPathDebugBuffer[0] || !mpPathDebugBuffer[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpPathDebugBuffer[i] = Buffer::createStructured(
                mpDevice, 36 * sizeof(float), mScreenRes.x * mScreenRes.y,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpPathDebugBuffer[i]->setName("PT::PathDebugBuffer" + std::to_string(i));
        }
    }
}

void ComplexLuminairesReSTIR_PT::preparePathTracingPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PreparePathTracingPass");

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mPathTracingPass.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mPathTracingPass.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_VIRTUAL_POINT_LIGHTS", mUseDirectVPLs || mUseIndirectVPLs ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_DIRECT_POINT_LIGHTS", mUseDirectVPLs ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_INDIRECT_POINT_LIGHTS", mUseIndirectVPLs ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_BSDF_SAMPLES", mUseBSDFSamples ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_NEE", mUseNEE ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("USE_MIS", mUseMIS ? "1" : "0");
    mPathTracingPass.pProgram->addDefine("MODE", std::to_string(mMode));

    if (!mPathTracingPass.pVars)
    {
        if (mpEmissiveLightSampler)
            mPathTracingPass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());
        mPathTracingPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }

    // Set constants.
    auto var = mPathTracingPass.pVars->getRootVar();
    mpEmissiveLightSampler->setShaderData(var["LightCB"]["gEmissiveLightSampler"]);
    mpSampleGenerator->setShaderData(var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
    setReservoirData(renderData, var);
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gMinConnectionDistance"] = mMinConnectionDistance;
    var["CB"]["gRoughnessThreshold"] = mRoughnessThreshold;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gLinkedList"] = mpReprojectionLinkedList;
    var["gHeadCounter"] = mpHeadCounter;
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gPathReservoir"] = mpPathReservoirs[mFrameCount % 2];
    var["gReconnectionData"] = mpReconnectionData[mFrameCount % 2];
    var["gPathDebugBuffer"] = mpPathDebugBuffer[0];
    var["gNEESampleBuffer"] = mpNEESamples[mFrameCount % 2];
}

void ComplexLuminairesReSTIR_PT::setReservoirData(const RenderData& renderData, const ShaderVar& var)
{
    auto reservoirDataVar = var["prh"];
    reservoirDataVar["CB"]["gFrameDim"] = mScreenRes;
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

void ComplexLuminairesReSTIR_PT::prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareGenerateSamplesPass");

    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpLinkedListCounter->getUAV().get(), uint4(0));

    mGenerateSamplesPass.pProgram->addDefine("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");
    mGenerateSamplesPass.pProgram->addDefine("USE_DIRECT_POINT_LIGHTS", mUseDirectVPLs ? "1" : "0");
    mGenerateSamplesPass.pProgram->addDefine("USE_INDIRECT_POINT_LIGHTS", mUseIndirectVPLs ? "1" : "0");
    mGenerateSamplesPass.pProgram->addDefine("USE_BACKPROJECTION", mUseBackprojection ? "1" : "0");
    mGenerateSamplesPass.pProgram->addDefine("PHOTON_BUFFER_SIZE_GLOBAL", std::to_string(mMaxPhotonCount));
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
    uint2 frameDim = renderData.getDefaultTextureDims();
    const auto& cameraData = mpScene->getCamera()->getData();
    setSampleData(renderData, var);
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["PerFrame"]["gFrameDim"] = frameDim;
    var["CB"]["gFlags"] = flags;
    var["CB"]["gMaxRecursion"] = mMaxRecursion;
    var["CB"]["gAABBSize"] = mAABBSize;
    var["CB"]["gNormalizedPixelArea"] = getNormalizedPixelSize(frameDim, focalLengthToFovY(cameraData.focalLength, cameraData.frameHeight), cameraData.aspectRatio);
    var["CB"]["gPixelWidthHeight"] = getPixelWidthHeight(frameDim, focalLengthToFovY(cameraData.focalLength, cameraData.frameHeight), cameraData.aspectRatio);
    var["gDirectVPLBuffer"] = mpDirectVPLBuffer;
    var["gIndirectVPLBuffer"] = mpIndirectVPLBuffer;
    var["gPhotonAABBs"] = mpPhotonAABBs;
    var["gPhotonCounter"] = mpPhotonCounter;
    var["gLinkedListCounter"] = mpLinkedListCounter;
    var["gLinkedList"] = mpReprojectionLinkedList;
    var["gHeadCounter"] = mpHeadCounter;
}

void ComplexLuminairesReSTIR_PT::prepareDirectIlluminationPass(RenderContext* pRenderContext, const RenderData& renderData)
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
    var["CB"]["gLuminaireSampleCount"] = mDispatchedPhotons;
    var["CB"]["gPhotonCount"] = mDispatchedPhotons;
    var["CB"]["gConeExponent"] = mConeExponent;
    var["CB"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["CB"]["gPenumbraAngle"] = mPenumbraAngle;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    var["gDirectVPLBuffer"] = mpDirectVPLBuffer;
}

void ComplexLuminairesReSTIR_PT::prepareDirectIlluminationReferencePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareReferencePass");
    if (!mDirectIlluminationReferencePass.pVars)
    {
        mDirectIlluminationReferencePass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mDirectIlluminationReferencePass.pVars->getRootVar();

    uint flags = 0;
    var["CB"]["gFlags"] = flags;
    var["CB"]["gPhotonCount"] = mDispatchedPhotons;
    var["CB"]["gPhotonRadius"] = mAABBSize;
    var["gPhotonAABBs"] = mpPhotonAABBs;
    var["gDirectVPLBuffer"] = mpDirectVPLBuffer;
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    setSceneData(renderData, var);

    mpPhotonAS->bindTlas(var, "gPhotonAccelerationStructure");
}

void ComplexLuminairesReSTIR_PT::prepareDebugPass(RenderContext* pRenderContext, const RenderData& renderData)
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

void ComplexLuminairesReSTIR_PT::prepareSamplePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PrepareSamplePass");

    if (!mSamplePass.pVars)
    {
        if (mpEmissiveLightSampler)
            mSamplePass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());
        mSamplePass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mSamplePass.pVars->getRootVar();
    mpEmissiveLightSampler->setShaderData(var["LightCB"]["gEmissiveLightSampler"]);
    mpSampleGenerator->setShaderData(var);
    var["UI"]["gNumberOfLightSamples"] = mNumberOfLightSamples;
    var["UI"]["gNumberOfBSDFSamples"] = mNumberOfBSDFSamples;
    var["UI"]["gNumberOfLuminaireSamples"] = mNumberOfLuminaireSamples;
    var["UI"]["gLuminaireSampleCount"] = mDispatchedPhotons;
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gOutDebug1"] = renderData[kOutputDebug1]->asTexture();
    var["CameraData"]["gPrevCamViewProjection"] = mTemporalCameraViewProjection;
    var["CameraData"]["gPrevCamPos"] = mTemporalCameraPos;
    var["CameraData"]["gPrevCamForward"] = mTemporalCameraForward;
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gLuminaireSamples"] = mpDirectVPLBuffer;
    setReservoirData(renderData, var);
    setSampleData(renderData, var);
    setSceneData(renderData, var);
}

void ComplexLuminairesReSTIR_PT::prepareResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PrepareResamplePass");
    if (!mpResamplePass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kResampleShader).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines; 
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpEmissiveLightSampler->getDefines());
        mpResamplePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    auto var = mpResamplePass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gCausticReservoirPrev"] = mpCausticReservoirs[(mFrameCount + 1) % 2];
    var["gPathReservoir"] = mpPathReservoirs[mFrameCount % 2];
    var["gPathReservoirPrev"] = mpPathReservoirs[(mFrameCount + 1) % 2];
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["UI"]["gRejectionAngle"] = mAngleThreshold;
    var["UI"]["gRejectionDistance"] = mDistanceThreshold;
    var["UI"]["gPixelRadius"] = mSpatialSampleRadius;
    var["UI"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["UI"]["gPenumbraAngle"] = mPenumbraAngle;
    var["UI"]["gMinConnectionDistance"] = mMinConnectionDistance;
    var["UI"]["gRoughnessThreshold"] = mRoughnessThreshold;
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
    FALCOR_ASSERT(mpResamplePass);
}

void ComplexLuminairesReSTIR_PT::preparePathResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PreparePathResamplePass");

    if (!mPathResamplePass.pVars)
    {
        mPathResamplePass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }

    auto var = mPathResamplePass.pVars->getRootVar();
    mpSampleGenerator->setShaderData(var);
    var["gPathDebugBuffer"] = mpPathDebugBuffer[1];
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gCausticReservoirPrev"] = mpCausticReservoirs[(mFrameCount + 1) % 2];
    var["gPathReservoir"] = mpPathReservoirs[mFrameCount % 2];
    var["gPathReservoirPrev"] = mpPathReservoirs[(mFrameCount + 1) % 2];
    var["gReconnectionData"] = mpReconnectionData[mFrameCount % 2];
    var["gReconnectionDataPrev"] = mpReconnectionData[(mFrameCount + 1) % 2];
    var["gNEESampleBuffer"] = mpNEESamples[mFrameCount % 2];
    var["gNEESampleBufferPrev"] = mpNEESamples[(mFrameCount + 1) % 2];
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gOutDebug1"] = renderData[kOutputDebug1]->asTexture();
    var["gOutColor"] = renderData[kOutputColor]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["UI"]["gPixelRadius"] = mSpatialSampleRadius;
    var["UI"]["gMinConnectionDistance"] = mMinConnectionDistance;
    var["UI"]["gRoughnessThreshold"] = mRoughnessThreshold;
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
}

void ComplexLuminairesReSTIR_PT::prepareCombinePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PrepareCombinePass");
    if (!mpCombinePass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kCombineShader).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines; 
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpEmissiveLightSampler->getDefines());
        mpCombinePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    auto var = mpCombinePass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
    setReservoirData(renderData, var);
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gOutputColor"] = renderData[kOutputColor]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    FALCOR_ASSERT(mpCombinePass);
}

void ComplexLuminairesReSTIR_PT::prepareSplattingCombinePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PrepareCombinePass");
    if (!mpSplatCombinePass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderSplatCombine).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines; 
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpEmissiveLightSampler->getDefines());
        mpSplatCombinePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    auto var = mpSplatCombinePass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
    setReservoirData(renderData, var);
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gSplattingHits"] = mpSplattingHits;
    var["gOutputColor"] = renderData[kOutputColor]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    FALCOR_ASSERT(mpSplatCombinePass);
}

void ComplexLuminairesReSTIR_PT::prepareReservoirs(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint reservoirSize = mScreenRes.x * mScreenRes.y;
    if (!mpCausticReservoirs[0] || !mpCausticReservoirs[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpCausticReservoirs[i] = Buffer::createStructured(
                mpDevice, sizeof(float3) + sizeof(float4) + 2 * sizeof(uint), reservoirSize,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpCausticReservoirs[i]->setName("ReSTIR::CausticReservoir" + std::to_string(i));
        }
    }

    if (!mpPathReservoirs[0] || !mpPathReservoirs[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpPathReservoirs[i] = Buffer::createStructured(
                mpDevice, 12 * sizeof(float), reservoirSize,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpPathReservoirs[i]->setName("ReSTIR::PathReservoir" + std::to_string(i));
        }
    }
}

void ComplexLuminairesReSTIR_PT::prepareNEEBuffer(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpNEESamples[0] || !mpNEESamples[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpNEESamples[i] = Buffer::createStructured(
                mpDevice, 12 * sizeof(float), mScreenRes.x * mScreenRes.y,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpNEESamples[i]->setName("ReSTIR::NEESampleBuffer" + std::to_string(i));
        }
    }
}

void ComplexLuminairesReSTIR_PT::prepareReconnectionData(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpReconnectionData[0] || !mpReconnectionData[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpReconnectionData[i] = Buffer::createStructured(
                mpDevice, 24 * sizeof(float), mScreenRes.x * mScreenRes.y,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpReconnectionData[i]->setName("ReSTIR::ReconnectionData" + std::to_string(i));
        }
    }
}

void ComplexLuminairesReSTIR_PT::prepareSceneData(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPrevVBuffers[0] || !mpPrevVBuffers[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpPrevVBuffers[i] = Texture::create2D(
                    mpDevice, mScreenRes.x, mScreenRes.y, ResourceFormat::RGBA32Uint, 1u, Texture::kMaxPossible,
                    nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
            mpPrevVBuffers[i]->setName("ReSTIR::PrevVBuffer" + std::to_string(i));
        }
    }
    if (!mpPrevViews[0] || !mpPrevViews[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpPrevViews[i] = Texture::create2D(
                    mpDevice, mScreenRes.x, mScreenRes.y, ResourceFormat::RGBA32Float, 1u, Texture::kMaxPossible,
                    nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
            mpPrevViews[i]->setName("ReSTIR::PrevViews" + std::to_string(i));
        }
    }
}

void ComplexLuminairesReSTIR_PT::sample(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::sample");
    mpScene->raytrace(pRenderContext,mSamplePass.pProgram.get(),mSamplePass.pVars, uint3(dispatchSize, 1));
}

void ComplexLuminairesReSTIR_PT::resample(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::resample");
    mpResamplePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}

void ComplexLuminairesReSTIR_PT::combine(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::combine");
    mpCombinePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}

float ComplexLuminairesReSTIR_PT::getNormalizedPixelArea()
{
    // Update Image plane distance
    auto& cameraData = mpScene->getCamera()->getData();
    float fovY = focalLengthToFovY(cameraData.focalLength, cameraData.frameHeight);

    // Get normalized pixel area
    float h = tan(fovY / 2.f) * 2.f;
    float w = h * cameraData.aspectRatio;
    float wPix = w / mScreenRes.x;
    float hPix = h / mScreenRes.y;

    return wPix * hPix;
}

void ComplexLuminairesReSTIR_PT::updateScreenData(const RenderData& renderData)
{
    auto& screenDims = renderData.getDefaultTextureDims();
    if (screenDims.x != mScreenRes.x || screenDims.y != mScreenRes.y)
    {
        mScreenRes = screenDims;
        mNormalizedPixelArea = getNormalizedPixelArea();
    }
}

void ComplexLuminairesReSTIR_PT::prepareSplattingData(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint2 screenRes = renderData.getDefaultTextureDims();

    if (!mpSplattingGlobalCounter)
    {
        mpSplattingGlobalCounter = Buffer::createStructured(
            mpDevice, sizeof(uint), 2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None,
            nullptr, false
        );
        mpSplattingGlobalCounter->setName("SplattingGlobalCounter");
    }

    if (!mpSplattingCellCounter)
    {
        mpSplattingCellCounter = Buffer::createStructured(
            mpDevice, sizeof(uint), screenRes.x * screenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpSplattingCellCounter->setName("SplattingCellCounter");
    }

    if (!mpSplattingCellOffsets)
    {
        mpSplattingCellOffsets = Buffer::createStructured(
            mpDevice, sizeof(uint), screenRes.x * screenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpSplattingCellOffsets->setName("SplattingCellOffsets");
    }

    if (!mpSplattingSortingData)
    {
        mpSplattingSortingData = Buffer::createStructured(
            mpDevice, sizeof(uint4), screenRes.x * screenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpSplattingSortingData->setName("SplattingSortingData");
    }

    if (!mpSplattingSortedReservoirs)
    {
        mpSplattingSortedReservoirs = Buffer::createStructured(
            mpDevice, sizeof(uint2), screenRes.x * screenRes.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
        mpSplattingSortedReservoirs->setName("SplattingSortedReservoirs");
    }

    if (!mpSplattingHits)
    {
        auto vBuffer = renderData[kInputVBuffer]->asTexture();
        mpSplattingHits = Texture::create2D(
            mpDevice, mScreenRes.x, mScreenRes.y, vBuffer->getFormat(), 1u, 1u,
            nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        mpSplattingSortedReservoirs->setName("SplattingHit");
    }

    if (mFrameCount == 0)
    {
        const CameraData& camData = mpScene->getCamera()->getData();
        mTemporalCameraViewProjection = camData.viewProjMat;
        mTemporalCameraPos = camData.posW;
        mTemporalCameraForward = math::normalize(camData.cameraW);
    }
}

void ComplexLuminairesReSTIR_PT::prepareTemporalSplattingPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Splat Caustic Reservoirs");

    pRenderContext->clearUAV(mpSplattingGlobalCounter->getUAV(0).get(), uint4(0));
    pRenderContext->clearUAV(mpSplattingCellCounter->getUAV(0).get(), uint4(0));
    pRenderContext->clearUAV(mpSplattingCellOffsets->getUAV(0).get(), uint4(0));
   
    if (!mpTemporalSplatReservoirs)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderTemporalSplatReservoirs).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());

        mpTemporalSplatReservoirs = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpTemporalSplatReservoirs);

    // Set variables
    auto var = mpTemporalSplatReservoirs->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var); // Set scene data

    var["CB"]["gFrameDim"] = mScreenRes;

    var["gCellCounter"] = mpSplattingCellCounter;
    var["gGlobalCounter"] = mpSplattingGlobalCounter;
    var["gSplatSortData"] = mpSplattingSortingData;

    var["gSplattingHits"] = mpSplattingHits;

    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gOutDebug1"] = renderData[kOutputDebug1]->asTexture();
    setSceneData(renderData, var);
}

void ComplexLuminairesReSTIR_PT::prepareSortSplattingDataPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Sort Splatted Reservoirs");

    //Init Shaders
    if (!mpSplatSortComputeCellOffsets)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderSortSplatReservoirs).csEntry("computeCellOffsets").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());

        mpSplatSortComputeCellOffsets = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpSplatSortComputeCellOffsets);
    if (!mpSplatSortCellData)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderSortSplatReservoirs).csEntry("sortCellData").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add("USE_ENV_BACKROUND", mpScene->useEnvBackground() ? "1" : "0");

        mpSplatSortCellData = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpSplatSortCellData);

    //Lambda for shader vars as they are the same for both shaders
    auto setProgramVars = [&](ShaderVar& var)
    {
        mpScene->setRaytracingShaderData(pRenderContext, var); // Set scene data
        var["CB"]["gFrameDim"] = mScreenRes;

        var["gGlobalCounter"] = mpSplattingGlobalCounter;
        var["gCellCounter"] = mpSplattingCellCounter;
        var["gCellOffsets"] = mpSplattingCellOffsets;
        var["gSortingData"] = mpSplattingSortingData;
        var["gSortedReservoirs"] = mpSplattingSortedReservoirs;
    };

    pRenderContext->uavBarrier(mpSplattingGlobalCounter.get());
    auto var = mpSplatSortComputeCellOffsets->getRootVar();
    setProgramVars(var);

    var = mpSplatSortCellData->getRootVar();
    setProgramVars(var);
}

void ComplexLuminairesReSTIR_PT::prepareSplattingResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::PrepareSplattingResamplePass");
    if (!mpSplatResamplePass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderSplatResample).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines; 
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(mpEmissiveLightSampler->getDefines());
        mpSplatResamplePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    auto var = mpSplatResamplePass->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);
    mpSampleGenerator->setShaderData(var);
    var["gCausticReservoir"] = mpCausticReservoirs[mFrameCount % 2];
    var["gCausticReservoirPrev"] = mpCausticReservoirs[(mFrameCount + 1) % 2];
    var["gSplattedPixels"] = mpSplattingSortedReservoirs;
    var["gCellCounters"] = mpSplattingCellCounter;
    var["gCellOffsets"] = mpSplattingCellOffsets;
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["PerFrame"]["gNormalizedPixelArea"] = mNormalizedPixelArea;
    var["UI"]["gRejectionAngle"] = mAngleThreshold;
    var["UI"]["gRejectionDistance"] = mDistanceThreshold;
    var["UI"]["gPixelRadius"] = mSpatialSampleRadius;
    var["UI"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["UI"]["gPenumbraAngle"] = mPenumbraAngle;
    var["CameraData"]["gPrevCamViewProjection"] = mTemporalCameraViewProjection;
    var["CameraData"]["gPrevCamPos"] = mTemporalCameraPos;
    var["CameraData"]["gPrevCamForward"] = mTemporalCameraForward;
    var["gSplattingHits"] = mpSplattingHits;
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["gOutDebug1"] = renderData[kOutputDebug1]->asTexture();
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    setSampleData(renderData, var);
    FALCOR_ASSERT(mpSplatResamplePass);
}

void ComplexLuminairesReSTIR_PT::reprojectPrevData(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Execute Compute Pass
    const uint2 targetDim = mScreenRes;
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpTemporalSplatReservoirs->execute(pRenderContext, uint3(targetDim, 1));
}

void ComplexLuminairesReSTIR_PT::sortSplattingData(RenderContext* pRenderContext, const RenderData& renderData)
{
        pRenderContext->uavBarrier(mpSplattingGlobalCounter.get());
        {
            const uint2 targetDim = mScreenRes;
            FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
            mpSplatSortComputeCellOffsets->execute(pRenderContext, uint3(targetDim, 1));
        }

        pRenderContext->uavBarrier(mpSplattingGlobalCounter.get());
        pRenderContext->uavBarrier(mpSplattingCellOffsets.get());

        {
            const uint targetDim = mScreenRes.x * mScreenRes.y;
            FALCOR_ASSERT(targetDim > 0);
            mpSplatSortCellData->execute(pRenderContext, uint3(targetDim, 1, 1));
        }
}

void ComplexLuminairesReSTIR_PT::resampleWithSplatting(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SplatResample");
    mpSplatResamplePass->execute(pRenderContext, mScreenRes.x, mScreenRes.y);
}

void ComplexLuminairesReSTIR_PT::combineWithSplatting(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SplatCombine");
    mpSplatCombinePass->execute(pRenderContext, mScreenRes.x, mScreenRes.y);
}

void ComplexLuminairesReSTIR_PT::directIllumiantionVPL(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIllumination");
    mpDirectIlluminationPass->execute(pRenderContext, launchDim.x, launchDim.y);
}

void ComplexLuminairesReSTIR_PT::directIlluminationReference(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIlluminationReference");
    mpScene->raytrace(pRenderContext, mDirectIlluminationReferencePass.pProgram.get(), mDirectIlluminationReferencePass.pVars, uint3(launchDim, 1));
}

//Main
void ComplexLuminairesReSTIR_PT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    auto& dict = renderData.getDictionary();

    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    uint lightPathsSq = static_cast<uint>(std::floor(std::sqrt(mMaxPhotonCount)));
    mDispatchedPhotons = lightPathsSq * lightPathsSq;

    updateScreenData(renderData);
    prepareLight(pRenderContext, renderData);

    //prepareResources
    prepareDirectVPLBuffer(pRenderContext, renderData);
    prepareIndirectVPLBuffer(pRenderContext, renderData);
    preparePhotonAABBBuffer(pRenderContext, renderData);
    prepareCounter(pRenderContext, renderData);
    prepareAccelerationStructure();
    prepareLinkedList(pRenderContext, renderData);

    //prepareShaders
    prepareGenerateSamplesPass(pRenderContext, renderData);
    mChangedPhotonBufferSize = false;

    //generateSamples
    uint3 dispatchDims = uint3(lightPathsSq, lightPathsSq, 1u);
    mpScene->raytrace(pRenderContext,mGenerateSamplesPass.pProgram.get(),mGenerateSamplesPass.pVars, dispatchDims);
    getPhotonCount(pRenderContext);
    buildAccelerationStructure(pRenderContext, renderData);

    uint2 launchDim = mScreenRes;
    //Pass for simple direct illumination
    switch (mMode)
    {
    case 0:
        preparePathTracingPass(pRenderContext, renderData);
        //create paths
        mpScene->raytrace(pRenderContext,mPathTracingPass.pProgram.get(),mPathTracingPass.pVars, uint3(mScreenRes, 1));
        break;
    case 1:
        prepareSceneData(pRenderContext, renderData);
        prepareReservoirs(pRenderContext, renderData);
        prepareNEEBuffer(pRenderContext, renderData);
        prepareReconnectionData(pRenderContext, renderData);
        preparePathDebugBuffer(pRenderContext, renderData);
        preparePathResamplePass(pRenderContext, renderData);
        preparePathTracingPass(pRenderContext, renderData);
        //create paths
        mpScene->raytrace(pRenderContext,mPathTracingPass.pProgram.get(),mPathTracingPass.pVars, uint3(mScreenRes, 1));
        //resample paths
        mpScene->raytrace(pRenderContext,mPathResamplePass.pProgram.get(),mPathResamplePass.pVars, uint3(mScreenRes, 1));
        break;
    default:
        break;
    }

    //Debugpass for displaying dispatched photons
    if (mRenderPhotons)
    {
        prepareDebugPass(pRenderContext, renderData);
        FALCOR_PROFILE(pRenderContext, "DebugPass");
        mpScene->raytrace(pRenderContext, mDebugPass.pProgram.get(),mDebugPass.pVars, uint3(launchDim, 1));
    }
    ++mFrameCount;
}

//user interface
void ComplexLuminairesReSTIR_PT::renderUI(Gui::Widgets& widget)
{
    if (auto vplGroup= widget.group("VPLs"))
    {
        widget.text("Stored Photons: " + std::to_string(mStoredPhotons) + "/ " + std::to_string(mDispatchedPhotons));
        widget.text("Reprojection linked list entries: " + std::to_string(mLinkedListEntries));
        mChangedPhotonBufferSize |= widget.var("Number of Photons", mMaxPhotonCount, 1u, 10000000u);
        mOptionsChanged |= widget.var("Recursion Depth", mMaxRecursion, 0u, 50u);
        mOptionsChanged |= widget.var("Cos Opening Angle", mCosOpeningAngle, 0.f, 1.f, 0.001f, false, "%.6f");
        mOptionsChanged |= widget.var("Penumbra Angle", mPenumbraAngle, 0.f, mCosOpeningAngle);
        mOptionsChanged |= widget.var("Photon AABB Size", mAABBSize, 0.f, 1.f);
        mOptionsChanged |= widget.var("Point Light Radius", mPointLightRadius, 0.f, 1.f);
        mOptionsChanged |= widget.checkbox("Show Photons", mRenderPhotons);
        mOptionsChanged |= widget.dropdown("Mode", kModes, mMode);
        mOptionsChanged |= mChangedPhotonBufferSize;
    }
    if (auto restirGroup = widget.group("ReSTIR"))
    {
        if (auto sampleGroup = widget.group("Sample Generation"))
        {
            widget.var("Number of Light Samples", mNumberOfLightSamples, 0u, 1024u);
            widget.var("Number of BSDF Samples", mNumberOfBSDFSamples, 0u, 1024u);
            widget.var("Number of Luminaire Samples", mNumberOfLuminaireSamples, 0u, 1024u);
        }
        if (auto resampleGroup = widget.group("Resampling"))
        {
            widget.tooltip("Radius for spatial samples in pixels");
            widget.var("Spatial radius", mSpatialSampleRadius, 0u, 1024u);
            widget.var("Hybrid Shift connection distance threshold", mMinConnectionDistance, 0.0f, 1.0f, 0.001f);
            widget.var("roughness threshold", mRoughnessThreshold, 0.0f, 1.0f, 0.001f);
        }
    }
    if (auto ptGroup = widget.group("PT"))
    {
        mOptionsChanged |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
        widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

        mOptionsChanged |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
        widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

        mOptionsChanged |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
        widget.tooltip("Use importance sampling for materials", true);

        mOptionsChanged |= widget.checkbox("Use direct VPLs", mUseDirectVPLs);
        widget.tooltip("Use VPLs to approximate direct light of complex luminaire", true);

        mOptionsChanged |= widget.checkbox("Use indirect VPLs", mUseIndirectVPLs);
        widget.tooltip("Use VPLs to approximate indirect light of complex luminaire", true);

        mOptionsChanged |= widget.checkbox("Use BackProjection", mUseBackprojection);
        widget.tooltip("Use photon backprojection for indirect light of complex luminaire", true);

        mOptionsChanged |= widget.checkbox("Use BSDF samples", mUseBSDFSamples);
        widget.tooltip("Collect light on hit surfaces", true);

        mOptionsChanged |= widget.checkbox("NEE", mUseNEE);
        widget.tooltip("Conduct next event estimation on every hit", true);

        mOptionsChanged |= widget.checkbox("Use MIS", mUseMIS);
        widget.tooltip("Use MIS for combining BSDF and NEE samples", true);
    }
    if (auto debugGroup = widget.group("Debug"))
    {
        if (widget.button("Reset frame count"))
            mFrameCount = 0;
        if (widget.button("Clear Reservoirs"))
        {
            mpCausticReservoirs[0] = nullptr;
            mpCausticReservoirs[1] = nullptr;
            mpPathReservoirs[0] = nullptr;
            mpPathReservoirs[1] = nullptr;
        }
        if (widget.button("Freeze"))
            mMode = 69;
    }
}

void ComplexLuminairesReSTIR_PT::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mFrameCount = 0;
    //Reset Scene
    mpScene = pScene;

    mGenerateSamplesPass = RayTraceProgramHelper::create();
    mSamplePass = RayTraceProgramHelper::create();
    mDebugPass = RayTraceProgramHelper::create();
    mDirectIlluminationReferencePass = RayTraceProgramHelper::create();
    mPathTracingPass = RayTraceProgramHelper::create();
    mPathResamplePass = RayTraceProgramHelper::create();
    mpDirectIlluminationPass.reset();
    mpEmissiveLightSampler.reset();
    mpResamplePass.reset();
    mpCombinePass.reset();
    mpEmissiveLightSampler.reset();
    mpTemporalSplatReservoirs.reset();
    mpSplatSortComputeCellOffsets.reset();
    mpSplatSortCellData.reset();
    mpSplatResamplePass.reset();
    mpSplatCombinePass.reset();
    if (mpPhotonAS)
        mpPhotonAS->clearAABBBuffers(pRenderContext, mpPhotonAABBs);

    if (mpScene)
    {
        const auto& bounds= mpScene->getSceneBounds();
        const float sceneExtend = math::length(bounds.extent());

        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }
        prepareRayTracingShader(pRenderContext);
        mNormalizedPixelArea = getNormalizedPixelArea();
    }
}

void ComplexLuminairesReSTIR_PT::RayTraceProgramHelper::initRTProgram(ref<Device> device,ref<Scene> scene,const std::string& shaderName,
                                                     uint maxPayloadBytes,const Program::TypeConformanceList& globalTypeConformances)
{
    RtProgram::Desc desc;
    desc.addShaderModules(scene->getShaderModules());
    desc.addShaderLibrary(shaderName);
    desc.setShaderModel(kShaderModel);
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

void ComplexLuminairesReSTIR_PT::RayTraceProgramHelper::initRTCollectionProgram(
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

void ComplexLuminairesReSTIR_PT::RayTraceProgramHelper::initProgramVars(ref<Device> pDevice,ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
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
