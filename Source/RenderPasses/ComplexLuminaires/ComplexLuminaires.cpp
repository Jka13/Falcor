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
    //CL
    const std::string kGenerateSamplesShader = "RenderPasses/ComplexLuminaires/Shader/GenerateSamples.rt.slang";
    const std::string kDirectIlluminationPass = "RenderPasses/ComplexLuminaires/Shader/DirectIllumination.cs.slang";
    const std::string kDirectIlluminationReferencePass = "RenderPasses/ComplexLuminaires/Shader/DirectIlluminationReference.rt.slang";
    const std::string kDebugPass = "RenderPasses/ComplexLuminaires/Shader/Debug.rt.slang";

    //ReSTIR
    const std::string kSampleShader = "RenderPasses/ComplexLuminaires/Shader/Sample.rt.slang";
    const std::string kResampleShader = "RenderPasses/ComplexLuminaires/Shader/Resample.cs.slang";
    const std::string kCombineShader = "RenderPasses/ComplexLuminaires/Shader/Combine.cs.slang";

    //Splatting
    const std::string kShaderTemporalSplatReservoirs = "RenderPasses/ComplexLuminaires/Shader/TemporalSplatReservoir.cs.slang";
    const std::string kShaderSplatResample = "RenderPasses/ComplexLuminaires/Shader/SplattingResample.cs.slang";
    const std::string kShaderSortSplatReservoirs = "RenderPasses/ComplexLuminaires/Shader/SortSplatReservoirs.cs.slang";

    const std::string kShaderModel = "6_5";
    const uint kMaxPayloadBytes = 96u;

    const std::string kOutputColor = "color";
    const std::string kOutputDebug = "debug";
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
    };
    const Gui::DropdownList kModes{{0, "VPL"}, {1, "Reference"}, {2, "ReSTIR"}, {3, "ReSTIR Splatting"}};
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
            mpDevice, 3 * sizeof(float3), mMaxPhotonCount,
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
        mpPhotonAABBs = Buffer::createStructured(mpDevice, sizeof(AABB), mMaxPhotonCount);
        mpPhotonAABBs->setName("PM::PhotonAABB");
    }
}

void ComplexLuminaires::preparePhotonCounter(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpPhotonCounter)
    {
        mpPhotonCounter = Buffer::create(mpDevice, sizeof(uint) * 4);
        mpPhotonCounter->setName("PM::PhotonCounterGPU");
    }
    if (!mpPhotonCounterCPU)
    {
        mpPhotonCounterCPU = Buffer::create(mpDevice, sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpPhotonCounterCPU->setName("PM::PhotonCounter");
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
    pRenderContext->uavBarrier(mpPhotonCounter.get());
    pRenderContext->uavBarrier(mpPhotonAABBs.get());
    pRenderContext->uavBarrier(mpPhotonBuffer.get());
    uint currentPhotons = mFrameCount > 0 ? uint(float(mDispatchedPhotons) * 1.15f) : mMaxPhotonCount;
    std::vector<uint64_t> photonBuildSize = {std::min(mMaxPhotonCount, currentPhotons)};
    mpPhotonAS->update(pRenderContext, photonBuildSize);
}

void ComplexLuminaires::prepareRayTracingShader(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    //CL
    mGenerateSamplesPass.initRTProgram(mpDevice, mpScene, kGenerateSamplesShader, kMaxPayloadBytes, globalTypeConformances);
    mDebugPass.initRTCollectionProgram(mpDevice, mpScene, kDebugPass, kMaxPayloadBytes, globalTypeConformances);
    mDirectIlluminationReferencePass.initRTCollectionProgram(mpDevice, mpScene, kDirectIlluminationReferencePass, kMaxPayloadBytes, globalTypeConformances);
    //ReSTIR
    mSamplePass.initRTProgram(mpDevice, mpScene, kSampleShader, kMaxPayloadBytes, globalTypeConformances);
}

void ComplexLuminaires::setSceneData(const RenderData& renderData, const ShaderVar& var)
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

void ComplexLuminaires::getPhotonCount(RenderContext* pRenderContext)
{
    // Copy the photonCounter to a CPU Buffer
    pRenderContext->copyBufferRegion(mpPhotonCounterCPU.get(), 0, mpPhotonCounter.get(), 0, sizeof(uint32_t));

    void* data = mpPhotonCounterCPU->map(Buffer::MapType::Read);
    std::memcpy(&mDispatchedPhotons, data, sizeof(uint));
    mpPhotonCounterCPU->unmap();
}

void ComplexLuminaires::setReservoirData(const RenderData& renderData, const ShaderVar& var)
{
    auto reservoirDataVar = var["rh"];
    reservoirDataVar["CB"]["gFrameDim"] = mScreenRes;
}

void ComplexLuminaires::prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareGenerateSamplesPass");

    pRenderContext->clearUAV(mpPhotonCounter->getUAV().get(), uint4(0));

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
    var["gPhotonCounter"] = mpPhotonCounter;
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
    var["CB"]["gLuminaireSampleCount"] = mDispatchedPhotons;
    var["CB"]["gPhotonCount"] = mMaxPhotonCount;
    var["CB"]["gConeExponent"] = mConeExponent;
    var["CB"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["CB"]["gPenumbraAngle"] = mPenumbraAngle;
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
    var["CB"]["gPhotonCount"] = mMaxPhotonCount;
    var["CB"]["gPhotonRadius"] = mAABBSize;
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

void ComplexLuminaires::prepareSamplePass(RenderContext* pRenderContext, const RenderData& renderData)
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
    var["UI"]["gPhotonCount"] = mMaxPhotonCount;
    var["UI"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["UI"]["gPenumbraAngle"] = mPenumbraAngle;
    var["CameraData"]["gPrevCamViewProjection"] = mTemporalCameraViewProjection;
    var["CameraData"]["gPrevCamPos"] = mTemporalCameraPos;
    var["CameraData"]["gPrevCamForward"] = mTemporalCameraForward;
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["gReservoir"] = mpSampleReservoirs[mFrameCount % 2];
    var["gLuminaireSamples"] = mpPhotonBuffer;
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
}

void ComplexLuminaires::prepareResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
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
    var["gReservoir"] = mpSampleReservoirs[mFrameCount % 2];
    var["gReservoirPrev"] = mpSampleReservoirs[(mFrameCount + 1) % 2];
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["UI"]["gRejectionAngle"] = mAngleThreshold;
    var["UI"]["gRejectionDistance"] = mDistanceThreshold;
    var["UI"]["gPixelRadius"] = mSpatialSampleRadius;
    var["UI"]["gCosOpeningAngle"] = mCosOpeningAngle;
    var["UI"]["gPenumbraAngle"] = mPenumbraAngle;
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    FALCOR_ASSERT(mpResamplePass);
}

void ComplexLuminaires::prepareCombinePass(RenderContext* pRenderContext, const RenderData& renderData)
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
    setReservoirData(renderData, var);
    var["gReservoir"] = mpSampleReservoirs[mFrameCount % 2];
    var["gOutputColor"] = renderData[kOutputColor]->asTexture();
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    FALCOR_ASSERT(mpCombinePass);
}

void ComplexLuminaires::prepareReservoirs(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint reservoirSize = mScreenRes.x * mScreenRes.y;
    if (!mpSampleReservoirs[0] || !mpSampleReservoirs[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpSampleReservoirs[i] = Buffer::createStructured(
                mpDevice, 4 * sizeof(float3) + 3 * sizeof(float) + sizeof(uint), reservoirSize,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpSampleReservoirs[i]->setName("ReSTIR::Reservoir" + std::to_string(i));
        }
    }
}

void ComplexLuminaires::prepareSceneData(RenderContext* pRenderContext, const RenderData& renderData)
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

void ComplexLuminaires::sample(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::sample");
    mpScene->raytrace(pRenderContext,mSamplePass.pProgram.get(),mSamplePass.pVars, uint3(dispatchSize, 1));
}

void ComplexLuminaires::resample(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::resample");
    mpResamplePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}

void ComplexLuminaires::combine(RenderContext* pRenderContext, uint2 dispatchSize)
{
    FALCOR_PROFILE(pRenderContext, "ReSTIR::combine");
    mpCombinePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}

float ComplexLuminaires::getNormalizedPixelArea()
{
    if (!mpScene)
        return 1.0;

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

void ComplexLuminaires::updateScreenData(const RenderData& renderData)
{
    auto& screenDims = renderData.getDefaultTextureDims();
    if (screenDims.x != mScreenRes.x || screenDims.y != mScreenRes.y)
    {
        mScreenRes = screenDims;
        mNormalizedPixelArea = getNormalizedPixelArea();
    }
}

void ComplexLuminaires::prepareSplattingData(RenderContext* pRenderContext, const RenderData& renderData)
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
}

void ComplexLuminaires::prepareTemporalSplattingPass(RenderContext* pRenderContext, const RenderData& renderData)
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
    setSceneData(renderData, var);
}

void ComplexLuminaires::prepareSortSplattingDataPass(RenderContext* pRenderContext, const RenderData& renderData)
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

void ComplexLuminaires::prepareSplattingResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
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
    var["gReservoir"] = mpSampleReservoirs[mFrameCount % 2];
    var["gReservoirPrev"] = mpSampleReservoirs[(mFrameCount + 1) % 2];
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
    var["gOutDebug"] = renderData[kOutputDebug]->asTexture();
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    FALCOR_ASSERT(mpSplatResamplePass);
}

void ComplexLuminaires::reprojectPrevData(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Execute Compute Pass
    const uint2 targetDim = mScreenRes;
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    mpTemporalSplatReservoirs->execute(pRenderContext, uint3(targetDim, 1));
}

void ComplexLuminaires::sortSplattingData(RenderContext* pRenderContext, const RenderData& renderData)
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

void ComplexLuminaires::resampleWithSplatting(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "SplatResample");
    mpSplatResamplePass->execute(pRenderContext, mScreenRes.x, mScreenRes.y);
}


void ComplexLuminaires::directIllumiantionVPL(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIllumination");
    mpDirectIlluminationPass->execute(pRenderContext, launchDim.x, launchDim.y);
}

void ComplexLuminaires::directIlluminationReference(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim)
{
    FALCOR_PROFILE(pRenderContext, "DirectIlluminationReference");
    mpScene->raytrace(pRenderContext, mDirectIlluminationReferencePass.pProgram.get(), mDirectIlluminationReferencePass.pVars, uint3(launchDim, 1));
}

void ComplexLuminaires::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ++mFrameCount;
    if (!mpScene)
        return;

    auto& dict = renderData.getDictionary();

    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }
    updateScreenData(renderData);
    prepareLight(pRenderContext, renderData);
    //prepareResources
    preparePhotonBuffer(pRenderContext, renderData);
    preparePhotonAABBBuffer(pRenderContext, renderData);
    preparePhotonCounter(pRenderContext, renderData);
    prepareAccelerationStructure();

    //prepareShaders
    prepareGenerateSamplesPass(pRenderContext, renderData);
    mChangedPhotonBufferSize = false;

    //generateSamples
    mpScene->raytrace(pRenderContext,mGenerateSamplesPass.pProgram.get(),mGenerateSamplesPass.pVars, uint3(mMaxPhotonCount, 1, 1));
    getPhotonCount(pRenderContext);
    buildAccelerationStructure(pRenderContext, renderData);

    uint2 launchDim = mScreenRes;
    //Pass for simple direct illumination
    switch (mMode)
    {
    case 0:
        prepareDirectIlluminationPass(pRenderContext, renderData);
        directIllumiantionVPL(pRenderContext, renderData, launchDim);
        break;
    case 1:
        prepareDirectIlluminationReferencePass(pRenderContext, renderData);
        directIlluminationReference(pRenderContext, renderData, launchDim);
        break;
    case 2:
        prepareReservoirs(pRenderContext, renderData);
        prepareSceneData(pRenderContext, renderData);

        prepareSamplePass(pRenderContext, renderData);
        prepareResamplePass(pRenderContext, renderData);
        prepareCombinePass(pRenderContext, renderData);

        sample(pRenderContext, launchDim);
        resample(pRenderContext, launchDim);
        combine(pRenderContext, launchDim);
        break;
    case 3:
        prepareReservoirs(pRenderContext, renderData);
        prepareSceneData(pRenderContext, renderData);
        prepareSplattingData(pRenderContext, renderData);

        prepareSamplePass(pRenderContext, renderData);
        prepareSplattingResamplePass(pRenderContext, renderData);
        prepareCombinePass(pRenderContext, renderData);
        prepareTemporalSplattingPass(pRenderContext, renderData);
        prepareSortSplattingDataPass(pRenderContext, renderData);

        sample(pRenderContext, launchDim);
        reprojectPrevData(pRenderContext, renderData);
        sortSplattingData(pRenderContext, renderData);
        resampleWithSplatting(pRenderContext);
        combine(pRenderContext, launchDim);
        {
            //Copy Camera data for splatting
            const CameraData& camData = mpScene->getCamera()->getData();
            mTemporalCameraViewProjection = camData.viewProjMat;
            mTemporalCameraPos = camData.posW;
            mTemporalCameraForward = math::normalize(camData.cameraW);
        }
        break;
    default:
        break;
    }
    //Debugpass for displaying dispatched photons
    if (mShowDebug)
    {
        prepareDebugPass(pRenderContext, renderData);
        FALCOR_PROFILE(pRenderContext, "DebugPass");
        mpScene->raytrace(pRenderContext, mDebugPass.pProgram.get(),mDebugPass.pVars, uint3(launchDim, 1));
    }
}

void ComplexLuminaires::renderUI(Gui::Widgets& widget)
{
    widget.text("Dispatched Photons: " + std::to_string(mDispatchedPhotons) + "/ " + std::to_string(mMaxPhotonCount));
    mChangedPhotonBufferSize |= widget.var("Number of Photons", mMaxPhotonCount, 1u, 10000000u);
    mOptionsChanged |= widget.var("Recursion Depth", mMaxRecursion, 0u, 50u);
    mOptionsChanged |= widget.var("Cos Opening Angle", mCosOpeningAngle, 0.f, 1.f, 0.001f, false, "%.6f");
    mOptionsChanged |= widget.var("Penumbra Angle", mPenumbraAngle, 0.f, mCosOpeningAngle);
    mOptionsChanged |= widget.var("Photon AABB Size", mAABBSize, 0.f, 1.f);
    mOptionsChanged |= widget.checkbox("Show Photons", mShowDebug);
    mOptionsChanged |= widget.dropdown("Mode", kModes, mMode);
    mOptionsChanged |= mChangedPhotonBufferSize;
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
            widget.var("Angle rejection threshold", mAngleThreshold, 0.0f, 1.0f, 0.001f);
            widget.var("Distance rejection threshold", mDistanceThreshold, 0.0f, 1.0f, 0.001f);
        }
    }
}

void ComplexLuminaires::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    //Reset Scene
    mpScene = pScene;

    mGenerateSamplesPass = RayTraceProgramHelper::create();
    mSamplePass = RayTraceProgramHelper::create();
    mDebugPass = RayTraceProgramHelper::create();
    mDirectIlluminationReferencePass = RayTraceProgramHelper::create();
    mpDirectIlluminationPass.reset();
    mpEmissiveLightSampler.reset();
    mpResamplePass.reset();
    mpCombinePass.reset();
    mpEmissiveLightSampler.reset();
    mpTemporalSplatReservoirs.reset();
    mpSplatSortComputeCellOffsets.reset();
    mpSplatSortCellData.reset();
    mpSplatResamplePass.reset();
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
