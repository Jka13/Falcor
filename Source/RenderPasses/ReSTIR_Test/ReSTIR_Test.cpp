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
#include "ReSTIR_Test.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Math/FalcorMath.h"

namespace
{
const std::string kGenerateSamplesShader = "RenderPasses/ReSTIR_Test/Shader/GenerateSamples.rt.slang";
const std::string kResampleShader = "RenderPasses/ReSTIR_Test/Shader/Resample.cs.slang";
const std::string kCombineShader = "RenderPasses/ReSTIR_Test/Shader/Combine.cs.slang";

const std::string kShaderModel = "6_5";
const uint kMaxPayloadBytes = 96u;

const std::string kOutputColor = "color";
const std::string kInputVBuffer= "vBuffer";
const std::string kInputView= "view";
const std::string kInputMVec = "motionVector";

const Falcor::ChannelList kInputChannels{
    {kInputVBuffer, "gVBuffer", "vBuffer", false /*optional*/},
    {kInputMVec, "gMVec", "motionVector", false /*optional*/},
    {kInputView, "gView", "view", false /*optional*/},
};

const Falcor::ChannelList kOutputChannels{
    {kOutputColor, "gOutColor", "Output Color (linear)", false /*optional*/, ResourceFormat::RGBA32Float},
};
const Gui::DropdownList kModes{{0, "Test"}};
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIR_Test>();
}

ReSTIR_Test::ReSTIR_Test(ref<Device> pDevice, const Properties& props)
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

Properties ReSTIR_Test::getProperties() const
{
    return {};
}

RenderPassReflection ReSTIR_Test::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ReSTIR_Test::prepareRayTracingShader(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    mGenerateSamplesPass.initRTProgram(mpDevice, mpScene, kGenerateSamplesShader, kMaxPayloadBytes, globalTypeConformances);
}

void ReSTIR_Test::setSceneData(const RenderData& renderData, const ShaderVar& var)
{
    auto sceneDataVar = var["sdh"];
    sceneDataVar["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
    sceneDataVar["gView"] = renderData[kInputView]->asTexture();
    sceneDataVar["gMVec"] = renderData[kInputMVec]->asTexture();
}

void ReSTIR_Test::setReservoirData(const RenderData& renderData, const ShaderVar& var)
{
    auto reservoirDataVar = var["rh"];
    reservoirDataVar["CB"]["gFrameDim"] = renderData.getDefaultTextureDims();
}

void ReSTIR_Test::prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareGenerateSamplesPass");

    if (!mGenerateSamplesPass.pVars)
    {
        if (mpEmissiveLightSampler)
            mGenerateSamplesPass.pProgram->addDefines(mpEmissiveLightSampler->getDefines());
        mGenerateSamplesPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);
    }
    auto var = mGenerateSamplesPass.pVars->getRootVar();
    mpEmissiveLightSampler->setShaderData(var["LightCB"]["gEmissiveLightSampler"]);
    mpSampleGenerator->setShaderData(var);
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    var["gReservoir"] = mpSampleReservoirs[mFrameCount % 2];
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
}

void ReSTIR_Test::prepareResamplePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareResampePass");
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
    var["PerFrame"]["gFrameCount"] = mFrameCount;
    setReservoirData(renderData, var);
    setSceneData(renderData, var);
    FALCOR_ASSERT(mpResamplePass);
}

void ReSTIR_Test::prepareCombinePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareCombinePass");
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

void ReSTIR_Test::prepareReservoirs(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint2 frameDim = renderData.getDefaultTextureDims();
    uint reservoirSize = frameDim.x * frameDim.y;
    if (!mpSampleReservoirs[0] || !mpSampleReservoirs[1])
    {
        for (uint i = 0; i < 2; ++i)
        {
            mpSampleReservoirs[i] = Buffer::createStructured(
                mpDevice, 3 * sizeof(float3) + sizeof(float) + sizeof(uint), reservoirSize,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
            );
            mpSampleReservoirs[i]->setName("ReSTIR_Test::Reservoir" + std::to_string(i));
        }
    }
}

void ReSTIR_Test::prepareLight(RenderContext* pRenderContext, const RenderData& renderData)
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

void ReSTIR_Test::generateSamples(RenderContext* pRenderContext, uint2 dispatchSize)
{
    mpScene->raytrace(pRenderContext,mGenerateSamplesPass.pProgram.get(),mGenerateSamplesPass.pVars, uint3(dispatchSize, 1));
}

void ReSTIR_Test::resample(RenderContext* pRenderContext, uint2 dispatchSize)
{
    mpResamplePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}

void ReSTIR_Test::combine(RenderContext* pRenderContext, uint2 dispatchSize)
{
    mpCombinePass->execute(pRenderContext, dispatchSize.x, dispatchSize.y);
}


void ReSTIR_Test::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ++mFrameCount;
    if (!mpScene)
        return;

    prepareReservoirs(pRenderContext, renderData);
    prepareLight(pRenderContext, renderData);
    prepareGenerateSamplesPass(pRenderContext, renderData);
    prepareResamplePass(pRenderContext, renderData);
    prepareCombinePass(pRenderContext, renderData);

    uint2 dispatchSize = renderData.getDefaultTextureDims();
    generateSamples(pRenderContext, dispatchSize);
    if (!mFirstFrame)
        resample(pRenderContext, dispatchSize);
    mFirstFrame = false;
    combine(pRenderContext, dispatchSize);
}

void ReSTIR_Test::renderUI(Gui::Widgets& widget)
{}

void ReSTIR_Test::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    //Reset Scene
    mpScene = pScene;

    mGenerateSamplesPass = RayTraceProgramHelper::create();
    mpResamplePass.reset();
    mpCombinePass.reset();
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

void ReSTIR_Test::RayTraceProgramHelper::initRTProgram(ref<Device> device,ref<Scene> scene,const std::string& shaderName,
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

void ReSTIR_Test::RayTraceProgramHelper::initProgramVars(ref<Device> pDevice,ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator)
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
