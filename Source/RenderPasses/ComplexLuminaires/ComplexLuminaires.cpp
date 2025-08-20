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
    const Gui::DropdownList kModes{{0, "Test"}};
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
    if (!mpPhotonBuffer)
    {
        mpPhotonBuffer = Buffer::createStructured(
            mpDevice, 3 * sizeof(float3), mDispatchedPhotonsPerIteration,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false
        );
        mpPhotonBuffer->setName("ComplexLuminaires::PhotonBuffer");
    }
}

void ComplexLuminaires::prepareRayTracingShader(RenderContext* pRenderContext)
{
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();
    mGenerateSamplesPass.initRTProgram(mpDevice, mpScene, kGenerateSamplesShader, kMaxPayloadBytes, globalTypeConformances);
}

void ComplexLuminaires::prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "PrepareGenerateSamplesPass");

    mGenerateSamplesPass.pProgram->addDefine("USE_EMISSIVE_LIGHT", mpScene->useEmissiveLights() ? "1" : "0");

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

    var["CB"]["gFlags"] = flags;
    var["CB"]["gMaxRecursion"] = mMaxRecursion;
    var["gPhotonBuffer"] = mpPhotonBuffer;
}


void ComplexLuminaires::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ++mFrameCount;
    if (!mpScene)
        return;
    //prepareResources
    preparePhotonBuffer(pRenderContext, renderData);
    prepareLight(pRenderContext, renderData);

    //prepareShaders
    prepareGenerateSamplesPass(pRenderContext, renderData);

    //generateSamples
    mpScene->raytrace(pRenderContext,mGenerateSamplesPass.pProgram.get(),mGenerateSamplesPass.pVars, uint3(mDispatchedPhotonsPerIteration, 1, 1));
}

void ComplexLuminaires::renderUI(Gui::Widgets& widget)
{
}

void ComplexLuminaires::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    //Reset Scene
    mpScene = pScene;

    mGenerateSamplesPass = RayTraceProgramHelper::create();
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
