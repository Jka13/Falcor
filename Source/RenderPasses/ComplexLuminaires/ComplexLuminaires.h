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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
// Light samplers
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
//Accel
#include "Rendering/AccelerationStructure/CustomAccelerationStructure.h"

using namespace Falcor;

class ComplexLuminaires : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ComplexLuminaires, "ComplexLuminaires", "Insert pass description here.");

    static ref<ComplexLuminaires> create(ref<Device> pDevice, const Properties& props) { return make_ref<ComplexLuminaires>(pDevice, props); }

    ComplexLuminaires(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    void prepareLight(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareRayTracingShader(RenderContext* pRenderContext);
    void prepareGenerateSamplesPass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareDirectIlluminationPass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareDirectIlluminationReferencePass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareDebugPass(RenderContext* pRenderContext, const RenderData& renderData);

    void directIllumiantionVPL(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim);
    void directIlluminationReference(RenderContext* pRenderContext, const RenderData& renderData, uint2 launchDim);

    void preparePhotonBuffer(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePhotonAABBBuffer(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareAccelerationStructure();
    void buildAccelerationStructure(RenderContext* pRenderContext, const RenderData& renderData);
    void setSceneData(const RenderData& renderData, const ShaderVar& var);

    //ReSTIR
    void setReservoirData(const RenderData& renderData, const ShaderVar& var);
    void prepareSamplePass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareResamplePass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareCombinePass(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareReservoirs(RenderContext* pRenderContext, const RenderData& renderData);
    void sample(RenderContext* pRenderContext, uint2 dispatchSize);
    void resample(RenderContext* pRenderContext, uint2 dispatchSize);
    void combine(RenderContext* pRenderContext, uint2 dispatchSize);

private:

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;


    std::unique_ptr<EmissiveLightSampler> mpEmissiveLightSampler;

    EmissiveLightSamplerType mEmissiveLightSamplerType = EmissiveLightSamplerType::Power;
    LightBVHSampler::Options mLightBVHOptions;

    //Complex Luminaires
    ref<Buffer> mpPhotonBuffer;
    ref<Buffer> mpPhotonAABBs;
    std::unique_ptr<CustomAccelerationStructure> mpPhotonAS;

    uint mFrameCount = 0;

    uint mDispatchedPhotonsPerIteration = 200000;
    uint mMaxPhotonCount = 200000;
    uint mMaxRecursion = 20;
    float mCosOpeningAngle = 0.5f;
    float mPenumbraAngle = 0.0f;
    float mAABBSize = 0.004f;

    //CL UI
    uint mMode = 1;
    float mConeExponent = 1;
    bool mShowDebug = false;
    bool mChangedPhotonBufferSize = false;
    bool mOptionsChanged = false;

    //ReSTIR
    std::array<ref<Buffer>, 2> mpSampleReservoirs;
    std ::array<ref<Texture>, 2> mpPrevVBuffers;
    std ::array<ref<Texture>, 2> mpPrevViews;

    //ReSTIR UI
    uint mNumberOfLightSamples = 32;
    uint mNumberOfBSDFSamples = 1;
    uint mSpatialSampleRadius = 20; 
    float mAngleThreshold = 0.8f;
    float mDistanceThreshold = 0.9f;

    struct RayTraceProgramHelper
    {
        ref<RtProgram> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        static const RayTraceProgramHelper create()
        {
            RayTraceProgramHelper r;
            r.pProgram = nullptr;
            r.pBindingTable = nullptr;
            r.pVars = nullptr;
            return r;
        }

        void initRTProgram(
            ref<Device> device,
            ref<Scene> scene,
            const std::string& shaderName,
            uint maxPayloadBytes,
            const Program::TypeConformanceList& globalTypeConformances
        );

        void initRTCollectionProgram(
            ref<Device> device,
            ref<Scene> scene,
            const std::string& shaderName,
            uint maxPayloadBytes,
            const Program::TypeConformanceList& globalTypeConformances
        );

        void initProgramVars(ref<Device> pDevice, ref<Scene> pScene, ref<SampleGenerator> pSampleGenerator);
    };

    //CL Shader
    RayTraceProgramHelper mGenerateSamplesPass;
    RayTraceProgramHelper mDirectIlluminationReferencePass;
    RayTraceProgramHelper mDebugPass;
    ref<ComputePass> mpDirectIlluminationPass;

    //ReSTIR Shader
    RayTraceProgramHelper mSamplePass;
    ref<ComputePass> mpResamplePass;
    ref<ComputePass> mpCombinePass;
};
