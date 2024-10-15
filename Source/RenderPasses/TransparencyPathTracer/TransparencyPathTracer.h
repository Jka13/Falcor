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
#include "Core/Enum.h"
#include "RenderGraph/RenderPass.h"
#include "SharedTypes.slang"
#include "Rendering/AccelerationStructure/CustomAccelerationStructure.h"

using namespace Falcor;

class TransparencyPathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(TransparencyPathTracer, "TransparencyPathTracer", "Path Tracer with Transparencys");

    static ref<TransparencyPathTracer> create(ref<Device> pDevice, const Properties& props) { return make_ref<TransparencyPathTracer>(pDevice, props); }

    TransparencyPathTracer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    enum class ShadowEvalMode : uint
    {
        RayTracing = 0u,
        AVSM = 1u,
        StochSM = 2u,
        TmpStochSM = 3u,
        Accel = 4u,
    };

    FALCOR_ENUM_INFO(ShadowEvalMode,{
        {ShadowEvalMode::RayTracing, "RayTracing"},
        {ShadowEvalMode::AVSM, "AVSM"},
        {ShadowEvalMode::StochSM, "StochSM"},
        {ShadowEvalMode::TmpStochSM, "TmpStochSM"},
        {ShadowEvalMode::Accel, "Accel"}
    });

    enum class ShadowAccelMode : uint
    {
        ShadowMap3D = 0u,
        NormalOffset = 1u,
    };

    FALCOR_ENUM_INFO(ShadowAccelMode,{
        {ShadowAccelMode::ShadowMap3D, "3DShadowMap"},
        {ShadowAccelMode::NormalOffset, "NormalOffset"},
    });

private:
    struct LightMVP
    {
        float3   pos = float3(0);
        uint     _pad = 0;
        float4x4 view = float4x4();
        float4x4 projection = float4x4();
        float4x4 viewProjection = float4x4();
        float4x4 invViewProjection = float4x4();
        float4x4 invProjection = float4x4();
        float4x4 invView = float4x4();

        void calculate(ref<Light> light, float2 nearFar);
    };


    void updateSMMatrices(RenderContext* pRenderContext, const RenderData& renderData);
    void generateAVSM(RenderContext* pRenderCotext, const RenderData& renderData);
    void generateStochasticSM(RenderContext* pRenderContext, const RenderData& renderData);
    void generateTmpStochSM(RenderContext* pRenderContext, const RenderData& renderData);
    void generateAccelShadow(RenderContext* pRenderContext, const RenderData& renderData);
    void traceScene(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareVars();
    void debugShowShadowAccel(RenderContext* pRenderContext, const RenderData& renderData);
    void renderDebugGraph(const ImVec2& size);
    void prepareDebugBuffers(RenderContext* pRenderContext);
    void generateDebugRefFunction(RenderContext* pRenderContext, const RenderData& renderData);


    // Internal state
    ref<Scene> mpScene;                     ///< Current scene.
    ref<SampleGenerator> mpSampleGenerator; ///< GPU sample generator.

    // Configuration Tracer
    TPTLightSampleMode mLightSampleMode = TPTLightSampleMode::RIS;
    uint mMaxBounces = 0;               ///< Max number of indirect bounces (0 = none).
    uint mMaxAlphaTestPerBounce = 32;   ///< Max number of allowed alpha tests per bounce
    bool mComputeDirect = true;         ///< Compute direct illumination (otherwise indirect only).
    bool mUseImportanceSampling = true; ///< Use importance sampling for materials.
    bool mUseRussianRoulettePath = false;   ///< Russian Roulett to abort the path early
    bool mUseRussianRouletteForAlpha = false; ///< Use Russian Roulette for transparent materials
    ShadowEvalMode mShadowEvaluationMode = ShadowEvalMode::Accel;

    // Runtime data Tracer
    uint mFrameCount = 0; ///< Frame count since scene was loaded.
    bool mDebugFrameCount = false;
    bool mOptionsChanged = false;
    ref<Sampler> mpPointSampler;

    //Configuration Shadow Map
    struct{
        bool avsm = false;
        bool stochSM = false;
        bool tmpStochSM = false;
        bool accelShadow = false;
    }mGenInactive;
    bool mAVSMRebuildProgram = false;
    bool mAVSMTexResChanged = false;
    bool mAVSMUsePCF = false;
    bool mAVSMUseInterpolation = false;
    bool mAVSMUseRandomVariant = false; //Enables random rejection weithed with the mode
    bool mAVSMUnderestimateArea = false;
    uint mAVSMRejectionMode = 0;    //Triangle Area | Rectange Area | Heights | Height Heuristic
    uint mSMSize = 512;
    float2 mNearFar = float2(1.f, 40.f);
    float mDepthBias = 1e-6f;
    float mNormalDepthBias = 1e-3f;
    uint mNumberAVSMSamples = 8;    //< k for AVSM

    //Accel shadow settings
    static const uint mAccelApproxNumElementsPerPixel = 4u;
    std::vector<uint> mAccelShadowNumPoints;
    ShadowAccelMode mAccelMode = ShadowAccelMode::NormalOffset;
    uint mAccelShadowMaxNumPoints = 0;
    bool mAccelShadowUseCPUCounterOptimization = true;
    float mAccelShadowOverestimation = 1.1f;

    struct
    {
        bool enable = false;
        uint selectedLight = 0;
        uint2 cullMin = uint2(0);
        uint2 cullMax = uint2(4096);
        float blendT = 0.2f;
        uint visMode = 0;
        bool stopGeneration = false;
    }mAccelDebugShowAS;

    //Runtime Data DeepSM
    std::vector<LightMVP> mShadowMapMVP;
    std::vector<ref<Texture>> mAVSMDepths;        //Depths for the avsm
    std::vector<ref<Texture>> mAVSMTransmittance;    //Trancemittance for each point of the avsm
    std::vector<ref<Texture>> mStochDepths;           //Depths for Stochastic SM
    std::vector<ref<Texture>> mStochTransmittance;    //Transmittance for Stochstic SM
    std::vector<ref<Texture>> mTmpStochDepths;        // Depths for Temporal stochastic SM
    std::vector<ref<Texture>> mTmpStochTransmittance; // Transmittance for Temporal stochastic SM
    std::vector<ref<Buffer>> mAccelShadowAABB;          //For Accel AABB points
    std::vector<ref<Buffer>> mAccelShadowCounter;       //Counter for inserting points
    std::vector<ref<Buffer>> mAccelShadowCounterCPU;       //Counter for inserting points
    std::vector<ref<Buffer>> mAccelShadowData;          //Transparency Data
    std::unique_ptr<CustomAccelerationStructure> mpShadowAccelerationStrucure;   //AS

    //Settings and Data for Tranmittance UI Graph
    struct
    {
        float radiusSize = 4.f;
        float borderThickness = 3.f;
        float lineThickness = 5.f;
        float depthRangeOffset = 0.01f;     //Offset for depth scaling
        bool addDepthBias = true;
        bool enableDepthScaling = true;     //Enables depth range scaling to the relevant part of the graph
        bool graphOpen = false;         
        bool asStepFuction = false;
        uint selectedLight = 0;             //Selected light for the vis function
        int2 selectedPixel = int2(-1, -1);  //Selected (Camera) pixel
        bool selectPixelButton = false;     //True if the pixel can be selected with a button press
        bool mouseDown = false;
        bool genBuffers = false;            //True if buffers should be filled
    } mGraphUISettings;

    static const uint kGraphMaxFunctions = 3; //Ref, AVSM, StochSM
    static const uint kGraphDataMaxSize = 256;
    struct GraphFunctionData
    {
        std::string name = "";
        uint show = 0b11111111; 
        std::vector<std::array<float2, kGraphDataMaxSize>> cpuData;
        std::vector <ref<Buffer>> pointsBuffers;
    };
    std::array<GraphFunctionData, 2> mGraphFunctionDatas;

    // Ray tracing program.
    struct RayTracingPipeline
    {
        ref<RtProgram> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    };

    RayTracingPipeline mTracer;
    RayTracingPipeline mGenAVSMPip; //Volumetric Adaptive SM
    RayTracingPipeline mGenStochSMPip;    //Stochastic baised SM
    RayTracingPipeline mGenTmpStochSMPip;   //Temporal Stochastic baised SM
    RayTracingPipeline mGenAccelShadowPip;     //Acceleration structure based
    RayTracingPipeline mDebugGetRefFunction;

    // Rasterization resources
    struct
    {
        ref<GraphicsState> pState;
        ref<GraphicsProgram> pProgram;
        ref<GraphicsVars> pVars;
        ref<Fbo> pFBO;
        ref<Texture> pDepth;
    } mRasterShowAccelPass;
};

FALCOR_ENUM_REGISTER(TransparencyPathTracer::ShadowEvalMode);
FALCOR_ENUM_REGISTER(TransparencyPathTracer::ShadowAccelMode);
