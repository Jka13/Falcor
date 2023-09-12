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
#include "Core/Macros.h"
#include "Core/Enum.h"
#include "Core/State/GraphicsState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramReflection.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ProgramVersion.h"
#include "Core/Program/RtProgram.h"
#include "Utils/Properties.h"
#include "Utils/Debug/PixelDebug.h"
#include "Scene/Scene.h"

#include "ShadowMapData.slang"
#include "SMGaussianBlur.h"

#include <memory>
#include <type_traits>
#include <vector>
#include <map>

/*
    Wrapper Module for Shadow Maps, which allow ShadowMaps to be easily integrated into every Render Pass.
*/
namespace Falcor
{
class RenderContext;

class FALCOR_API ShadowMap
{
public:
    ShadowMap(ref<Device> device, ref<Scene> scene);

    // Renders and updates the shadow maps if necessary
    bool update(RenderContext* pRenderContext);

    void renderUI(Gui::Widgets& widget);

    // Returns a define List with all the defines. Need to be called once per frame to update defines
    DefineList getDefines() const;
    // Sets Shader data
    void setShaderData(const uint2 frameDim = uint2(1920u, 1080u));

    // Sets the shader data and binds the block to var "gShadowMap"
    void setShaderDataAndBindBlock(ShaderVar rootVar, const uint2 frameDim = uint2(1920u, 1080u));

    // Gets the parameter block needed for shader usage
    ref<ParameterBlock> getParameterBlock() const { return mpShadowMapParameterBlock; }

    //Get Normalized pixel size used in oracle function
    float getNormalizedPixelSize(uint2 frameDim, float fovY, float aspect);
    float getNormalizedPixelSizeOrtho(uint2 frameDim, float width, float height);    //Ortho case

private:
    struct ShaderParameters
    {
        float4x4 viewProjectionMatrix = float4x4();

        float3 lightPosition = float3(0, 0, 0);
        float farPlane = 30.f;
    };

    LightTypeSM getLightType(const ref<Light> light);
    void prepareShadowMapBuffers();
    void prepareRasterProgramms();
    void prepareProgramms();
    void prepareGaussianBlur();
    void setSMShaderVars(ShaderVar& var, ShaderParameters& params);
    void updateRasterizerStates();
    void handleNormalizedPixelSizeBuffer();
   

    DefineList getDefinesShadowMapGenPass() const;

    void renderCubeEachFace(uint index, ref<Light> light, RenderContext* pRenderContext);
    bool renderSpotLight(uint index, ref<Light> light, RenderContext* pRenderContext, std::vector<bool>& wasRendered);
    bool renderCascaded(uint index, ref<Light> light, RenderContext* pRenderContext);
    float4x4 getProjViewForCubeFace(uint face, const LightData& lightData, const float4x4& projectionMatrix);
    void calcProjViewForCascaded(uint index, const LightData& lightData);

    // Getter
    std::vector<ref<Texture>>& getShadowMapsCube() { return mpShadowMapsCube; }
    std::vector<ref<Texture>>& getShadowMaps() { return mpShadowMaps; }
    ref<Buffer> getViewProjectionBuffer() { return mpVPMatrixBuffer; }
    ref<Buffer> getLightMapBuffer() { return mpLightMapping; }
    ref<Sampler> getSampler() { return mpShadowSamplerPoint; }
    float getFarPlane() { return mFar; }
    float getNearPlane() { return mNear; }
    uint getResolution() { return mShadowMapSize; }
    uint getCountShadowMapsCube() const { return mpShadowMapsCube.size(); }
    uint getCountShadowMaps() const { return mpShadowMaps.size(); }

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    ref<Fbo> mpFbo;
    ref<Fbo> mpFboCube;
    ref<Fbo> mpFboCascaded;

    const uint kStagingBufferCount = 3;
    const uint kNumberDebugTextures = 16;

    uint mShadowMapSize = 1024;
    uint mShadowMapSizeCube = 1024;
    uint mShadowMapSizeCascaded = 2048;
    ResourceFormat mShadowMapFormat = ResourceFormat::D32Float;
    RasterizerState::CullMode mCullMode = RasterizerState::CullMode::Back;
    std::map<RasterizerState::CullMode, ref<RasterizerState>> mFrontClockwiseRS;
    std::map<RasterizerState::CullMode, ref<RasterizerState>> mFrontCounterClockwiseRS;

    // Settings
    ShadowMapType mShadowMapType = ShadowMapType::Variance;
    OracleDistFunction mOracleDistanceFunctionMode = OracleDistFunction::RoughnessSquare;
    float mNear = 0.1f;
    float mFar = 30.f;
    bool mUsePCF = false;
    bool mUsePoissonDisc = false;
    bool mBiasSettingsChanged = false;
    int32_t mBias = 0;
    float mSlopeBias = 0.f;
    bool mUseAlphaTest = true;
    uint mAlphaMode = 3;                    //Mode for the alpha test ; 1 = Basic, 2 = HashedIsotropic, 3 = HashedAnisotropic
    float mPoissonDiscRad = 0.5f;
    float mPoissonDiscRadCube = 0.015f;
    float mSMCubeWorldBias = 0.f;
    uint mCascadedLevelCount = 4;
    float mCascadedFrustumFix = 0.5f;
    float mCascZMult = 3.f;    //Pushes the z Values apart
    bool mCascadedStochasticBlend = true;
    float mCascadedStochasticBlendBand = 0.05f;
    float mExponentialSMConstant = 80.f;    //Value used in the paper
    float mEVSMConstant = 20.f;                 // Exponential Variance Shadow Map constant. Needs to be lower than the exponential counterpart
    float2 mHSMFilteredThreshold = float2(0.02f, 0.98f);     //Threshold for filtered shadow map variants
    bool mUseRayOutsideOfShadowMap = false;
    bool mVarianceUseSelfShadowVariant = true;

    bool mUseSMOracle = true;         ///< Enables Shadow Map Oracle function
    bool mUseOracleDistFactor = true; ///< Enables a lobe distance factor that is used in the oracle function TODO rename
    float mOracleCompaireValue = 1.f; ///< Compaire Value for the Oracle test. Tested against ShadowMapArea/CameraPixelArea.
    bool mUseHybridSM = false;        ///< Uses the Hybrid Shadow Maps (https://gpuopen.com/fidelityfx-hybrid-shadows/#details)

    bool mUseShadowMipMaps = true;        ///< Uses mip maps for applyable shadow maps
    bool mUseGaussianBlur = true;


    bool mApplyUiSettings = false;
    bool mAlwaysRenderSM = false;
    bool mFirstFrame = true;
    bool mResetShadowMapBuffers = false;
    bool mShadowResChanged = false;
    bool mRasterDefinesChanged = false;
    bool mTypeChanged = false;

    //Internal
    std::vector<float4x4> mCascadedVPMatrix;
    uint mCascadedMatrixStartIndex = 0;         //Start index for the matrix buffer
    bool mMultipleSMTypes = false;
    float mCascadedMaxFar = 1000000.f;
    bool mCascadedFirstThisFrame = true;
    std::vector<float> mCascadedZSlices;
    std::vector<float2> mCascadedWidthHeight;
    uint2 mNPSOffsets = uint2(0);   //x = idx first spot; y = idx first cascade

    std::unique_ptr<SMGaussianBlur> mpBlurShadowMap;
    std::unique_ptr<SMGaussianBlur> mpBlurCascaded;
    std::unique_ptr<SMGaussianBlur> mpBlurCube;
    
    //std::vector<bool> mIsCubeSM;
    std::vector<LightTypeSM> mPrevLightType;  // Vector to check if the Shadow Map Type is still correct
    std::vector<float4x4> mSpotDirViewProjMat;

    std::vector<ref<Texture>> mpCascadedShadowMaps; //Cascaded Shadow Maps for Directional Lights
    std::vector<ref<Texture>> mpShadowMapsCube;     // Cube Shadow Maps (Point Lights)
    std::vector<ref<Texture>> mpShadowMaps;         // 2D Texture Shadow Maps (Spot Lights + (WIP) Area Lights)
    ref<Buffer> mpLightMapping;
    ref<Buffer> mpVPMatrixBuffer;
    std::vector<ref<Buffer>> mpVPMatrixStangingBuffer;
    ref<Buffer> mpNormalizedPixelSize;             //Buffer with the normalized pixel size for each ShadowMap
    ref<Sampler> mpShadowSamplerPoint;
    ref<Sampler> mpShadowSamplerLinear;
    ref<Texture> mpDepthCascaded;                  //Depth texture needed for some types of cascaded (can be null)
    ref<Texture> mpDepthCube;                      //Depth texture needed for the cube map
    ref<Texture> mpDepth;                          //Depth texture needed for some types of 2D SM (can be null)
    ref<Texture> mpTestTex;

    ref<ComputePass> mpReflectTypes;               // Dummy pass needed to create the parameter block
    ref<ParameterBlock> mpShadowMapParameterBlock; // Parameter Block

    struct RasterizerPass
    {
        ref<GraphicsState> pState = nullptr;
        ref<GraphicsProgram> pProgram = nullptr;
        ref<GraphicsVars> pVars = nullptr;

        void reset()
        {
            pState.reset();
            pProgram.reset();
            pVars.reset();
        }
    };

    RasterizerPass mShadowCubePass;
    RasterizerPass mShadowMapPass;
    RasterizerPass mShadowMapCascadedPass;
};

}
