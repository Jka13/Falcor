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
#include "../TransparencyShadowMethod.h"
#include "Rendering/ShadowMaps/Blur/SMGaussianBlur.h"

class VirtualShadowMap : public TransparencyShadowMethod
{
public:
    virtual ~VirtualShadowMap() = default;

    VirtualShadowMap(ref<Device> pDevice, ref<Scene> pScene);

    void initAvailableMemoryStack();

    void initStackCounter();

    /** Generate resources needed to evaluate the Shadow Method
     */
    virtual void generate(RenderContext* pRenderContext, const RenderData& renderData) override;

    /** Returns defines needed for the method
     */
    virtual DefineList getDefines() override;

    /** Set the shader data for the VirtualShadowMapData shader
     */
    void setShadowData(const ShaderVar& var, bool RW);

    /** Set the needed shader data for the method (textures,buffer, etc)
     */
    virtual void setShaderData(const ShaderVar& var) override;

    /** Render UI for the method
     */
    virtual bool renderUI(Gui::Widgets& widget) override;

    void debugPass(RenderContext* pRenderContext, const RenderData& renderData, ref<Texture> debugOut, ref<Texture> colorOut) override;

private:
    struct LightVP
    {
        float4x4 viewProjection;
        float4x4 invViewProjection;
    };
    void updateViewProjection(ref<Light> pLight);
    void shiftClipMapOrigin(RenderContext* pRenderContext);
    void sampleViewFrustum(RenderContext* pRenderContext, const RenderData& renderData);
    void updateClipMaps(RenderContext* pRenderContext);
    void updateRenderBuffer(RenderContext* pRenderContext);
    void invalidateRenderData(RenderContext* pRenderContext);
    void setDirectionalLightSource();
    void prepareResources(RenderContext* pRenderContext);
    // Function that generates the profiler passes in case they are not executed this frame
    void dummyProfileGeneration(RenderContext* pRenderContext);
    //Runtime
    uint mFrameCount = 0;
    // Shader Resources
    uint mRenderBudget = 512; //Render Budget in terms of how many pages are rendered at most every frame
    uint2 mClipMapSize = uint2(4096);
    uint2 mPageSize = uint2(128); //in Texel
    uint2 mVirtualClipMapSize = uint2(32); //TODO calculate this accordingly to clip map size and page size
    const uint mNumClipMaps = 1; 
    std::vector<ref<Texture>> mpPhysicalClipMaps;
    std::vector<ref<Texture>> mpVirtualClipMaps;
    uint mDirectionalLightSourceIndex = 0;
    float4x4 mView;
    std::vector<LightVP> mLightVPs;
    // Clip Map Handles
    float mClipMap0Extention = 2;
    std::vector<float2> mInitCameraPosWs;
    std::vector<int2> mOverallOriginOffsets;
    std::vector<int2> mClipMapOriginOffsets;
    float2 mVirtualClipMapExtentionInLightViewSpace = float2(2 * mClipMap0Extention / mVirtualClipMapSize.x, 2 * mClipMap0Extention / mVirtualClipMapSize.y);
    bool mMoved = false;
    // Memory Management Resources
    bool mFirstExecute = true;
    ref<Buffer> mpRenderBuffer;
    uint mRenderBufferSize;
    std::vector<ref<Buffer>> mpAvailableMemoryStack;
    uint mAvailableMemorySize;
    ref<Buffer> mpStackCounter;
    uint mStackCounterSize;
    RayTracingPipeline mGenVirtualShadowMapPip;
    ref<ComputePass> mpSampleViewFrustumPass;
    ref<ComputePass> mpUpdateOriginShiftPass;
    ref<ComputePass> mpUpdateVirtualClipMapPass;
    ref<ComputePass> mpUpdateRenderBufferPass;
    ref<ComputePass> mpInvalidateRenderDataPass;
    ref<ComputePass> mpDebugMemoryPass;
    // Memory Debug View
    bool mShowMemoryDebugView = true;
};
