#pragma once

#ifndef NUM_CLIPMAPS
#define NUM_CLIPMAPS 1
#endif
//import Rendering.ShadowMaps.ShadowTestFunctions;
//import Rendering.ShadowMaps.ShadowMapHelpers;
//__exported import Scene.Raytracing;
//__exported import Scene.Intersection;

struct VirtualShadowMapShaderData
{
    cbuffer SMCB
    {
        uint gClipMap0Extention;
        uint2 gClipMapOrigin;
        float3 gCameraPosW;
        uint2 gClipMapSize;
        uint2 gPageSize;
        uint2 gClipMap0Resolution;
        uint2 gTopLeftClipMapCorner;
    }

    cbuffer ShadowVPs
    {
        float4x4 gViewProjection;
        float4x4 gInvViewProjection; //Only valid if there is a directional light
    }

    RWTexture2D<uint> gVirtualClipMapsRW[NUM_CLIPMAPS];
    Texture2D<uint> gVirtualClipMaps[NUM_CLIPMAPS];

    RWTexture2D<float> gPhysicalClipMapsRW[NUM_CLIPMAPS];
    Texture2D<float> gPhysicalClipMaps[NUM_CLIPMAPS];

    //public:
    void writeShadowValue(float distance, float3 posW)
    {
        uint clipMapIdx = posWToClipMapIdx(posW);
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapIdx);
        uint2 physicalAddress = getPhysicalAddress(clipMapIdx);
        //gVirtualClipMapsRW[clipMapIdx][virtualAddress] = physicalAddress;
        gPhysicalClipMapsRW[clipMapIdx][virtualAddress] = distance;
    }

    float readShadowValue(float3 posW)
    {
        uint clipMapIdx = posWToClipMapIdx(posW);
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapIdx);
        switch (virtualAddress.x)
        {
            default:
                //physicalAddress = gVirtualClipMaps[clipMapIdx][virtualAddress];
                //return gPhysicalClipMaps[clipMapIdx][physicalAddress];
        }
        uint2 physicalAddress = getPhysicalAddress(clipMapIdx);
        return gPhysicalClipMaps[clipMapIdx][physicalAddress];
    }
    void updateRequiredPages(float3 posW, uint2 pixel)
    {
        uint clipMapIdx = posWToClipMapIdx(posW);
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapIdx);
        if (clipMapIdx < NUM_CLIPMAPS)
        {
           gVirtualClipMapsRW[clipMapIdx][virtualAddress] = 1;
        }
    }
    //private:

    float4x4 getViewProjectionMatrixForClipMapN(uint clipMapIdx)
    {
        float4x4 viewProjection = gViewProjection;
        return viewProjection; 
    }

    uint2 posWToVirtualAddress(float3 posW, uint clipMapIdx)
    {
        float2 projectedPosW = mul(gViewProjection, float4(posW, 1.f)).xy;
        projectedPosW = projectedPosW * pow(0.5, clipMapIdx);
        projectedPosW = projectedPosW * 0.5 + 0.5;
        float2 virtualAddress = projectedPosW * gClipMapSize;
        virtualAddress /= (float2)gPageSize;
        virtualAddress = floor(virtualAddress);
        return uint2(virtualAddress);
    }

    uint2 getPhysicalAddress(uint clipMapIdx)
    {
        uint physicalAddress;
        return physicalAddress;
    }

    uint posWToClipMapIdx(float3 posW)
    {
        uint clipMapIdx = 0;
        float2 projectedPos = mul(gViewProjection, float4(posW,1.f)).xy;
        while (projectedPos.x > 1 || projectedPos.y > 1)
        {
            projectedPos.xy *= 0.5;
            ++clipMapIdx;
        }
        return clipMapIdx;
    }

    uint2 convertAddressToTexel(uint address)
    {
        return uint2(address % gClipMapSize.x, address/gClipMapSize.x);
    }    

} gVirtualShadowMapData;
