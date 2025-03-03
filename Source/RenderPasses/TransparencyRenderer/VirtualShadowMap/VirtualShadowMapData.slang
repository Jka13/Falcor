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
        int2 gClipMapOriginOffset;
        float3 gCameraPosW;
        uint2 gClipMapSize;
        uint2 gPageSize;
        uint2 gVirtualClipMapSize;
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

    RWBuffer<uint> gAvailableMemory[NUM_CLIPMAPS];
    RWBuffer<uint> gAllocatedMemory[NUM_CLIPMAPS];
    RWBuffer<uint> gCountBuffer;
    RWBuffer<uint> gRenderQueue;

    enum Queue : uint8_t
    {
        AvailableMemory = 0,
        AllocatedMemory = 1,
        Render = 2
    }

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

    void freeAllocatedMemory(uint clipMapIdx)
    {
        while (!queueIsEmpty(Queue::AllocatedMemory, clipMapIdx))
        {
            pop(Queue::AllocatedMemory, clipMapIdx);
        }; 
    }

    void updateRequiredPages(float3 posW, uint clipMapLevel)
    {
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapLevel);
        if (any(virtualAddress >= gPageSize))
        {
            return;
        }
        gVirtualClipMapsRW[clipMapLevel][virtualAddress] = 1;
    }

    void requestMemory(uint clipMapLevel)
    {
            
    }
    //Debug
    void clearVirtualTextures(uint2 pixel)
    {
        if (any(pixel >= gClipMapSize))
        {
            return;
        }
        for (uint clipMapLevel = 0; clipMapLevel < NUM_CLIPMAPS; ++clipMapLevel)
        {
            gVirtualClipMapsRW[clipMapLevel][pixel] = 0;
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
        projectedPosW.y *= -1;
        projectedPosW = projectedPosW * pow(0.5, clipMapIdx);
        projectedPosW = projectedPosW * 0.5 + 0.5;
        uint2 virtualAddress = uint2(projectedPosW * gClipMapSize);
        virtualAddress /= gPageSize;
        return virtualAddress;
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
    // Queue operations
    bool queueIsEmpty(Queue q, uint clipMapIdx)
    {
        uint front = getQueueFrontIdx(q, clipMapIdx);
        uint back = front + 1;
        return gCountBuffer[front] == gCountBuffer[back];
    }

    uint getQueueFrontIdx(Queue q, uint clipMapIdx)
    {
        switch (q)
        {
        case Queue::AllocatedMemory:
            return (uint) q * NUM_CLIPMAPS * 2 + clipMapIdx * 2;
        case Queue::AvailableMemory:
            return (uint) q * NUM_CLIPMAPS * 2 + clipMapIdx * 2;
        case Queue::Render:
            return (uint) q * NUM_CLIPMAPS * 2 + 1;
        default:
            return 0;
        }
    }

    uint pop(Queue q, uint clipMapIdx)
    {
        uint frontIdx = getQueueFrontIdx(q, clipMapIdx);
        uint front = 0;
        switch (q)
        {
        case Queue::AllocatedMemory:
            InterlockedAdd(gCountBuffer[frontIdx], 1u, front);
            return gAllocatedMemory[clipMapIdx][front];
        case Queue::AvailableMemory:
            InterlockedAdd(gCountBuffer[frontIdx], 1u, front);
            return gAvailableMemory[clipMapIdx][front];
        case Queue::Render:
            InterlockedAdd(gCountBuffer[frontIdx], 1u, front);
            return gRenderQueue[front];
        }
    }

    void add(Queue q, uint clipMapIdx, uint value)
    {
        uint backIdx = getQueueFrontIdx(q, clipMapIdx) + 1;
        uint back = 0;
        switch (q)
        {
        case Queue::AllocatedMemory:
            InterlockedAdd(gCountBuffer[backIdx], 1u, back);
            gAllocatedMemory[clipMapIdx][back] = value;
        case Queue::AvailableMemory:
            InterlockedAdd(gCountBuffer[backIdx], 1u, back);
            gAvailableMemory[clipMapIdx][back] = value;
        case Queue::Render:
            InterlockedAdd(gCountBuffer[backIdx], 1u, back);
            gRenderQueue[back] = value;
        }
    }
} gVirtualShadowMapData;
