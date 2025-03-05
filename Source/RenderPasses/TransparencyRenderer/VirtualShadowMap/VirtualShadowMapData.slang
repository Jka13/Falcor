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
        float3 gCameraPosW;
        uint gClipMap0Extention;
        int2 gClipMapOriginOffset;
        uint2 gClipMapSize;
        uint2 gPageSize;
        uint2 gVirtualClipMapSize;
    }

    cbuffer ShadowVPs
    {
        float4x4 gViewProjection;
        float4x4 gInvViewProjection; //Only valid if there is a directional light
    }

    cbuffer QCB
    {
        uint gCountBufferSize;
        uint gRenderQueueSize;
        uint gAllocatedMemorySize;
        uint gAvailableMemorySize;
    }

    RWTexture2D<uint> gVirtualClipMapsRW[NUM_CLIPMAPS];
    Texture2D<uint> gVirtualClipMaps[NUM_CLIPMAPS];

    RWTexture2D<float> gPhysicalClipMapsRW[NUM_CLIPMAPS];
    Texture2D<float> gPhysicalClipMaps[NUM_CLIPMAPS];

    RWBuffer<uint> gAvailableMemory[NUM_CLIPMAPS];
    RWBuffer<uint> gAllocatedMemory[NUM_CLIPMAPS];
    RWBuffer<uint> gCountBuffer;
    RWBuffer<uint> gRenderQueue;

    enum Queue : uint
    {
        AllocatedMemory = 0,
        AvailableMemory = 1,
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
        Queue q = Queue::AllocatedMemory;
        uint frontIdx = getQueueFrontIdx(q, clipMapIdx);
        uint front = gCountBuffer[frontIdx];
        uint back = gCountBuffer[frontIdx + 1];
        while (!(front >= back))
        {
            uint address = pop(q, clipMapIdx);
            if (address == uint(-1))
            {
                return;
            }
            add(Queue::AvailableMemory, clipMapIdx, address);
            front = gCountBuffer[frontIdx];
        }
    }

    void updateRenderQueue(uint clipMapIdx, uint address)
    {
        Queue q = Queue::Render;
        add(q, 0, clipMapIdx);
        add(q, 0, address);
    }

    void updateRequiredPages(float3 posW, uint clipMapLevel)
    {
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapLevel);
        if (any(virtualAddress >= gPageSize))
        {
            return;
        }
        gVirtualClipMapsRW[clipMapLevel][virtualAddress] = 1;
        uint address = pop(Queue::AvailableMemory, clipMapLevel);
        if (address == uint(-1))
        {
            return;
        }
        add(Queue::AllocatedMemory, clipMapLevel, address);
        //updateRenderQueue(clipMapLevel, address);
    }

    void initializeAvailableMemory(uint clipMapLevel, uint2 pixel)
    {
        if (any(pixel >= gVirtualClipMapSize))
        {
            return;
        }
        uint index = pixel.y * gVirtualClipMapSize.x + pixel.x;
        pixel *= gPageSize;
        gAvailableMemory[clipMapLevel][index] = convertTexelToAddress(pixel);
    }

    void initializeCounter(uint2 pixel)
    {
        if (pixel.x >= gCountBufferSize || pixel.y != 0)
        {
            return;
        }
        gCountBuffer[pixel.x] = 0;
        if (pixel.x < NUM_CLIPMAPS)
        {
            uint backIdx = getQueueFrontIdx(Queue::AvailableMemory, pixel.x) + 1;
            gCountBuffer[backIdx] = gAvailableMemorySize;
        }
    }

    // Queue operations
    bool queueIsEmpty(Queue q, uint clipMapIdx)
    {
        uint front = getQueueFrontIdx(q, clipMapIdx);
        uint back = front + 1;
        return gCountBuffer[front] >= gCountBuffer[back];
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
            return (uint) q * NUM_CLIPMAPS * 2;
        default:
            return uint(-1);
        }
    }

    uint pop(Queue q, uint clipMapIdx)
    {
        uint frontIdx = getQueueFrontIdx(q, clipMapIdx);
        uint front = gCountBuffer[frontIdx];
        uint back = gCountBuffer[frontIdx + 1];
        if (front >= back)
        {
            return uint(-1);
        }
        InterlockedAdd(gCountBuffer[frontIdx], 1u, front);
        switch (q)
        {
        case Queue::AllocatedMemory:
            return gAllocatedMemory[clipMapIdx][front % gAllocatedMemorySize];
        case Queue::AvailableMemory:
            return gAvailableMemory[clipMapIdx][front % gAvailableMemorySize];
        case Queue::Render:
            return gRenderQueue[front % gRenderQueueSize];
        default:
            return uint(-1);
        }
    }

    void add(Queue q, uint clipMapIdx, uint value)
    {
        uint backIdx = getQueueFrontIdx(q, clipMapIdx) + 1;
        uint back = 0;
        InterlockedAdd(gCountBuffer[backIdx], 1u, back);
        switch (q)
        {
        case Queue::AllocatedMemory:
            gAllocatedMemory[clipMapIdx][back % gAllocatedMemorySize] = value;
        case Queue::AvailableMemory:
            gAvailableMemory[clipMapIdx][back % gAvailableMemorySize] = value;
        case Queue::Render:
            gRenderQueue[back % gRenderQueueSize] = value;
        }
    }

    void clearRenderQueue(uint2 pixel)
    {
        if(any(pixel > uint2(0)))
        {
            return;
        }
        uint frontIdx = getQueueFrontIdx(Queue::Render, 0);
        gCountBuffer[frontIdx] = 0;
        gCountBuffer[frontIdx + 1] = 0;
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

    uint convertTexelToAddress(uint2 texel)
    {
        return texel.y * gClipMapSize.x + texel.x;
    }

} gVirtualShadowMapData;
