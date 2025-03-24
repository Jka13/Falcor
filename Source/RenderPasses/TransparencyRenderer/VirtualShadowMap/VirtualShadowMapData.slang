#pragma once

#ifndef NUM_CLIPMAPS
#define NUM_CLIPMAPS 1
#endif

struct VirtualShadowMapShaderData
{
    cbuffer SMCB
    {
        float3 gCameraPosW;
        uint gClipMap0Extention;
        uint gRenderBudget;
        int2 gClipMapOriginOffsets[2];
        int2 gOverallOriginOffset;
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
        uint gRenderBufferSize;
        uint gAvailableMemorySize;
    }

    RWTexture2D<uint> gVirtualClipMapsRW[NUM_CLIPMAPS];
    Texture2D<uint> gVirtualClipMaps[NUM_CLIPMAPS];

    RWTexture2D<float> gPhysicalClipMapsRW[NUM_CLIPMAPS];
    Texture2D<float> gPhysicalClipMaps[NUM_CLIPMAPS];

    RWBuffer<uint> gAvailableMemoryStack[NUM_CLIPMAPS];
    RWBuffer<uint> gStackCounter;
    RWBuffer<uint> gRenderBuffer;

    //public:
    void writeShadowValue(float distance, float3 posW)
    {
    }

    float readShadowValue(float3 posW)
    {
        uint clipMapIdx = posWToClipMapIdx(posW);
        uint2 virtualAddress;
        uint2 offset;
        uint status;
        uint clipMapLevel;
        for (; clipMapIdx < NUM_CLIPMAPS; ++clipMapIdx)
        {
            uint4 packedData = posWToVirtualAddressWithOffset(posW, clipMapIdx);
            int2 overallOffset = int2((float2) gOverallOriginOffset * pow(0.5, clipMapIdx));
            int2 lastOffset = int2((float2) gClipMapOriginOffsets[1] * pow(0.5, clipMapIdx));
            virtualAddress = (packedData.xy + overallOffset + lastOffset) % gVirtualClipMapSize;
            offset = packedData.zw;
            status = getStatus(gVirtualClipMaps[clipMapIdx][virtualAddress]);
            clipMapLevel = clipMapIdx;
            if (isActive(status))
            {
                break;
            }
        }
        if (!isActive(status))
        {
            return 1.0;
        }
        uint address = getAddress(gVirtualClipMaps[clipMapLevel][virtualAddress]);
        uint2 texel = convertAddressToTexel(address);
        return gPhysicalClipMaps[clipMapLevel][texel + offset];
    }
    
    float onlyReadClipMapX(float3 posW, uint clipMapIdx)
    {
        uint4 packedData = posWToVirtualAddressWithOffset(posW, clipMapIdx);
        int2 overallOffset = int2((float2) gOverallOriginOffset * pow(0.5, clipMapIdx));
        int2 lastOffset = int2((float2) gClipMapOriginOffsets[1] * pow(0.5, clipMapIdx));
        uint2 virtualAddress = (packedData.xy + overallOffset + lastOffset) % gVirtualClipMapSize;
        uint2 offset = packedData.zw;
        uint status = getStatus(gVirtualClipMaps[clipMapIdx][virtualAddress]);
        if (!isActive(status))
        {
            return 1.0;
        }
        uint address = getAddress(gVirtualClipMaps[clipMapIdx][virtualAddress]);
        uint2 texel = convertAddressToTexel(address);
        return gPhysicalClipMaps[clipMapIdx][texel + offset];
    }

    void updateRequiredPages(float3 posW, uint clipMapLevel)
    {
        uint2 virtualAddress = posWToVirtualAddress(posW, clipMapLevel);
        if (any(virtualAddress >= gVirtualClipMapSize))
        {
            return;
        }
        int2 overallOffset = int2((float2) gOverallOriginOffset * pow(0.5, clipMapLevel));
        int2 lastOffset = int2((float2) gClipMapOriginOffsets[1] * pow(0.5, clipMapLevel));
        virtualAddress = (virtualAddress + lastOffset + overallOffset) % gVirtualClipMapSize;
        uint status = getStatus(gVirtualClipMapsRW[clipMapLevel][virtualAddress]);
        if (!isInactive(status))
        {
            return;
        }
        setPending(gVirtualClipMapsRW[clipMapLevel][virtualAddress]);
    }

    void writeThreadID(uint2 threadId, uint clipMapLevel)
    {
        gVirtualClipMapsRW[clipMapLevel][threadId] = threadId.x + threadId.y * gVirtualClipMapSize.x;
    }
    
    void assignMemory(uint2 texel, uint clipMapLevel)
    {
        uint status = getStatus(gVirtualClipMapsRW[clipMapLevel][texel]);
        if (!isPending(status))
        {
            return;
        }
        uint address = pop(clipMapLevel);
        setAddress(gVirtualClipMapsRW[clipMapLevel][texel], address);
    }

    void invalidatePages(uint2 pixel, uint clipMapLevel)
    {
        uint status = getStatus(gVirtualClipMapsRW[clipMapLevel][pixel]);
        if (isInactive(status))
        {
            return;
        }
        uint address = getAddress(gVirtualClipMapsRW[clipMapLevel][pixel]);
        setInactive(gVirtualClipMapsRW[clipMapLevel][pixel]);
        add(clipMapLevel, address);
    }

    void updateRenderBuffer(uint2 pixel, uint clipMapLevel)
    {
        uint status = getStatus(gVirtualClipMapsRW[clipMapLevel][pixel]);
        if (!isPending(status))
        {
            return;
        }
        uint index;
        InterlockedAdd(gStackCounter[NUM_CLIPMAPS], 1u, index);
        if (index >= gRenderBufferSize)
        {
            return;
        }
        uint entry = clipMapLevel * gVirtualClipMapSize.x * gVirtualClipMapSize.y + pixel.x + pixel.y * gVirtualClipMapSize.x;
        setActive(gRenderBuffer[index]);
        setAddress(gRenderBuffer[index], entry);
        setActive(gVirtualClipMapsRW[clipMapLevel][pixel]);
    }

    void shiftOrigin(int2 pixel, uint clipMapLevel)
    {
        uint status = getStatus(gVirtualClipMapsRW[clipMapLevel][pixel]);
        if (isInactive(status))
            return;
        int2 overallOffset = int2((float2) gOverallOriginOffset * pow(0.5, clipMapLevel));
        int2 lastOffset = int2((float2) gClipMapOriginOffsets[1] * pow(0.5, clipMapLevel));
        int2 shiftedPixel = (pixel - gVirtualClipMapSize - overallOffset) % gVirtualClipMapSize - lastOffset;
        if (all(shiftedPixel < (int2) gVirtualClipMapSize) && all(shiftedPixel >= 0))
        {
            return;
        }
        uint address = getAddress(gVirtualClipMapsRW[clipMapLevel][pixel]);
        gVirtualClipMapsRW[clipMapLevel][pixel] = 1;
        add(clipMapLevel, address);
    }

    void resetRenderBuffer()
    {
        gStackCounter[NUM_CLIPMAPS] = 0;
    }

    void invalidateRenderData(uint index)
    {
        setInactive(gRenderBuffer[index]);
    }

    //Stack Management
    uint pop(uint clipMapLevel)
    {
        uint stackIdx; 
        InterlockedAdd(gStackCounter[clipMapLevel], uint(-1), stackIdx);
        return gAvailableMemoryStack[clipMapLevel][(stackIdx - 1) % gAvailableMemorySize];
    }

    void add(uint clipMapLevel, uint value)
    {
        uint stackIdx;
        InterlockedAdd(gStackCounter[clipMapLevel], 1u, stackIdx);
        gAvailableMemoryStack[clipMapLevel][stackIdx % gAvailableMemorySize] = value;
    }

    // private:
    uint getStatus(uint texel)
    {
        return texel & 0xf0000000;
    }

    uint getAddress(uint texel)
    {
        return texel & 0x0fffffff;
    }

    bool isActive(uint status)
    {
        return status == 0x20000000;
    }

    bool isPending(uint status)
    {
        return status == 0x10000000;
    }

    bool isInactive(uint status)
    {
        return status == 0x00000000;
    }

    void setActive(inout uint texel)
    {
        texel = texel & 0x0fffffff;
        texel = texel | 0x20000000;
    }

    void setPending(inout uint texel)
    {
        texel = texel & 0x0fffffff;
        texel = texel | 0x10000000;
    }

    void setInactive(inout uint texel)
    {
        texel = texel & 0x0fffffff;
    }

    void setAddress(inout uint texel, uint address)
    {
        texel = texel & 0xf0000000;
        texel = texel | address;
    }

    uint3 unpackRenderData(uint data)
    {
        uint clipMapLevel = data / (gVirtualClipMapSize.x * gVirtualClipMapSize.y);
        uint textureBlock = data % (gVirtualClipMapSize.x * gVirtualClipMapSize.y);
        uint2 virtualTexel = uint2(textureBlock % gVirtualClipMapSize.x, textureBlock / gVirtualClipMapSize.x);
        return uint3(clipMapLevel, virtualTexel);
    }

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
        uint2 virtualAddress = uint2(projectedPosW * gVirtualClipMapSize);
        return virtualAddress;
    }

    uint4 posWToVirtualAddressWithOffset(float3 posW, uint clipMapIdx)
    {
        float2 projectedPosW = mul(gViewProjection, float4(posW, 1.f)).xy;
        projectedPosW.y *= -1;
        projectedPosW = projectedPosW * pow(0.5, clipMapIdx);
        projectedPosW = projectedPosW * 0.5 + 0.5;
        int2 virtualAddress = uint2(projectedPosW * gClipMapSize);
        uint2 offset = virtualAddress % gPageSize;
        virtualAddress /= gPageSize;
        return uint4(virtualAddress, offset);
    }

    uint posWToClipMapIdx(float3 posW)
    {
        uint clipMapIdx = 0;
        float2 projectedPos = abs(mul(gViewProjection, float4(posW, 1.f)).xy);
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
