#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;

#ifndef PATH_LENGTH
#define PATH_LENGTH 1
#endif

cbuffer CB
{
    uint2 gDispatchResolution;
    uint gMaxRecursion;
}

struct PathData
{
    float4 luminance;
    uint count;
}

struct LightPathVertex
{
    float3 flux;
    float forwardPdf;
    float backwardPdf;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;
};

struct CameraPathVertex
{
    float3 luminance;
    float forwardPdf;
    float backwardPdf;
    float3 thp;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;
};

// inputs
RWStructuredBuffer<LightPathVertex> gLightPaths;
RWStructuredBuffer<CameraPathVertex> gCameraPaths;
RWStructuredBuffer<PathData> gPathData;

// outputs
RWTexture2D<float4> gColor;

uint pixelToLightpathIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (gMaxRecursion + 1);
}
uint pixelToPathDataIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (2 * gMaxRecursion);
}

uint pixelToCamerapathIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * gMaxRecursion;
}

float4 average(uint2 pixel)
{
    uint index = pixelToPathDataIndex(pixel, gDispatchResolution);
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    float4 luminance = float4(0);
    for (uint pathOffset = 0; pathOffset < PATH_LENGTH; ++pathOffset)
    {
        luminance = gPathData[index + pathOffset].luminance;
        luminance /= gPathData[index + pathOffset].count > 0 ? float(gPathData[index + pathOffset].count) : 1;
        color += luminance; 
    }
    return color; 
}

float4 onlyShowPathLengthX(uint index, uint pathLength)
{
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    color += gPathData[index + pathLength].luminance;
    return color; 
}

float4 debug(uint2 pixel)
{
    uint index = pixelToCamerapathIndex(pixel, gDispatchResolution);
    return float4(gCameraPaths[index + 0].luminance, 1.0);
}

float4 onlyShowCameraPath(uint2 pixel)
{
    uint index = pixelToCamerapathIndex(pixel, gDispatchResolution);
    float3 color = float3(0.0, 0.0, 0.0);
    for (uint pathOffset = 0; pathOffset < PATH_LENGTH; ++pathOffset)
    {
        color += gCameraPaths[index + pathOffset].luminance * gCameraPaths[index + pathOffset].thp;
    }
    return float4(color, 1.0);
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    //gColor[pixel] = average(pixel);
    //gColor[pixel] = debug(pixel);
    //gColor[pixel] = onlyShowCameraPath(pixel);
}
