#ifndef PATH_LENGTH
#define PATH_LENGTH 1
#endif

cbuffer CB
{
    uint2 gDispatchResolution;
    uint gMaxRecursion;
}

//inputs
RWStructuredBuffer<float4> gPathData;

// outputs
RWTexture2D<float4> gColor;

uint pixelToPathDataIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (2 * gMaxRecursion - 1);
}

float4 averageMIS(uint index)
{
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    for (uint pathOffset = 0; pathOffset < PATH_LENGTH; ++pathOffset)
    {
        color += gPathData[index + pathOffset] / (pathOffset + 1);
    }
    return color; 
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    uint index = pixelToPathDataIndex(pixel, gDispatchResolution);
    gColor[pixel] = averageMIS(index);
    //gColor[pixel] = gPathData[index];
}
