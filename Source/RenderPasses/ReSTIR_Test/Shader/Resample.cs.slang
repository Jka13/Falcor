#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

import Scene.RaytracingInline;

import Helpers.ReservoirHandler;

cbuffer PerFrame
{
    uint gFrameCount;
}

StructuredBuffer<rh.Reservoir> gReservoir;
RWStructuredBuffer<rh.Reservoir> gReservoirSpatial;

//TODO implement uniform sampling of a disc
bool sampleDisc(uint2 pixel, uint pixelRadius, out uint2 sample, inout SampleGenerator sg)
{
    float2 rnd = sampleNext2D(sg);
    float rotationAngle = 2 * M_PI * rnd.x;
    float radius = pixelRadius * rnd.y;
    sample = pixel + uint2(cos(radius), sin(radius));
    return all(sample > 0) && all(sample < rh.gFrameDim);
}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    uint2 pixelSample;
    bool sampleValid = sampleDisc(pixel, 20, pixelSample, sg);
    if (sampleValid)
    {
    }
}
