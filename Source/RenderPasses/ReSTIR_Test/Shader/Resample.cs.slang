#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

import Scene.RaytracingInline;

import Helpers.ReservoirHandler;
import Helpers.SceneDataHandler;

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
    uint index = rh.getIndex(pixel);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    let textureSampler = ExplicitLodTextureSampler(0);
    uint2 pixelSample;
    rh.Reservoir reservoir = gReservoir[index];
    rh.Sample s = reservoir.sample;
    reservoir.w_sum = s.p_hat * reservoir.w_sum;

    bool sampleValid = sampleDisc(pixel, 20, pixelSample, sg);
    if (sampleValid)
    {
        //get sample data
        uint sampleIndex = rh.getIndex(pixelSample);
        rh.Reservoir reservoirSample = gReservoir[sampleIndex];
        rh.Sample spatialSample = reservoirSample.sample;
        //get shading data
        float3 rayDir = sdh.getRayDir(pixel);
        HitInfo hit = sdh.getHitInfo(pixel);
        ShadingData sd = {};
        //TODO handle invalid shading data
        bool valid = sdh.getShadingData(hit, rayDir, sd);
        //get new brdf
        float3 toLight = normalize(spatialSample.posW - sd.posW);
        float distanceToLight = distance(spatialSample.posW, sd.posW);
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);

        float p_hat = luminance(brdfSample * spatialSample.flux / (distanceToLight * distanceToLight));
        float w = 0.5 * p_hat * reservoirSample.w_sum;
        reservoir.addSample(spatialSample, w, sg);
        reservoir.finalizeSample();
    }
    gReservoirSpatial[index] = reservoir;
}
