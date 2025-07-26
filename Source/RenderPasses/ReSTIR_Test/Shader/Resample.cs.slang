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

struct ResampleData
{
    rh.Reservoir reservoir;
    float3 rayDir;
    HitInfo hit;
    ShadingData sd;
    bool shadingDataIsValid;

    __init(uint2 pixel)
    {
        uint index = rh.getIndex(pixel);
        this.reservoir = gReservoir[index];
        this.rayDir = sdh.getRayDir(pixel);
        this.hit = sdh.getHitInfo(pixel);
        this.shadingDataIsValid = sdh.getShadingData(hit, rayDir, sd); 
    }
}

//TODO implement uniform sampling of a disc
bool sampleDisc(uint2 pixel, uint pixelRadius, out uint2 sample, inout SampleGenerator sg)
{
    float2 rnd = sampleNext2D(sg);
    float rotationAngle = 2 * M_PI * rnd.x;
    float radius = pixelRadius * rnd.y;
    sample = pixel + uint2(cos(radius), sin(radius));
    bool sampleValid = all(sample > 0) && all(sample < rh.gFrameDim);
    return sampleValid; 
}

float getResamplePHat(ShadingData sd, rh.Sample sample)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        float distanceToLight = distance(sample.posW, sd.posW);
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float p_hat = luminance(brdfSample * sample.flux / (distanceToLight * distanceToLight));
        return p_hat;
}

void updateSample(ShadingData sd, inout rh.Reservoir base, rh.Reservoir pair)
{
    float p_hat = getResamplePHat(sd, base.sample);
    float c_sum = pair.confidenceWeight * pair.sample.p_hat + base.confidenceWeight * p_hat;
    float m = base.confidenceWeight * p_hat / c_sum;
    //w_sum currently contains W = w_sum / p_hat
    float w_sum = base.w_sum * base.sample.p_hat;
    float w = m * p_hat * w_sum;
    base.w_sum = w_sum;
}

void resample(ShadingData sd, inout rh.Reservoir base, rh.Reservoir pair, inout SampleGenerator sg)
{
    float p_hat = getResamplePHat(sd, pair.sample);
    float c_sum = base.confidenceWeight * base.sample.p_hat + pair.confidenceWeight * p_hat;
    float m = pair.confidenceWeight * p_hat / c_sum;
    //w_sum currently contains W = w_sum / p_hat
    float w_sum = pair.w_sum * pair.sample.p_hat;
    float w = m * p_hat * w_sum;
    base.addSample(pair.sample, w, pair.confidenceWeight, sg);
}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    uint index = rh.getIndex(pixel);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    let textureSampler = ExplicitLodTextureSampler(0);
    uint2 pixelSample;
    ResampleData sampleData = ResampleData(pixel);
    bool sampleValid = sampleDisc(pixel, 20, pixelSample, sg);

    if (sampleValid)
    {
        //TODO check if the two samples are similar
        //get sample data
        ResampleData spatialSampleData = ResampleData(pixelSample);
        updateSample(spatialSampleData.sd, sampleData.reservoir, spatialSampleData.reservoir);
        resample(sampleData.sd, sampleData.reservoir, spatialSampleData.reservoir, sg);
        sampleData.reservoir.finalizeSample();
    }
    gReservoirSpatial[index] = sampleData.reservoir;
}
