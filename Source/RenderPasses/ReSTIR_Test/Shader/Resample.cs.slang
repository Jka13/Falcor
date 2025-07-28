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

RWStructuredBuffer<rh.Reservoir> gReservoir;
RWStructuredBuffer<rh.Reservoir> gReservoirSpatial;

struct ResampleData
{
    rh.Reservoir reservoir;
    ShadingData sd;
    bool shadingDataIsValid;

    __init(uint2 pixel)
    {
        uint index = rh.getIndex(pixel);
        this.reservoir = gReservoir[index];
        this.shadingDataIsValid = sdh.getShadingData(pixel, this.sd); 
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
        float p_hat = luminance(brdfSample * sample.flux);// / max(distanceToLight * distanceToLight, 0.0001));
        return p_hat;
}

//float getResamplePHat(ShadingData sd, rh.Sample sample, uint index)
//{
//        let textureSampler = ExplicitLodTextureSampler(0);
//        float3 toLight = normalize(sample.posW - sd.posW);
//        float distanceToLight = distance(sample.posW, sd.posW);
//        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
//        let bsdfProperties = bsdf.getProperties(sd);
//        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
//        float squareDistance = distanceToLight * distanceToLight;
//        float p_hat = luminance(brdfSample * sample.flux / max(squareDistance, 0.0001));
//        //rh.Reservoir test = rh.Reservoir();
//        //test.sample.flux = sd.faceN;
//        //test.sample.posW = sd.posW;
//        //test.sample.p_hat = p_hat;
//        //gReservoir[index] = test;
//        return p_hat;
//}

void updateSampleWeight(inout ResampleData baseData, ResampleData neighbourData)
{
    float p_hat = getResamplePHat(neighbourData.sd, baseData.reservoir.sample);
    float c_sum = neighbourData.reservoir.confidenceWeight * p_hat + baseData.reservoir.confidenceWeight * baseData.reservoir.sample.p_hat;
    float m = baseData.reservoir.confidenceWeight * baseData.reservoir.sample.p_hat / max(c_sum, 0.0001);
    //w_sum currently contains W = w_sum / p_hat
    float w_sum = baseData.reservoir.w_sum * baseData.reservoir.sample.p_hat;
    float w = m * baseData.reservoir.sample.p_hat * w_sum;
    baseData.reservoir.w_sum = w;
}

void resample(inout ResampleData baseData, ResampleData neighbourData, inout SampleGenerator sg, uint index)
{
    float p_hat = getResamplePHat(neighbourData.sd, baseData.reservoir.sample);
    float c_sum = baseData.reservoir.confidenceWeight * baseData.reservoir.sample.p_hat + neighbourData.reservoir.confidenceWeight * p_hat;
    float m = neighbourData.reservoir.confidenceWeight * p_hat / c_sum;
    //w_sum currently contains W = w_sum / p_hat
    float w_sum = neighbourData.reservoir.w_sum * neighbourData.reservoir.sample.p_hat;
    float w = m * neighbourData.reservoir.sample.p_hat * w_sum;
    gReservoir[index] = neighbourData.reservoir;
    baseData.reservoir.addSample(neighbourData.reservoir.sample, w, neighbourData.reservoir.confidenceWeight, sg);
}

bool isNeighbourValid(ShadingData sd1, ShadingData sd2)
{
    float distance = distance(sd1.posW, sd2.posW);
    float angle = dot(sd1.faceN, sd2.faceN);
    bool comparable = angle >= 0.9;
    return comparable;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    uint index = rh.getIndex(pixel);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    ResampleData pixelData = ResampleData(pixel);
    bool valid = pixelData.shadingDataIsValid;
    if (!valid)
    {
        gReservoirSpatial[index] = pixelData.reservoir;
        return;
    }

    uint2 neighbourPixel;
    valid &= sampleDisc(pixel, 20, neighbourPixel, sg);
    if (!valid)
    {
        gReservoirSpatial[index] = pixelData.reservoir;
        return;
    }

    ResampleData neighbourData = ResampleData(neighbourPixel);
    valid &= neighbourData.shadingDataIsValid;
    if (!valid)
    {
        gReservoirSpatial[index] = pixelData.reservoir;
        return;
    }

    //TODO check relative distance
    valid &= isNeighbourValid(neighbourData.sd, pixelData.sd);
    if (valid)
    {
        //get sample data
        updateSampleWeight(pixelData, neighbourData);
        resample(pixelData, neighbourData, sg, index);
        pixelData.reservoir.finalizeSample();
    }
    gReservoirSpatial[index] = pixelData.reservoir;
}
