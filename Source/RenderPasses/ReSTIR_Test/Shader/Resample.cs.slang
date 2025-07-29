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
RWStructuredBuffer<rh.Reservoir> gReservoirPrev;

struct ResampleData
{
    rh.Reservoir reservoir;
    ShadingData sd;
    bool shadingDataIsValid;

    __init(uint2 pixel, bool usePrevData)
    {
        uint index = rh.getIndex(pixel);
        if (usePrevData)
            reservoir = gReservoirPrev[index];
        else
            reservoir = gReservoir[index];

        shadingDataIsValid = sdh.getShadingData(pixel, this.sd); 
    }
}

//TODO implement uniform sampling of a disc
bool sampleDisc(uint2 pixel, uint pixelRadius, out int2 sample, inout SampleGenerator sg)
{
    float2 rnd = sampleNext2D(sg);
    float rotationAngle = 2 * M_PI * rnd.x;
    float radius = pixelRadius * rnd.y;
    sample = pixel + int2(cos(radius), sin(radius));
    bool sampleValid = all(sample > 0) && all(sample < rh.gFrameDim);
    return sampleValid; 
}

float getTargetFunction(ShadingData sd, rh.Sample sample)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        float distanceToLight = distance(sample.posW, sd.posW);
        float squaredDistance = distanceToLight * distanceToLight;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float targetFunction = 0;
        if (squaredDistance > 0) 
            targetFunction = brdfSample * luminance(sample.flux) / squaredDistance;
        return targetFunction;
}

float calcMISWeight(inout ResampleData base, ResampleData neighbour)
{
    float3 toLight = normalize(neighbour.reservoir.sample.posW - base.sd.posW);
    float distanceToLight = distance(neighbour.reservoir.sample.posW, base.sd.posW);
    bool isVisible = sdh.dispatchShadowRay(base.sd, toLight, distanceToLight);
    float targetFunction = 0;
    if (isVisible)
    {
        targetFunction = getTargetFunction(neighbour.sd, base.reservoir.sample);
    }
    float c_sum = neighbour.reservoir.confidenceWeight * targetFunction + base.reservoir.confidenceWeight * base.reservoir.sample.targetFunction;
    float misWeight = base.reservoir.confidenceWeight * base.reservoir.sample.targetFunction / max(c_sum, 0.0001);
    return misWeight;
}

bool isNeighbourValid(ShadingData sd1, ShadingData sd2)
{
    float3 camPosW = gScene.camera.getPosition();
    float distance1 = distance(sd1.posW, camPosW);
    float distance2 = distance(sd2.posW, camPosW);
    float maxDistance = max(distance1, distance2);
    float minDistance = min(distance1, distance2);
    float angle = dot(sd1.faceN, sd2.faceN);
    bool comparable = angle >= 0.8 && minDistance / maxDistance > 0.95;
    return comparable;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    ResampleData pixelData = ResampleData(pixel, false);
    if (!pixelData.shadingDataIsValid) 
        return;

    uint index = rh.getIndex(pixel);

    uint2 prevPixel = pixel;
    float2 rnd = sampleNext2D(sg);
    bool valid = sdh.applyMotionVector(prevPixel, rnd);

    int2 neighbourPixel;
    valid &= sampleDisc(prevPixel, 20, neighbourPixel, sg);
    if (!valid)
        return;

    ResampleData neighbourData = ResampleData(neighbourPixel, true);
    valid &= neighbourData.shadingDataIsValid;

    //TODO check relative distance
    valid &= isNeighbourValid(neighbourData.sd, pixelData.sd);
    if (true)
    {
        //update current w_sum for resampling
        float baseMIS = calcMISWeight(pixelData, neighbourData);
        float baseWeight = baseMIS * pixelData.reservoir.sample.targetFunction;
        pixelData.reservoir.w_sum *= baseWeight;

        float neighbourMIS = calcMISWeight(neighbourData, pixelData);
        float targetFunction = getTargetFunction(pixelData.sd, neighbourData.reservoir.sample);
        float neighbourWeight = neighbourMIS * targetFunction * neighbourData.reservoir.w_sum;
        neighbourData.reservoir.sample.targetFunction = targetFunction;
        pixelData.reservoir.addSample(neighbourData.reservoir.sample, neighbourWeight, neighbourData.reservoir.confidenceWeight, sg);
        pixelData.reservoir.finalizeSample();
    }
    valid = true;
    ResampleData temporalData = ResampleData(prevPixel, true);
    if (!temporalData.shadingDataIsValid)
    {
        gReservoir[index] = pixelData.reservoir;
        return;
    }
    valid &= isNeighbourValid(temporalData.sd, pixelData.sd);
    if (true)
    {
        float baseMIS = calcMISWeight(pixelData, temporalData);
        float baseWeight = baseMIS * pixelData.reservoir.sample.targetFunction;
        pixelData.reservoir.w_sum *= baseWeight;

        float temporalMIS = calcMISWeight(temporalData, pixelData);
        float temporalTargetFunction = getTargetFunction(pixelData.sd, temporalData.reservoir.sample);
        float temporalWeight = temporalMIS * temporalTargetFunction * temporalData.reservoir.w_sum;
        temporalData.reservoir.sample.targetFunction = temporalTargetFunction;
        pixelData.reservoir.addSample(temporalData.reservoir.sample, temporalWeight, temporalData.reservoir.confidenceWeight, sg);
        pixelData.reservoir.finalizeSample();
    }
    gReservoir[index] = pixelData.reservoir;
}
