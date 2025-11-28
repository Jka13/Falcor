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

cbuffer UI
{
    uint gPixelRadius;
    float gRejectionAngle;
    float gRejectionDistance;
    float gCosOpeningAngle;
    float gPenumbraAngle;
}


cbuffer PerFrame
{
    uint gFrameCount;
}

RWStructuredBuffer<rh.Reservoir> gReservoir;
RWStructuredBuffer<rh.Reservoir> gReservoirPrev;
RWTexture2D<float4> gOutDebug;

struct ResampleData
{
    rh.Reservoir reservoir;
    ShadingData sd;
    bool shadingDataIsValid;

    __init(uint2 pixel, uint index, bool usePrevData)
    {
        if (usePrevData)
        {
            reservoir = gReservoirPrev[index];
            shadingDataIsValid = sdh.getPrevShadingData(pixel, this.sd); 
        }
        else
        {
            reservoir = gReservoir[index];
            shadingDataIsValid = sdh.getShadingData(pixel, this.sd); 
        }
    }
}

//TODO implement uniform sampling of a disc
bool sampleDisc(uint2 pixel, uint pixelRadius, out int2 sample, inout SampleGenerator sg)
{
    float2 rnd = sampleNext2D(sg);
    float rotationAngle = 2 * M_PI * rnd.x;
    float radius = pixelRadius * rnd.y;
    sample = int2(pixel) + int2(int(cos(rotationAngle) * radius), int(sin(rotationAngle) * radius));
    bool sampleValid = all(sample >= 0) && all(sample < rh.gFrameDim);
    return sampleValid; 
}

bool inCone(const float3 toLight, const float3 lightDir)
{
    float cosTheta = -dot(toLight, lightDir);
    return cosTheta > gCosOpeningAngle;
}

float getTargetFunction(ShadingData sd, rh.Sample sample, float3 direction)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        if (!inCone(toLight, direction))
            return 0.0;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float targetFunction = brdfSample * luminance(sample.flux);
        return targetFunction;
}

float getTargetFunction(ShadingData sd, rh.Sample sample, float3 coneDirection, uint2 pixel, SampleGenerator sg)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        if (!inCone(toLight, coneDirection))
            return 0.0;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        //brdfSample = luminance(bsdf.eval(sd, evalDirection, sg));
        gOutDebug[pixel] = float4(sample.flux * bsdf.eval(sd, toLight, sg) / bsdf.evalPdf(sd, toLight), 1);
        return brdfSample * luminance(sample.flux);
}

float calcMISWeight(ResampleData base, ResampleData neighbour, uint2 pixel)
{
    float misWeight = 0.0;
    float targetFunction = 0.0;

    float3 toLight = base.reservoir.sample.posW - neighbour.sd.posW;
    float distanceToLight = length(toLight);
    toLight /= distanceToLight;
    bool isVisible = sdh.dispatchShadowRay(neighbour.sd, toLight, distanceToLight);
    if (isVisible)
        targetFunction = getTargetFunction(neighbour.sd, base.reservoir.sample, base.reservoir.pad);
    float c_sum = neighbour.reservoir.confidenceWeight * targetFunction + base.reservoir.confidenceWeight * base.reservoir.sample.targetFunction;
    if (c_sum > 0)
        misWeight = base.reservoir.confidenceWeight * base.reservoir.sample.targetFunction / c_sum;
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
    bool comparable = angle > gRejectionAngle && minDistance / maxDistance > gRejectionDistance;
    return comparable;
}


[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    sdh.copyVBuffer(pixel);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    uint index = rh.getIndex(pixel);

    ResampleData pixelData = ResampleData(pixel, index, false);
    if (!pixelData.shadingDataIsValid) 
        return;


    uint2 prevPixel = pixel;
    float2 rnd = sampleNext2D(sg);
    bool valid = sdh.applyMotionVector(prevPixel, rnd);

    uint prevIndex = rh.getIndex(prevPixel);

    ResampleData temporalData = ResampleData(prevPixel, prevIndex, true);
    if (!temporalData.shadingDataIsValid)
        return;

    valid &= isNeighbourValid(temporalData.sd, pixelData.sd);
    if (valid)
    {

        float baseMIS = calcMISWeight(pixelData, temporalData, pixel);
        float baseWeight = baseMIS * pixelData.reservoir.sample.targetFunction;
        pixelData.reservoir.w_sum *= baseWeight;

        float3 toLight = temporalData.reservoir.sample.posW - pixelData.sd.posW;
        float distanceToLight = length(toLight);
        toLight /= distanceToLight;
        bool isVisible = sdh.dispatchShadowRay(pixelData.sd, toLight, distanceToLight);

        float temporalMIS = 0.0;
        float targetFunction = 0.0;
        if (isVisible)
        {
            targetFunction = getTargetFunction(pixelData.sd, temporalData.reservoir.sample, temporalData.reservoir.pad, pixel, sg);
            float c_sum = pixelData.reservoir.confidenceWeight * targetFunction + temporalData.reservoir.confidenceWeight * temporalData.reservoir.sample.targetFunction;
            if (c_sum > 0)
                temporalMIS = temporalData.reservoir.confidenceWeight * temporalData.reservoir.sample.targetFunction / c_sum;
        }
        float temporalWeight = temporalMIS * targetFunction * temporalData.reservoir.w_sum;
        temporalData.reservoir.sample.targetFunction = targetFunction;
        pixelData.reservoir.addSample(temporalData.reservoir.sample, temporalData.reservoir.pad, temporalWeight, temporalData.reservoir.confidenceWeight, sg);
        pixelData.reservoir.finalizeSample();
    }

    valid = true;
    int2 neighbourPixel;
    sampleDisc(pixel, gPixelRadius, neighbourPixel, sg);

    uint2 prevNeighbourPixel = uint2(neighbourPixel);
    valid &= sdh.applyMotionVector(prevNeighbourPixel, rnd);
    if (!valid)
    {
        gReservoir[index] = pixelData.reservoir;
        return;
    }

    prevIndex = rh.getIndex(prevNeighbourPixel);

    //ResampleData neighbourData = ResampleData(prevNeighbourPixel, prevIndex, true);
    //valid &= neighbourData.shadingDataIsValid;

    //valid &= isNeighbourValid(neighbourData.sd, pixelData.sd);
    //if (valid)
    //{
    //    //update current w_sum for resampling
    //    float baseMIS = calcMISWeight(pixelData, neighbourData);
    //    float baseWeight = baseMIS * pixelData.reservoir.sample.targetFunction;
    //    pixelData.reservoir.w_sum *= baseWeight;

    //    float3 toLight = neighbourData.reservoir.sample.posW - pixelData.sd.posW;
    //    float distanceToLight = length(toLight);
    //    toLight /= distanceToLight;
    //    bool isVisible = sdh.dispatchShadowRay(pixelData.sd, toLight, distanceToLight);

    //    float neighbourMIS = 0.0;
    //    float targetFunction = 0.0; 
    //    if (isVisible)
    //    {
    //        targetFunction = getTargetFunction(pixelData.sd, neighbourData.reservoir.sample, neighbourData.reservoir.pad);
    //        float c_sum = pixelData.reservoir.confidenceWeight * targetFunction + neighbourData.reservoir.confidenceWeight * neighbourData.reservoir.sample.targetFunction;
    //        if (c_sum > 0)
    //            neighbourMIS = neighbourData.reservoir.confidenceWeight * neighbourData.reservoir.sample.targetFunction / c_sum;
    //    }
    //    float neighbourWeight = neighbourMIS * targetFunction * neighbourData.reservoir.w_sum;
    //    neighbourData.reservoir.sample.targetFunction = targetFunction;
    //    pixelData.reservoir.addSample(neighbourData.reservoir.sample, neighbourData.reservoir.pad, neighbourWeight, neighbourData.reservoir.confidenceWeight, sg);
    //    pixelData.reservoir.finalizeSample();
    //}
    gReservoir[index] = pixelData.reservoir;
}
