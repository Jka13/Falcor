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

cbuffer CameraData
{
    float4x4 gPrevCamViewProjection;
    float3 gPrevCamPos;
    float3 gPrevCamForward;
}

cbuffer PerFrame
{
    uint gFrameCount;
    float gNormalizedPixelArea;
}

RWStructuredBuffer<rh.Reservoir> gReservoir;
RWStructuredBuffer<rh.Reservoir> gReservoirPrev;

//Splatting
StructuredBuffer<uint> gCellCounters; //Number of Reservoirs per Pixel
StructuredBuffer<uint> gCellOffsets; // Initial offset in the sorted reservoir buffer
StructuredBuffer<uint2> gSplattedPixels;

struct SplattingResampleData
{
    rh.Reservoir reservoir;
    ShadingData sd;
    bool shadingDataIsValid;

    __init(uint2 pixel, uint index, bool usePrevData)
    {
        if (usePrevData)
        {
            reservoir = gReservoirPrev[index];
            shadingDataIsValid = sdh.getShadingData(HitInfo(sdh.gVBuffer[pixel]), reservoir.pad1, this.sd); 
        }
        else
        {
            reservoir = gReservoir[index];
            shadingDataIsValid = sdh.getShadingData(HitInfo(sdh.gVBuffer[pixel]), reservoir.pad1, this.sd); 
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

float getTargetFunction(ShadingData sd, rh.Sample sample, float3 coneDirection, float3 evalDirection)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        if (!inCone(toLight, coneDirection))
            return 0.0;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, evalDirection);
        float targetFunction = brdfSample * luminance(sample.flux);
        return targetFunction;
}

// Backprojects to camera with viewProjection matrix
float3 getCameraProjection(float4x4 viewProjection, float3 posW) {
    float4 screenPos = mul(viewProjection, float4(posW, 1));
    screenPos /= screenPos.w;
    screenPos.y *= -1;
    screenPos.xy = screenPos.xy * 0.5 + 0.5; // UV from [-1,1] to [0,1]
    return screenPos.xyz;
}

// Returns the Jacobian determinant for a subpixel location
float getSubpixelLocationJacobian(float dist, float3 toCamera, float3 surfaceN, float3 camForward)
{
    float distSquared = dist * dist;
    float cosNormal = dot(toCamera, surfaceN);
    float cosSensor = dot(-toCamera, camForward);
    return abs(cosNormal / (distSquared)) / abs(pow(cosSensor, 3));
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

    SplattingResampleData pixelData = SplattingResampleData(pixel, index, false);
    if (!pixelData.shadingDataIsValid) 
        return;

    float3 toPrevCam = gPrevCamPos - pixelData.sd.posW;
    float distToPrevCam = length(toPrevCam);
    toPrevCam /= distToPrevCam;

    float3 prevScreenPos = getCameraProjection(gPrevCamViewProjection, pixelData.sd.posW);

    bool isVisible = (all(prevScreenPos.xyz >= 0) && all(prevScreenPos.xyz < 1));
    if (isVisible)
    {
        isVisible = sdh.dispatchShadowRay(pixelData.sd, toPrevCam, distToPrevCam);
    }

    float p0 = pixelData.reservoir.confidenceWeight * pixelData.reservoir.sample.targetFunction;
    float p1 = 0.0;

    if (isVisible)
    {
        float jacobianWeight = getSubpixelLocationJacobian(distToPrevCam, toPrevCam, pixelData.sd.faceN, gPrevCamForward);
        float jacobianShifted = jacobianWeight / pixelData.reservoir.sample.subpixelJacobian;

        float targetFunction = getTargetFunction(pixelData.sd, pixelData.reservoir.sample, pixelData.reservoir.pad, toPrevCam);
        uint2 prevPixel = uint2(floor(prevScreenPos.xy * rh.gFrameDim));
        uint prevIndex = rh.getIndex(prevPixel);
        uint c1 = gReservoirPrev[prevIndex].confidenceWeight;
        float p1 = c1 * targetFunction * jacobianShifted;
    }
    float sum = p0 + p1;
    float misWeight = sum > 0 ? p0 / sum : 0;

    float baseWeight = misWeight * pixelData.reservoir.sample.targetFunction;
    pixelData.reservoir.w_sum *= baseWeight;

    bool valid = true;
    uint numReservoirs = gCellCounters[index];
    uint cellOffsets = gCellOffsets[index];

    const float3 currentCamPosW = gScene.camera.getPosition();
    const float3 currentCamForward = normalize(gScene.camera.data.cameraW);
    float newConfidence = pixelData.reservoir.confidenceWeight;
    SplattingResampleData splatData;
    for (uint i = 0; i < numReservoirs; ++i)
    {
        uint2 splattedPixel = gSplattedPixels[cellOffsets + i];
        uint splattedIndex = rh.getIndex(splattedPixel);

        splatData = SplattingResampleData(splattedPixel, splattedIndex, true);
        if (!splatData.shadingDataIsValid)
            continue;

        float prevMIS = 0.0;
        float targetFunction = 0.0;
        float jacobianToCurrent = 0.0;
        float jacobianShifted = 0.0;

        valid &= isNeighbourValid(splatData.sd, pixelData.sd);
        if (valid)
        {
            float p0 = splatData.reservoir.sample.targetFunction * splatData.reservoir.confidenceWeight;

            float3 toCurrentCam = currentCamPosW - splatData.sd.posW;
            float distToCurrentCam = length(toCurrentCam);
            toCurrentCam /= distToCurrentCam;

            jacobianToCurrent = getSubpixelLocationJacobian(distToCurrentCam, toCurrentCam, splatData.sd.faceN, currentCamForward);
            jacobianShifted = splatData.reservoir.sample.subpixelJacobian > 0 ? jacobianToCurrent / splatData.reservoir.sample.subpixelJacobian : 0;
            targetFunction = getTargetFunction(splatData.sd, splatData.reservoir.sample, splatData.reservoir.pad, toCurrentCam);
            float p1 = splatData.reservoir.confidenceWeight * targetFunction * jacobianShifted;
            float sum = p0 + p1;
            prevMIS = sum > 0 ? p0 / sum : 0;
        }
        float w = prevMIS * targetFunction * splatData.reservoir.w_sum * jacobianShifted;
        splatData.reservoir.sample.targetFunction = targetFunction;
        splatData.reservoir.sample.subpixelJacobian = jacobianToCurrent;
        pixelData.reservoir.addSample(splatData.reservoir.sample, splatData.reservoir.pad, w, splatData.reservoir.confidenceWeight, sg);
    }
    pixelData.reservoir.finalizeSample();
    gReservoir[index] = pixelData.reservoir;
}
