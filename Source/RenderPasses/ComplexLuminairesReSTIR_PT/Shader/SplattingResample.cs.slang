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
import Helpers.VPLSampler;

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
RWTexture2D<PackedHitInfo> gSplattingHits;

RWTexture2D<float4> gOutDebug;
RWTexture2D<float4> gOutDebug1;

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

float getTargetFunction(ShadingData sd, rh.Sample sample, float3 coneDirection, float3 evalDirection, inout SampleGenerator sg)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        float flux = luminance(sh.samplePointLight(sd.posW, sample.posW, coneDirection, sample.flux));
        const uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, hints);
        let bsdfProperties = bsdf.getProperties(sd);
        float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
        float NdotL = saturate(dot(bsdfProperties.guideNormal, evalDirection));
        //float brdfSample = luminance((bsdf.eval(sd, evalDirection, sg) / NdotL) * NdotV);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, evalDirection) / NdotL * NdotV;
        float targetFunction = brdfSample * flux;
        return isnan(targetFunction) ? 0.0 : targetFunction;
}

float getTargetFunction(ShadingData sd, rh.Sample sample, float3 coneDirection, float3 evalDirection, uint2 pixel, inout SampleGenerator sg)
{
    let textureSampler = ExplicitLodTextureSampler(0);
    float3 toLight = normalize(sample.posW - sd.posW);
    float3 flux = sh.samplePointLight(sd.posW, sample.posW, coneDirection, sample.flux);
    const uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
    float NdotL = saturate(dot(bsdfProperties.guideNormal, evalDirection));
    float3 brdfSample = bsdf.eval(sd, evalDirection, sg) / NdotL * NdotV;
    brdfSample = isnan(brdfSample) ? 0.0 : brdfSample;
    return luminance(brdfSample * flux);
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
    gOutDebug[pixel] = float4(0);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    uint index = rh.getIndex(pixel);

    uint numReservoirs = gCellCounters[index];
    uint cellOffsets = gCellOffsets[index];

    gOutDebug[pixel] = numReservoirs;

    PackedHitInfo currentHit = sdh.gVBuffer[pixel];

    rh.Reservoir currentReservoir = gReservoir[index];
    ShadingData currentSD = {};

    if (!sdh.getShadingData(pixel, currentReservoir.pad1, currentSD))
        return;

    
    float3 toPrevCam = gPrevCamPos - currentSD.posW;
    float distToPrevCam = length(toPrevCam);
    toPrevCam /= distToPrevCam;

    float3 prevScreenPos = getCameraProjection(gPrevCamViewProjection, currentSD.posW);
    bool isVisible = (all(prevScreenPos.xyz >= 0) && all(prevScreenPos.xyz < 1));

    if (isVisible)
    {
        isVisible = sdh.dispatchShadowRay(currentSD, toPrevCam, distToPrevCam);
    }

    float p0 = currentReservoir.confidenceWeight * currentReservoir.sample.targetFunction;
    float p1 = 0.0;
    float jacobianShift = 0.0;

    if (isVisible)
    {
        float jacobianWeight = getSubpixelLocationJacobian(distToPrevCam, toPrevCam, currentSD.faceN, gPrevCamForward);
        jacobianShift = currentReservoir.sample.subpixelJacobian > 0.0 ? jacobianWeight / currentReservoir.sample.subpixelJacobian : 0.0;
        float targetFunction = getTargetFunction(currentSD, currentReservoir.sample, currentReservoir.pad, toPrevCam, sg);
        uint2 prevPixel = uint2(floor(prevScreenPos.xy * rh.gFrameDim));
        uint prevIndex = rh.getIndex(prevPixel);
        uint c1 = gFrameCount == 0 ? 1 : gReservoirPrev[prevIndex].confidenceWeight;
        p1 = c1 * targetFunction * jacobianShift;
    }

    float sum = p0 + p1;
    float misWeight = sum > 0 ? p0 / sum : 0;

    float baseWeight = misWeight * currentReservoir.sample.targetFunction;
    currentReservoir.w_sum *= baseWeight;


    bool valid = true;

    const float3 currentCamPosW = gScene.camera.getPosition();
    const float3 currentCamForward = normalize(gScene.camera.data.cameraW);
    float newConfidence = currentReservoir.confidenceWeight;
    for (uint i = 0; i < numReservoirs; ++i)
    {
        uint2 splattedPixel = gSplattedPixels[cellOffsets + i];
        uint splattedIndex = rh.getIndex(splattedPixel);

        ShadingData splattingSD = {};
        HitInfo hit = HitInfo(gSplattingHits[splattedPixel]);
        rh.Reservoir splattingReservoir = gReservoirPrev[splattedIndex];

        if (!sdh.getShadingData(hit, splattingReservoir.pad1, splattingSD))
            continue;


        float prevMIS = 0.0;
        float targetFunction = 0.0;
        float jacobianToCurrent = 0.0;
        float jacobianShift = 0.0;

        valid &= isNeighbourValid(splattingSD, currentSD);
        float3 toCurrentCam = currentCamPosW - splattingSD.posW;
        float distToCurrentCam = length(toCurrentCam);
        toCurrentCam /= distToCurrentCam;

        let textureSampler = ExplicitLodTextureSampler(0);
        const uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;

        isVisible = sdh.dispatchShadowRay(splattingSD, toCurrentCam, distToCurrentCam);
        float3 pixelPosToSample = normalize(splattingReservoir.sample.posW - currentSD.posW);
        if (valid & isVisible)
        {
            float p0 = splattingReservoir.sample.targetFunction * splattingReservoir.confidenceWeight;
            jacobianToCurrent = getSubpixelLocationJacobian(distToCurrentCam, toCurrentCam, splattingSD.faceN, currentCamForward);
            jacobianShift = splattingReservoir.sample.subpixelJacobian > 0.0 ? jacobianToCurrent / splattingReservoir.sample.subpixelJacobian : 0.0;
            targetFunction = getTargetFunction(splattingSD, splattingReservoir.sample, splattingReservoir.pad, toCurrentCam, sg);
            float p1 = currentReservoir.confidenceWeight * targetFunction * jacobianShift;
            float sum = p0 + p1;
            prevMIS = sum > 0 ? p0 / sum : 0.0;
            prevMIS = isnan(prevMIS) ? 0.0 : prevMIS;
        }
        float w = prevMIS * targetFunction * splattingReservoir.w_sum * jacobianShift;
        splattingReservoir.sample.targetFunction = targetFunction;
        splattingReservoir.sample.subpixelJacobian = jacobianToCurrent;
        if (currentReservoir.addSample(splattingReservoir.sample, splattingReservoir.pad, w, 0, sg))
        {
            currentHit = gSplattingHits[splattedPixel];
            currentReservoir.pad1 = splattingReservoir.pad1;
        }
    }

    // Use motion vector and bilinear interpolation for new confidence weight
    float2 prevPixel = float2(pixel) + sdh.gMVec[pixel] * rh.gFrameDim;
    int2 prevPixelTopLeft = int2(floor(prevPixel));
    float2 fractionalCoord = saturate(prevPixel - float2(prevPixelTopLeft));
    for (uint x = 0; x < 2; x++) {
        for (uint y = 0; y < 2; y++)
        {
            int2 prevPixel = prevPixelTopLeft + int2(x, y);
            if (any(prevPixel < 0) || any(prevPixel >= rh.gFrameDim))
                continue;

            rh.Reservoir prevReservoir = gReservoirPrev[rh.getIndex(prevPixel)];
            //Exclude non contributing reservoirs
            //if (prevReservoir.pathLength > 0)
            //    continue;
            float bilinearWeight = lerp(1.0 - x, x, fractionalCoord.x) * lerp(1.0 - y, y, fractionalCoord.y);
            newConfidence += bilinearWeight * prevReservoir.confidenceWeight;
        }
    }

    currentReservoir.confidenceWeight = uint(newConfidence);
    currentReservoir.finalizeSample();
    gReservoir[index] = currentReservoir;
    gSplattingHits[pixel] = currentHit;
}
