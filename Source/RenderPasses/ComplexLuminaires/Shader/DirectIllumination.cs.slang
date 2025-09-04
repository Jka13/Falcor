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

import Helpers.SceneDataHandler;

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer CB
{
    uint gPhotonCount;
    float gConeExponent;
    float gCosOpeningAngle;
    float gPenumbraAngle;
}

struct Photon
{
    float3 posW;
    float3 direction;
    float3 flux;
}

StructuredBuffer<Photon> gPhotonBuffer;
RWTexture2D<float4> gOutColor;


float3 samplePointLight(const float3 shadingPosW, const Photon light)
{
    float kMinLightDistSqr = 1e-9f;
    // Compute direction and distance to light.
    // The distance is clamped to a small epsilon to avoid div-by-zero below.
    float3 toLight = light.posW - shadingPosW;
    float distSqr = dot(toLight, toLight);
    distSqr =max(distSqr, kMinLightDistSqr); 
    float distance = sqrt(distSqr);
    float3 dir = toLight / distance;

    // Calculate the falloff for spot-lights.
    float cosTheta = -dot(dir, light.direction);
    float falloff = 1.f;
    float openingAngle = acos(gCosOpeningAngle); //light property
    if (cosTheta < gCosOpeningAngle)
    {
        falloff = 0.f;
    }
    else if (gPenumbraAngle > 0.f)
    {
        float deltaAngle = openingAngle - acos(cosTheta);
        falloff = smoothstep(0.f, gPenumbraAngle, deltaAngle);
    }
    float radius = tan(openingAngle);
    float3 normalizationFactor = 1/ (M_2PI * (1 - gCosOpeningAngle));
    // Compute incident radiance at shading point.
    return normalizationFactor * light.flux * falloff / distSqr;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);

    //sample photon buffer
    uint photonIndex = uint(sampleNext1D(sg) * gPhotonCount);
    Photon lightSample = gPhotonBuffer[photonIndex];

    float dist = distance(sd.posW, lightSample.posW);
    float3 toLight = normalize(lightSample.posW - sd.posW);
    float cosTheta = dot(lightSample.direction, -toLight);
    if (cosTheta <= 0)
    {
        gOutColor[pixel] = float4(0.0);
        return;
    }
    bool isVisible = sdh.dispatchShadowRay(sd, toLight, dist);
    if (!isVisible)
    {
        gOutColor[pixel] = float4(0.0);
        return;
    }

    let lod = ExplicitLodTextureSampler(0.0);
    uint hints = 0;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    float3 brdf = max(bsdf.eval(sd, toLight, sg), 0);
    float3 luminance = samplePointLight(sd.posW, lightSample);
    //float3 luminance = (gConeExponent + 1) / M_2PI * lightSample.flux * pow(cosTheta, gConeExponent) / max(dist * dist, 0.01);
    gOutColor[pixel] = float4(brdf * luminance, 1.0);
}
