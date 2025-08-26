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
}

struct Photon
{
    float3 posW;
    float3 direction;
    float3 flux;
}

StructuredBuffer<Photon> gPhotonBuffer;
RWTexture2D<float4> gOutColor;

bool samplePointLight(const float3 shadingPosW, const LightData light, out AnalyticLightSample ls)
{
    // Get the position and normal.
    ls.posW = light.posW;
    ls.normalW = light.dirW;

    // Compute direction and distance to light.
    // The distance is clamped to a small epsilon to avoid div-by-zero below.
    float3 toLight = ls.posW - shadingPosW;
    float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance;

    // Calculate the falloff for spot-lights.
    float cosTheta = -dot(ls.dir, light.dirW);
    float falloff = 1.f;
    if (cosTheta < light.cosOpeningAngle)
    {
        falloff = 0.f;
    }
    else if (light.penumbraAngle > 0.f)
    {
        float deltaAngle = light.openingAngle - acos(cosTheta);
        falloff = smoothstep(0.f, light.penumbraAngle, deltaAngle);
    }

    // Compute incident radiance at shading point.
    ls.Li = light.intensity * falloff / distSqr;

    // For a point light, the PDF with respect to solid angle is a Dirac function. Set to zero.
    ls.pdf = 0.f;

    return true;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);

    //sample photon buffer
    uint photonIndex = uint(sampleNext1D(sg) * (gPhotonCount + 1));
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
    float3 pdf = bsdf.evalPdf(sd, toLight);
    if (any(pdf == 0))
    {
        gOutColor[pixel] = float4(0.0);
        return;
    }
    brdf /= pdf;

    float3 luminance = (gConeExponent + 1) / M_2PI * lightSample.flux * pow(cosTheta, gConeExponent) / max(dist * dist, 0.01);
    gOutColor[pixel] = float4(brdf * luminance, 1.0);
}
