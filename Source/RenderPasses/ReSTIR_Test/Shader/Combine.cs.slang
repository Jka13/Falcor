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
import Helpers.ReservoirHandler;

static const float kShadowRayEpsilon = 0.98f;

cbuffer PerFrame
{
    uint gFrameCount;
}

struct EvaluationData
{
    
}

RWTexture2D<float4> gOutputColor;

StructuredBuffer<rh.Reservoir> gReservoir;

bool dispatchShadowRay(ShadingData sd, float3 toLight, float distanceToLight) {
    // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
    SceneRayQuery<1> rayQuery;
    const Ray ray = Ray(sd.computeNewRayOrigin(), toLight, 0.f, distanceToLight * kShadowRayEpsilon);
    return rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
}

float4 calculatePixelColor(ShadingData sd, rh.Reservoir reservoir, float distanceToLight, float3 toLight, SampleGenerator sg)
{
    let textureSampler = ExplicitLodTextureSampler(0);
    let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
    let bsdfProperties = bsdf.getProperties(sd);
    let brdfSample = bsdf.eval(sd, toLight, sg);
    float4 color = float4(brdfSample * reservoir.sample.flux * reservoir.w_sum / max((distanceToLight * distanceToLight), 0.0001), 1.0); // after sample generation w_sum stores the weight W
    return color;

}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    uint index = rh.getIndex(pixel);
    rh.Reservoir reservoir = gReservoir[index];

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);

    float3 toLight = normalize(reservoir.sample.posW - sd.posW);
    float distanceToLight = distance(reservoir.sample.posW, sd.posW);

    float4 color = float4(0);
    bool isVisible = dispatchShadowRay(sd, toLight, distanceToLight);
    if (isVisible)
    {
        color = calculatePixelColor(sd, reservoir, distanceToLight, toLight, sg);
    }
    gOutputColor[pixel] = color;
}
