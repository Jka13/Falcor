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

RWTexture2D<float4> gOutputColor;

StructuredBuffer<rh.Reservoir> gReservoir;

bool dispatchShadowRay(ShadingData sd, float3 toLight, float distanceToLight) {
    // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
    SceneRayQuery<1> rayQuery;
    const Ray ray = Ray(sd.computeNewRayOrigin(), toLight, 0.f, distanceToLight * kShadowRayEpsilon);
    return rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
}

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    uint index = rh.getIndex(pixel);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    let textureSampler = ExplicitLodTextureSampler(0);

    rh.Reservoir reservoir = gReservoir[index];
    rh.Sample sample = reservoir.sample;

    HitInfo hit = sdh.getHitInfo(pixel);
    float3 rayDir = sdh.getRayDir(pixel);
    float4 color = float4(0);
    ShadingData sd = {};
    bool valid = sdh.getShadingData(hit, rayDir, sd);
    float3 connectionDir = normalize(sample.posW - sd.posW);
    float length = distance(sample.posW, sd.posW);
    bool isVisible = dispatchShadowRay(sd, connectionDir, length);
    if (isVisible)
    {
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        let brdfSample = bsdf.eval(sd, connectionDir, sg);
        color = float4(brdfSample * sample.flux * reservoir.w_sum, 1.0); //after sample generation w_sum stores the weight W
    }
    gOutputColor[pixel] = color;
}
