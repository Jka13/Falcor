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
import Helpers.VPLSampler;

cbuffer PerFrame
{
    uint gFrameCount;
}

RWTexture2D<float4> gOutputColor;
RWTexture2D<PackedHitInfo> gSplattingHits;

StructuredBuffer<rh.Reservoir> gReservoir;

[numthreads(16, 16, 1)]
void main(uint2 pixel : SV_DispatchThreadID)
{
    gOutputColor[pixel] = float4(0);
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    float3 color = float3(0.0);

    uint index = rh.getIndex(pixel);
    rh.Reservoir reservoir = gReservoir[index];

    const float3 rayDir = sdh.getRayDir(pixel);
    const HitInfo hit = HitInfo(gSplattingHits[pixel]);

    ShadingData sd = {};
    bool valid = sdh.getShadingData(hit, rayDir, sd);

    float3 toLight = normalize(reservoir.sample.posW - sd.posW);
    float distanceToLight = distance(reservoir.sample.posW, sd.posW);

    bool isVisible = sdh.dispatchShadowRay(sd, toLight, distanceToLight);
    let textureSampler = ExplicitLodTextureSampler(0);
    let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
    let bsdfProperties = bsdf.getProperties(sd);
    float3 brdfSample = bsdf.eval(sd, toLight, sg);
    if (isVisible)
    {
        float3 flux = sh.samplePointLight(sd.posW, reservoir.sample.posW, reservoir.pad, reservoir.sample.flux);
        color = brdfSample * flux * reservoir.w_sum; // after sample generation w_sum stores the weight W
    }
    gOutputColor[pixel] = float4(color + bsdfProperties.emission, 1.0);
}
