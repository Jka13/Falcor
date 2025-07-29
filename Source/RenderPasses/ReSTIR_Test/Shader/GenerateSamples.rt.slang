#include "Scene/SceneDefines.slangh"
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
import Scene.Raytracing;

import Helpers.SceneDataHandler;
import Helpers.ReservoirHandler;

/** Payload for scatter ray (80B).
*/
struct RayData
{
    float3  thp;            ///< Current path throughput. This is updated at each path vertex.
    uint   packedNormal;   ///< Face normal encoded in 16 bit polar coordinates
    float3  origin;         ///< Next path segment origin.
    uint terminated; ///< Set to true when path is terminated.
    float3  direction;      ///< Next path segment direction.
    bool    diffuseHit;     ///< saves if the his is diffuse
    bool isEmissive;        ///<If the hit is emissive
    float distance;

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static RayData create()
    {
        RayData d;
        d.terminated = 0;
        d.packedNormal = 0;
        d.thp = float3(1);
        d.origin = float3(0);
        d.direction = float3(0);
        d.isEmissive = false;
        d.distance = 0.f;
        return d;
    }
};

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

struct SampleData
{
    float w;
    float targetFunction;

    __init(TriangleLightSample tls, ShadingData sd, BSDFProperties bsdfProperties, uint M)
    {
        float3 toLight = normalize(tls.posW - sd.posW);
        float distanceToLight = distance(tls.posW, sd.posW);
        float lightSamplePDF = luminance(tls.pdf);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float tmp = luminance(tls.Le * brdfSample);
        if (lightSamplePDF > 0)
            w =  tmp / (lightSamplePDF * M);
        targetFunction = tmp / max((distanceToLight * distanceToLight), 0.0001);
    }
}

RWStructuredBuffer<rh.Reservoir> gReservoir;

[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes) {}

[shader("closesthit")]
void closestHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes) {}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload){}

[shader("raygeneration")]
void rayGen()
{
    const uint M = 32;
    uint2 pixel = DispatchRaysIndex().xy;
    uint index = rh.getIndex(pixel);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    rh.Reservoir reservoir = rh.Reservoir();

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);
    if (!valid)
    {
        gReservoir[index] = reservoir;
        return;
    }

    let textureSampler = ExplicitLodTextureSampler(0);
    let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
    let bsdfProperties = bsdf.getProperties(sd);

    for (uint i = 0; i < M; ++i)
    {
        TriangleLightSample tls = {};
        bool valid = gEmissiveLightSampler.sampleLight(sd.posW, sd.faceN, true, sg, tls);
        if (!valid)
            continue;
        SampleData sampleData = SampleData(tls, sd, bsdfProperties, M);
        rh.Sample sample = rh.Sample(tls.Le, tls.posW, sampleData.targetFunction);
        reservoir.addSample(sample, sampleData.w, 0, sg);
    }
    reservoir.confidenceWeight = 1;
    reservoir.finalizeSample();
    float3 toLight = normalize(reservoir.sample.posW - sd.posW);
    float distanceToLight = distance(reservoir.sample.posW, sd.posW);
    bool isVisible = sdh.dispatchShadowRay(sd, toLight, distanceToLight);
    reservoir.updateVisibility(isVisible);
    gReservoir[index] = reservoir;
}
