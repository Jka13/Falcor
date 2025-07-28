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
    uint M = 32;
    uint2 pixel = DispatchRaysIndex().xy;
    uint index = rh.getIndex(pixel);
    uint2 launchDim = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    let textureSampler = ExplicitLodTextureSampler(0);

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);
    let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
    let bsdfProperties = bsdf.getProperties(sd);

    rh.Reservoir reservoir = rh.Reservoir();
    if (!valid)
    {
        gReservoir[index] = reservoir;
        return;
    }

    float3 brdfSample = 0;
    float acceptedP_hat = 0;
    for (uint i = 0; i < M; ++i)
    {
        TriangleLightSample tls = {};
        valid = gEmissiveLightSampler.sampleLight(sd.posW, sd.faceN, true, sg, tls);
        float3 flux = float3(0.0, 0.0, 0.0);
        if (!valid)
            continue;
        float3 toLight = normalize(tls.posW - sd.posW);
        flux = tls.Le;
        brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float pdf = luminance(tls.pdf);
        float w = 0;
        rh.Sample sample = rh.Sample();
        sample.flux = flux;
        sample.posW = tls.posW;
        float p_hat = luminance(flux * brdfSample);
        if (pdf > 0)
            w = p_hat / (pdf * M);
        float distance = distance(tls.posW, sd.posW);
        sample.p_hat = p_hat / max((distance * distance), 0.0001);
        reservoir.addSample(sample, w, 0, sg);
    }
    reservoir.confidenceWeight = 1;
    reservoir.finalizeSample();
    gReservoir[index] = reservoir;
}
