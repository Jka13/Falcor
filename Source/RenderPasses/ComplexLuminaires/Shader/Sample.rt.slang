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
    PackedHitInfo hit;    

    /** Create ray payload with default parameters.
    */
    __init()
    {
        hit = uint4(0);
    }
};

cbuffer UI
{
    uint gNumberOfLightSamples;
    uint gNumberOfBSDFSamples;
    uint gNumberOfLuminaireSamples;
    uint gLuminaireSampleCount;
    float gCosOpeningAngle;
    float gPenumbraAngle;
}

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

struct Photon
{
    float3 posW;
    float3 direction;
    float3 flux;

    __init()
    {
        posW = float3(0);
        direction = float3(0);
        flux = float3(0);
    }
}

struct LightSampleData
{
    float w;
    float targetFunction;

    __init(TriangleLightSample tls, ShadingData sd, BSDFProperties bsdfProperties, float MISWeight)
    {
        w = 0;
        targetFunction = 0;

        float3 toLight = normalize(tls.posW - sd.posW);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        targetFunction =luminance(tls.Le) * brdfSample;

        if (tls.pdf > 0)
        {
            w = MISWeight * targetFunction / tls.pdf;
        }
    }
}

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

RWStructuredBuffer<rh.Reservoir> gReservoir;
StructuredBuffer<Photon> gLuminaireSamples;

[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.hit = HitInfo(triangleHit).getData();
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    HitInfo hitInfo = {};
    rayData.hit = hitInfo.getData();
}

[shader("raygeneration")]
void rayGen()
{
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

    for (uint i = 0; i < gNumberOfLightSamples; ++i)
    {
        TriangleLightSample tls = {};
        bool valid = gEmissiveLightSampler.sampleLight(sd.posW, sd.faceN, true, sg, tls);
        if (!valid)
            continue;
        float scatterPDF = luminance(bsdf.evalPdf(sd, tls.dir, true));
        float MISWeight = tls.pdf / (scatterPDF * float(gNumberOfBSDFSamples) + tls.pdf * float(gNumberOfLightSamples));
        LightSampleData sampleData = LightSampleData(tls, sd, bsdfProperties, MISWeight);
        rh.Sample sample = rh.Sample(tls.Le, tls.posW, sampleData.targetFunction);
        reservoir.addSample(sample, sampleData.w, 0, sg);
    }

    for (uint i = 0; i < gNumberOfLuminaireSamples; ++i)
    {
        uint rndIndex = gLuminaireSampleCount * sampleNext1D(sg);
        Photon luminaireSample = gLuminaireSamples[rndIndex];
        float3 luminance = samplePointLight(sd.posW, luminaireSample) / float(gLuminaireSampleCount);
        if (all(luminance <= 0))
            continue;

        float MISWeight = 1 / float(gNumberOfLuminaireSamples);

        TriangleLightSample tls = {};
        tls.Le = luminance;
        tls.posW = luminaireSample.posW;
        tls.pdf = 1 / float(gLuminaireSampleCount);

        LightSampleData sampleData = LightSampleData(tls, sd, bsdfProperties, MISWeight);
        rh.Sample sample = rh.Sample(tls.Le, tls.posW, sampleData.targetFunction);
        reservoir.addSample(sample, sampleData.w, 0, sg);
    }

    RayData rayData = RayData();
    RayDesc ray = {};
    ray.TMax = 10000.0;
    ray.TMin = 0.0001;

    for (uint i = 0; i < gNumberOfBSDFSamples; ++i)
    {
        BSDFSample bsdfSample = {};
        valid = bsdf.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!valid)
            continue;
        bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        ray.Direction = bsdfSample.wo;
        ray.Origin = sd.computeNewRayOrigin(!transmission);

        uint rayFlags = 0;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        ShadingData lightSd = {};
        HitInfo hit = HitInfo(rayData.hit);
        valid = sdh.getShadingData(hit, -ray.Direction, lightSd);
        if (!valid)
            continue;
        let lightBsdf = gScene.materials.getMaterialInstance(lightSd, textureSampler, 0);
        let lightBsdfProperties = lightBsdf.getProperties(lightSd);

        if (all(lightBsdfProperties.emission <= 0))
            continue;

        TriangleHit triangleHit = hit.getTriangleHit();
        TriangleLightHit tlh;
        tlh.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
        tlh.posW = lightSd.posW;
        tlh.normalW = lightSd.getOrientedFaceNormal();

        float lightPDF = luminance(gEmissiveLightSampler.evalPdf(sd.posW, sd.faceN, true, tlh));

        TriangleLightSample tls = {};
        tls.Le = lightBsdfProperties.emission;
        tls.posW = lightSd.posW;
        tls.pdf = bsdfSample.pdf;

        float MISWeight = tls.pdf / (tls.pdf * float(gNumberOfBSDFSamples) + lightPDF * float(gNumberOfLightSamples));

        LightSampleData sampleData = LightSampleData(tls, sd, bsdfProperties, MISWeight);
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
