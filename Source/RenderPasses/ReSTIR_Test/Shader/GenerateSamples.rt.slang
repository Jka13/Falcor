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
}

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

struct LightSampleData
{
    float w;
    float targetFunction;

    __init(TriangleLightSample tls, ShadingData sd, BSDFProperties bsdfProperties)
    {
        w = 0;
        targetFunction = 0;

        float3 toLight = normalize(tls.posW - sd.posW);
        float lightSamplePDF = luminance(tls.pdf);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);

        float MISWeight =  1 / float(gNumberOfBSDFSamples + gNumberOfLightSamples);

        float tmp = luminance(tls.Le) * brdfSample;
        if (lightSamplePDF > 0)
            this.w = MISWeight * tmp / lightSamplePDF;

        float distanceToLight = distance(tls.posW, sd.posW);
        float distanceSquared = distanceToLight * distanceToLight;
        if (distanceSquared > 0)
            this.targetFunction = tmp / distanceSquared;
    }
}

struct BSDFSampleData
{
    float w;
    float targetFunction;

    __init(BSDFLightSample bsdfLS, ShadingData sd, BSDFProperties bsdfProperties)
    {
        w = 0;
        targetFunction = 0;
        float3 toLight = normalize(bsdfLS.posW - sd.posW);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);

        float MISWeight =  1 / float(gNumberOfBSDFSamples + gNumberOfLightSamples);

        float distanceToLight = distance(bsdfLS.posW, sd.posW);
        if (distanceToLight > 0)
        {
            float distanceSquared = distanceToLight * distanceToLight;
            targetFunction = luminance(bsdfLS.Le) * brdfSample / distanceSquared;
        }
        float bsdfSamplePDF = luminance(bsdfLS.pdf);
        if (bsdfSamplePDF > 0)
            w = MISWeight * targetFunction / bsdfSamplePDF;
    }
}

struct BSDFLightSample
{
    float3 posW;
    float3 Le;
    float3 pdf;

    __init(float3 posWIn, float3 LeIn, float3 pdfIn)
    {
        posW = posWIn;
        Le = LeIn;
        pdf = pdfIn;
    }
}

RWStructuredBuffer<rh.Reservoir> gReservoir;

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
}

float getTargetFunction(ShadingData sd, rh.Sample sample)
{
        let textureSampler = ExplicitLodTextureSampler(0);
        float3 toLight = normalize(sample.posW - sd.posW);
        float distanceToLight = distance(sample.posW, sd.posW);
        float squaredDistance = distanceToLight * distanceToLight;
        let bsdf = gScene.materials.getMaterialInstance(sd, textureSampler, 0);
        let bsdfProperties = bsdf.getProperties(sd);
        float brdfSample = sdh.evalApproximateBRDF(sd, bsdfProperties, toLight);
        float targetFunction = 0;
        if (squaredDistance > 0) 
            targetFunction = brdfSample * luminance(sample.flux) / squaredDistance;
        return targetFunction;
}

float calcMISWeight(ShadingData sd, rh.Sample sample, rh.Reservoir base)
{
    //float3 toLight = normalize(neighbour.reservoir.sample.posW - base.sd.posW);
    //float distanceToLight = distance(neighbour.reservoir.sample.posW, base.sd.posW);
    //bool isVisible = sdh.dispatchShadowRay(base.sd, toLight, distanceToLight);
    //float targetFunction = 0;
    //if (isVisible)
    float targetFunction = getTargetFunction(sd, sample);
    float c_sum = targetFunction + base.sample.targetFunction;
    float misWeight = 0;
    if (c_sum > 0)
    {
        misWeight = base.sample.targetFunction / c_sum;
    }
    return misWeight;
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
        LightSampleData sampleData = LightSampleData(tls, sd, bsdfProperties);
        rh.Sample sample = rh.Sample(tls.Le, tls.posW, sampleData.targetFunction);
        reservoir.addSample(sample, sampleData.w, 0, sg);
    }

    RayData rayData = RayData();
    RayDesc ray = {};

    for (uint i = 0; i < gNumberOfBSDFSamples; ++i)
    {
        BSDFSample bsdfSample;
        valid = bsdf.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!valid)
            continue;
        bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        ray.Direction = bsdfSample.wo;
        ray.Origin = sd.computeNewRayOrigin(!transmission);
        ray.TMax = 10000.0;
        ray.TMin = 0.0001;

        uint rayFlags = 0;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        ShadingData bsdfSampleSd = {};
        HitInfo hit = HitInfo(rayData.hit);
        valid = sdh.getShadingData(hit, -ray.Direction, bsdfSampleSd);
        if (!valid)
            continue;
        let bsdf1 = gScene.materials.getMaterialInstance(bsdfSampleSd, textureSampler, 0);
        let bsdfPropertiesSample = bsdf1.getProperties(bsdfSampleSd);

        BSDFLightSample bsdfLS = BSDFLightSample(bsdfSampleSd.posW, bsdfPropertiesSample.emission, bsdfSample.pdf);
        BSDFSampleData bsdfSampleData = BSDFSampleData(bsdfLS, sd, bsdfProperties);
        rh.Sample sample = rh.Sample(bsdfPropertiesSample.emission, bsdfSampleSd.posW, bsdfSampleData.targetFunction);
        reservoir.addSample(sample, bsdfSampleData.w, 0, sg);
    }
    reservoir.confidenceWeight = 1;
    reservoir.finalizeSample();

    float3 toLight = normalize(reservoir.sample.posW - sd.posW);
    float distanceToLight = distance(reservoir.sample.posW, sd.posW);
    bool isVisible = sdh.dispatchShadowRay(sd, toLight, distanceToLight);
    reservoir.updateVisibility(isVisible);
    gReservoir[index] = reservoir;
}
