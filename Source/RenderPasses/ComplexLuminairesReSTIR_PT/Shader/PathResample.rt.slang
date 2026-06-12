#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import Scene.RaytracingInline;
import Scene.Raytracing;

import Helpers.SceneDataHandler;
import Helpers.PathReservoirHandler;
import Helpers.VPLSampler;

cbuffer UI
{
    uint gPixelRadius;
    float gMinConnectionDistance;
    float gRoughnessThreshold;
}

cbuffer PerFrame
{
    uint gFrameCount;
}

RWStructuredBuffer<prh.PathDebugData> gPathDebugBuffer;
RWStructuredBuffer<vplSampler.DirectVPL> gNEESampleBuffer;

RWStructuredBuffer<prh.CausticReservoir> gCausticReservoir;
RWStructuredBuffer<prh.CausticReservoir> gCausticReservoirPrev;

RWStructuredBuffer<prh.PathReservoir> gPathReservoir;
RWStructuredBuffer<prh.PathReservoir> gPathReservoirPrev;

RWStructuredBuffer<prh.ReconnectionData> gReconnectionData;
RWStructuredBuffer<prh.ReconnectionData> gReconnectionDataPrev;

RWTexture2D<float4> gOutDebug;
RWTexture2D<float4> gOutDebug1;
RWTexture2D<float4> gOutColor;

/** Payload for scatter ray (up to 72B).
*/
struct RayData
{
    PackedHitInfo hit;
    uint materialID;

    /** Create ray payload with default parameters.
    */
    __init()
    {
        HitInfo hitInfo = {};
        hit = hitInfo.getData();
        materialID = 0;
    }
};

float3 resampleCaustic(const uint index, const uint prevIndex, const uint2 pixel, inout SampleGenerator sg)
{
    prh.CausticReservoir reservoir = gCausticReservoir[index];
    prh.CausticReservoir temporalReservoir = gCausticReservoirPrev[prevIndex];
    prh.CausticSample temporalSample = temporalReservoir.sample;
    const uint c = temporalReservoir.confidenceWeight;
    float temporalWeight = temporalSample.w;

    //bool applyMIS = temporalSample.pathLength == reservoir.sample.pathLength;
    //gOutDebug[pixel] = float4(temporalSample.pathLength, reservoir.sample.pathLength, 0, 1);
    //if (applyMIS)
    //{
    //    float misSum = reservoir.confidenceWeight * reservoir.sample.pathPDF + c * temporalSample.pathPDF;
    //    float mis = 0.0;
    //    float temporalMIS = 0.0;

    //    if (misSum > 0)
    //    {
    //        mis = reservoir.confidenceWeight * reservoir.sample.pathPDF / misSum;

    //        temporalMIS = c * temporalSample.pathPDF / misSum;
    //        reservoir.w_sum *= mis;
    //    }
    //    temporalWeight *= temporalMIS;
    //}
    //reservoir.w_sum *= reservoir.sample.targetFunction;
    //reservoir.addSample(temporalSample, c, temporalWeight, sg);
    //reservoir.finalizeSample();

    float3 color = reservoir.sample.pathContribution * reservoir.w_sum;
    //gCausticReservoir[index] = reservoir;
    return color;
}

struct PathData
{
    bool    terminated;         ///< Set to true when path is terminated.
    float3  thp;                ///< Current path throughput. This is updated at each path vertex.
    uint    pathLength;         ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3  origin;             ///< Next path segment origin.
    float   bsdfSamplePDF;      ///< Probability of the BSDF sample
    float3  direction;          ///< Next path segment direction.
    uint    maxPathLength;      ///< Max path length
    float3  wi;                 ///< Incoming Direction
    uint    lobeType;
    float3  lastWeight;
    float   pad;
    float3  lastThp;
    float   pad1;

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Initializes ray payload with default parameters.
    */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.thp = float3(1);
        this.lastThp = float3(1);
        this.origin = float3(0);
        this.direction = float3(0);
        this.wi = float3(0);
        this.bsdfSamplePDF = 0.0;
        this.lobeType = 0;
        this.lastWeight = float3(1);
        this.sg = sg;
    }
};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  toLight;    ///< Ray direction for visibility evaluation (normalized).
    uint    index;

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
        for visibility rays.
        \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 posW, float3 faceN, float3 lightPos)
    {
        float3 rayDir = lightPos - posW;
        return computeRayOrigin(posW, dot(faceN, rayDir) >= 0 ? faceN : -faceN);
    }

    Ray getVisibilityRay() { return Ray(origin, toLight, 0.f, distance); }
};

void updatePathData(const BSDFSample bsdfSample, inout PathData pathData)
{
    pathData.thp *= bsdfSample.weight;
    pathData.lastThp *= pathData.lastWeight;
    pathData.bsdfSamplePDF = bsdfSample.pdf;
    pathData.terminated = !any(pathData.thp > 0.f);
    pathData.lobeType = bsdfSample.lobeType;
    pathData.lastWeight = bsdfSample.weight;
}

void generateScatterRay(const ShadingData sd, const BSDFSample bsdfSample, bool isCurveHit, float3 rayOrigin, inout PathData pathData)
{
    pathData.origin = rayOrigin;
    if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
    {
        pathData.origin = sd.computeNewRayOrigin(false);
    }
    pathData.wi = pathData.direction;
    pathData.direction = bsdfSample.wo;
}

void handleHit(RayData rayData, inout PathData pathData, uint2 pixel)
{
    HitInfo hit = HitInfo(rayData.hit);
    const bool isCurveHit = hit.getType() == HitType::Curve;
    const bool isTriangleHit = hit.getType() == HitType::Triangle;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = {};
    if (!sdh.getShadingData(hit, -pathData.direction, sd))
    {
        pathData.terminated = true;
        return;
    }

    // Create material instance.
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let miProperties = mi.getProperties(sd);

    // Check whether to terminate based on max depth.
    if (pathData.pathLength >= pathData.maxPathLength)
    {
        pathData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeNewRayOrigin();
    }

    //Sample BSDF for the next scatter ray and check for delta distribution
    BSDFSample bsdfSample;
    pathData.terminated = !mi.sample(sd, pathData.sg, bsdfSample, true);
    //{
    //    pathData.terminated = true;
    //    return;
    //}

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    generateScatterRay(sd, bsdfSample, isCurveHit, rayOrigin, pathData);
    updatePathData(bsdfSample, pathData);
    pathData.pathLength++;
}

void traceScatterRay(inout RayData rayData, inout PathData pathData)
{
    RayDesc ray;
    ray.Origin = pathData.origin;
    ray.Direction = pathData.direction;
    ray.TMin = 0.f;
    ray.TMax = FLT_MAX;

    uint rayFlags = 0;      // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

void randomReplay(uint2 pixel, inout RayData rayData, inout PathData pathData)
{
    pathData.direction = -sdh.getRayDir(pixel);
    pathData.wi = pathData.direction;
    rayData.hit = sdh.getPackedHitInfo(pixel);
    handleHit(rayData, pathData, pixel);
    // Follow path into the scene and compute its total contribution.
    for (uint depth = 1; depth < pathData.maxPathLength && !pathData.terminated; depth++)
    {
        traceScatterRay(rayData, pathData);
        handleHit(rayData, pathData, pixel);
    }
}

float3 reconnectPath(uint2 pixel, prh.ReconnectionData reconnectionData, RayData rayData, PathData pathData)
{
    float3 color = float3(0);
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo mainPathHit = HitInfo(rayData.hit);
    ShadingData mainPathSD = {};
    if(!sdh.getShadingData(mainPathHit, -pathData.wi, mainPathSD))
        return color;

    float3 rayOrigin = mainPathSD.computeNewRayOrigin();
    float3 reconnectionDir = normalize(reconnectionData.posW - rayOrigin);
    // Create material instance.
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(mainPathSD, lod, hints);
    let miProperties = mi.getProperties(mainPathSD);

    const float reconnectionDistance = distance(rayOrigin, reconnectionData.posW); 
    const bool isValidReconnection =  miProperties.roughness > gRoughnessThreshold && reconnectionDistance > gMinConnectionDistance;

    if (isValidReconnection && sdh.dispatchShadowRay(rayOrigin, reconnectionDir, reconnectionDistance))
    {
        float3 bsdf = mi.eval(mainPathSD, reconnectionDir, pathData.sg);
        float pdf = mi.evalPdf(mainPathSD, reconnectionDir);
        float3 reconnectionThp = float3(0);
        if (pdf > 0)
            reconnectionThp = bsdf / pdf; 
        color = pathData.lastThp * reconnectionThp * reconnectionData.pathContribution;
    }
    return color;
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateVPLSample(const float3 posW, const float3 faceN, out LightSample ls, uint2 pixel)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    vplSampler.DirectVPL vpl = gNEESampleBuffer[prh.getIndex(pixel)];

    vplSampler.VPLSample sample = {};

    if (!vplSampler.sampleVPL(vpl, posW, sample)) return false;
    
    // Setup returned sample.
    sample.pdf = vpl.pdf;
    ls.Li = sample.pdf > 0.0 ? sample.Li / sample.pdf : float3(0);
    ls.pdf = sample.pdf;
    ls.origin = sample.posW;

    // Prepare shadow ray data.
    // Offset shading position to avoid self-intersection (light position is already offset).
    float3 rayOrigin = ls.getRayOrigin(posW, faceN, ls.origin);
    float3 toLight = ls.origin - rayOrigin;
    ls.distance = length(toLight);
    ls.toLight = toLight / ls.distance;
    ls.origin = rayOrigin;

    return any(ls.Li > 0.0);
}

float3 resampleNEE(uint2 pixel, prh.ReconnectionData reconnectionData, RayData rayData, PathData pathData)
{
    float3 color = float3(0);

    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(rayData.hit);
    ShadingData sd = {};

    if(!sdh.getShadingData(hit, -pathData.wi, sd))
        return color;

    float3 rayOrigin = sd.computeNewRayOrigin();

    LightSample ls = {};
    if (!generateVPLSample(rayOrigin, sd.getOrientedFaceNormal(), ls, pixel)) //TODO: remove pixel
        return color;

    // Create material instance.
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let miProperties = mi.getProperties(sd);

    float3 brdf = max(mi.eval(sd, ls.toLight, pathData.sg), 0.0);
    float3 Lr = pathData.lastThp * brdf * ls.Li;

    if (!any(Lr > 0.0) || !sdh.dispatchShadowRay(ls.getVisibilityRay()))
        return color;

    return Lr;
}

float3 resampleBSDF(uint2 pixel, prh.ReconnectionData reconnectionData, RayData rayData, PathData pathData)
{
    float3 color = float3(0);

    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(rayData.hit);
    ShadingData sd = {};

    if(!sdh.getShadingData(hit, -pathData.wi, sd))
        return color;

    float3 rayOrigin = sd.computeNewRayOrigin();

    // Create material instance.
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let miProperties = mi.getProperties(sd);
    float3 selfEmission = miProperties.emission;

    color = pathData.lastThp * selfEmission;
    gOutDebug1[pixel] = float4(pathData.lastThp, 1);
    return color;
}

float3 resample(uint2 pixel)
{
    float3 color = float3(0);
    uint index = prh.getIndex(pixel);
    prh.ReconnectionData reconnectionData = gReconnectionData[index];

    if(all(reconnectionData.pathContribution <= 0))
        return color;

    SampleGenerator pathSG = SampleGenerator(reconnectionData.seed.xy, reconnectionData.seed.z);
    RayData rayData = RayData();
    PathData pathData = PathData(pathSG);

    switch(reconnectionData.resampleMode)
    {
        case 0:
            pathData.maxPathLength = reconnectionData.neePathLength;
            randomReplay(pixel, rayData, pathData);
            color = resampleNEE(pixel, reconnectionData, rayData, pathData);
            break;
        case 1:
            pathData.maxPathLength = reconnectionData.pathLength;
            randomReplay(pixel, rayData, pathData);
            color = reconnectPath(pixel, reconnectionData, rayData, pathData);
            break;
        case 2: 
            pathData.maxPathLength = reconnectionData.bsdfPathLength;
            randomReplay(pixel, rayData, pathData);
            color = resampleBSDF(pixel, reconnectionData, rayData, pathData);
            break;
        default:
            break;
    }
    return color;
}

float3 resamplePath(const uint index, const uint prevIndex, const uint2 pixel, inout SampleGenerator sg)
{
    prh.PathReservoir reservoir = gPathReservoir[index];
    prh.PathReservoir temporalReservoir = gPathReservoirPrev[prevIndex];
    prh.PathSample temporalSample = temporalReservoir.sample;
    const uint c = temporalReservoir.confidenceWeight;
    float temporalWeight = temporalSample.w;

    float3 color = reservoir.sample.pathContribution * reservoir.w_sum;
    //gPathReservoir[index] = reservoir;
    return color;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void miss(inout RayData rayData)
{
    //write invalid hit
    HitInfo hitInfo = {};
    rayData.hit = hitInfo.getData();
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.hit = HitInfo(triangleHit).getData();
    rayData.materialID = gScene.getMaterialID(triangleHit.instanceID);
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    //gOutDebug[pixel] = float4(0);
    gOutDebug1[pixel] = float4(0);

    sdh.copyVBuffer(pixel);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    uint index = prh.getIndex(pixel);

    uint2 prevPixel = pixel;
    //float2 rnd = sampleNext2D(sg);
    //bool valid = sdh.applyMotionVector(prevPixel, rnd);
    uint prevIndex = prh.getIndex(prevPixel);

    float3 color = float3(0);

    prh.PathReservoir temporalReservoir = gPathReservoirPrev[prevIndex];
    prh.PathReservoir reservoir = gPathReservoir[index];

    //color += resampleCaustic(index, prevIndex, pixel, sg);
    //color += resamplePath(index, prevIndex, pixel, sg);
    color += resample(pixel);

    gOutColor[pixel] = float4(color, 1);
}
