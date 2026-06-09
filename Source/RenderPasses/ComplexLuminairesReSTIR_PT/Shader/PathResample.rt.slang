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

RWStructuredBuffer<prh.CausticReservoir> gCausticReservoir;
RWStructuredBuffer<prh.CausticReservoir> gCausticReservoirPrev;

RWStructuredBuffer<prh.PathReservoir> gPathReservoir;
RWStructuredBuffer<prh.PathReservoir> gPathReservoirPrev;

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
    float3  direction;          ///< Next path segment direction.
    float   bsdfSamplePDF;      ///< Probability of the BSDF sample
    uint    maxPathLength;      ///< Max path length

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Initializes ray payload with default parameters.
    */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.thp = float3(1);
        this.origin = float3(0);
        this.direction = float3(0);
        this.bsdfSamplePDF = 0.0;
        this.sg = sg;
    }
};

void updatePathData(const BSDFSample bsdfSample, inout PathData pathData)
{
    pathData.thp *= bsdfSample.weight;
    pathData.bsdfSamplePDF = bsdfSample.pdf;
    pathData.terminated = !any(pathData.thp > 0.f);
}

void generateScatterRay(const ShadingData sd, const BSDFSample bsdfSample, bool isCurveHit, float3 rayOrigin, inout PathData pathData)
{
    pathData.origin = rayOrigin;
    if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
    {
        pathData.origin = sd.computeNewRayOrigin(false);
    }
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
    if (pathData.pathLength > pathData.maxPathLength)
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

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    generateScatterRay(sd, bsdfSample, isCurveHit, rayOrigin, pathData);
    updatePathData(bsdfSample, pathData);
    prh.PathDebugData.Node node = {};
    node.thp = pathData.thp;
    node.posW = pathData.origin;
    gPathDebugBuffer[prh.getIndex(pixel)].path[pathData.pathLength] = node;
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

void hybridShift(uint2 pixel, inout prh.PathReservoir reservoir)
{
    prh.ReconnectionData reconnectionData = reservoir.sample.rd;
    if(!any(reconnectionData.pathContribution > 0))
        return;
    SampleGenerator pathSG = reconnectionData.seed;
    RayData rayData = RayData();
    PathData pathData = PathData(pathSG);
    pathData.maxPathLength = reconnectionData.pathLength;
    pathData.maxPathLength = 1;

    pathData.direction = -sdh.getRayDir(pixel);
    rayData.hit = sdh.getPackedHitInfo(pixel);
    handleHit(rayData, pathData, pixel);
    // Follow path into the scene and compute its total contribution.
    for (uint depth = 1; depth <= pathData.maxPathLength && !pathData.terminated; depth++)
    {
        traceScatterRay(rayData, pathData);
        handleHit(rayData, pathData, pixel);
    }
    //gOutDebug[pixel] += float4(0, pathData.pathLength, 0, 0);
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo mainPathHit = HitInfo(rayData.hit);
    ShadingData mainPathSD = {};
    bool valid = sdh.getShadingData(mainPathHit, -pathData.direction, mainPathSD);

    float3 rayOrigin = mainPathSD.computeNewRayOrigin();
    float3 reconnectionDir = normalize(reconnectionData.posW - rayOrigin);
    // Create material instance.
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(mainPathSD, lod, hints);
    let miProperties = mi.getProperties(mainPathSD);
    BSDFSample bsdfSample = {};
    mi.sample(mainPathSD, pathData.sg, bsdfSample, true);

    HitInfo offPathHit = HitInfo(reconnectionData.reconnectionHit);
    ShadingData offPathSD = {};
    if (!sdh.getShadingData(offPathHit, -reconnectionDir, offPathSD))

    // Create material instance.
    let offPathMI = gScene.materials.getMaterialInstance(offPathSD, lod, hints);
    let offPathMIProperties = offPathMI.getProperties(offPathSD);
    bsdfSample = {};
    offPathMI.sample(offPathSD, pathData.sg, bsdfSample, true);

    const float reconnectionDistance = distance(mainPathSD.posW, offPathSD.posW); 
    const bool isValidReconnection = min(offPathMIProperties.roughness, miProperties.roughness) > gRoughnessThreshold && reconnectionDistance > gMinConnectionDistance;
    const bool isVisible = sdh.dispatchShadowRay(rayOrigin, reconnectionDir, reconnectionDistance);

    float3 bsdf = mi.eval(mainPathSD, reconnectionDir, pathData.sg);
    float pdf = 1.0;
    pdf = mi.evalPdf(mainPathSD, reconnectionDir);
    float3 bsdf1 = offPathMI.eval(offPathSD, reconnectionData.w_o, pathData.sg);
    float pdf1 = 1.0;
    pdf1 = mi.evalPdf(mainPathSD, reconnectionData.w_o);
    //gOutColor[pixel] = float4(pathData.thp * bsdf * bsdf1 * reconnectionData.pathContribution / (reconnectionData.pathPDF * reconnectionDistance * reconnectionDistance * pdf * pdf1), 1);
    float3 color = float3(0);
    gOutDebug[pixel] = float4(reconnectionData.pathContribution, 1);
    color = pathData.thp * reconnectionData.pathContribution;
    gOutColor[pixel] = float4(color, 1);
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

    color += resampleCaustic(index, prevIndex, pixel, sg);
    color += resamplePath(index, prevIndex, pixel, sg);

    hybridShift(pixel, reservoir);

    gOutColor[pixel] = float4(color, 1);
}
