#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Scene.Material.ShadingUtils;
import Rendering.Lights.LightHelpers;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;

#define USE_SUBGROUP_STORE 0

#if USE_EMISSIVE_LIGHT == 1
cbuffer Light
{
    EmissiveLightSampler gLightBVHSampler;
};
#endif

cbuffer PerFrame
{
    uint gFrameCount; // Frame count since scene was loaded.
    uint2 gFrameDim;
};

cbuffer CB
{
    uint gMaxRecursion; //Max photon recursion depths
    uint gFlags; //Flags for the pass. See GenerationFlags enum
};

struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  dir;        ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

struct CameraPathVertex
{
    float3 luminance;
    float3 thp;
    float3 posW;
    float3 direction;
    PackedHitInfo hit;

    static CameraPathVertex create()
    {
        CameraPathVertex cpv;
        cpv.thp = float3(0);
        cpv.posW = float3(0);
        cpv.direction = float3(0);
        cpv.hit = PackedHitInfo(uint4(0));
        return cpv;
    }
};
// inputs
RWStructuredBuffer<CameraPathVertex> gCameraPaths;

enum class GenerationFlags // : uint32_t
{
    None                = 0x00,
    AlphaTest           = 0x01,     //Enable Alpha test
    AdjShadingNormals   = 0x02,     //Adjust the shading normals
    EnableCaustics      = 0x04,     //Enable caustic photons.
    DeltaRejection      = 0x08,     //Accept every non delta material as diffuse
    DeltaRejReqDiff     = 0x10,     //Reqire diffuse part for delta rejection
    AnalyticLight       = 0x20,     //Use analytic light photons
};

static const uint kRayFlagTerminatedMiss = 1;
static const uint kRayFlagTerminatedHit = 2;


/** Payload for scatter ray (80B).
*/
struct RayData
{
    PackedHitInfo hit;
    uint _pad;
    uint terminated; ///< Set to true when path is terminated.

    static RayData create()
    {
        RayData d;
        d.hit = {};
        d.terminated = 0;
        return d;
    }
};

struct PathData
{
    float3 directLightLuminance;
    float3 thp;
    float3 origin;     ///< Next path segment origin.
    float3 direction;  ///< Next path segment direction.
    float3 posW;
    bool isEmissive;

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).


    static PathData create()
    {
        PathData d;
        d.directLightLuminance = float3(0);
        d.thp = float3(1);
        d.origin = float3(0);
        d.direction = float3(0);
        d.posW = float3(0);
        d.isEmissive = false;
        return d;
    }
};

struct ShadowData
{
    bool shadowed;
};

ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

void handleHit(inout RayData rayData, inout PathData pathData)
{
    // Evaluate Falcor's material parameters at the hit point.
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(rayData.hit);
    if (!hit.isValid())
    {
        rayData.terminated | kRayFlagTerminatedHit;
        return;
    }
    const bool isCurveHit = hit.getType() == HitType::Curve;
    ShadingData sd = loadShadingData(hit, pathData.origin, pathData.direction, lod);

    //Adjust shading normals if desired
    uint hints = 0;
    //if (gFlags & (uint) GenerationFlags::AdjShadingNormals)
    {
        hints |= (uint) MaterialInstanceHints::AdjustShadingNormal;
    }
    
    BSDFSample bsdfSample;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);

    bool valid = bsdf.sample(sd, pathData.sg, bsdfSample, true/*ImportanceSampling*/);
    
    bool transmission = bsdfSample.isLobe(LobeType::Transmission); //false is for transmissions 
    float3 rayOrigin; 
    if (isCurveHit) {
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else {
        rayOrigin = sd.computeNewRayOrigin(!transmission);
    }
    pathData.origin = rayOrigin;
    float3 reverseW_o = -pathData.direction;
    pathData.direction = bsdfSample.wo;
    pathData.isEmissive = luminance(bsdfProperties.emission) > 0;
    pathData.thp *= valid ? bsdfSample.weight : float3(0);
    //if throughput is 0, return
    if (!valid || all(pathData.thp <= 0))
    {
        rayData.terminated |= kRayFlagTerminatedHit;
        return;
    }

    LightSample ls;
    pathData.directLightLuminance = float3(0);
    if (!generateEmissiveSample(sd, false, pathData.sg, ls)) return;
    if (isShadowed(ls)) return;
    sd = loadShadingData(hit, pathData.origin, -ls.dir, lod);
    pathData.directLightLuminance = ls.Li * evaluateBRDF(sd, pathData.sg, reverseW_o); 
}

float3 evaluateBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_o)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    return max(mi.eval(sd, w_o, sg), 0);
}

bool generateEmissiveSample(const ShadingData sd, const bool upperHemisphere, inout SampleGenerator sg, inout LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.
#if USE_EMISSIVE_LIGHT == 0
    return false;
    #endif

    float3 faceNormal = sd.frontFacing ? sd.faceN : -sd.faceN;

    TriangleLightSample tls;
    if (!gLightBVHSampler.sampleLight(sd.posW, faceNormal, upperHemisphere, sg, tls)) return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = computeRayOrigin(sd.posW, dot(faceNormal, lightPos - sd.posW) >= 0 ? faceNormal : -faceNormal);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    return any(ls.Li > 0.f);
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    rayData.terminated |= kRayFlagTerminatedMiss;
}

[shader("miss")]
void shadowMiss(inout ShadowData shadowData : SV_RayPayload)
{
    shadowData.shadowed = false;
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

//Use anyhit for alpha test if activated
[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    //if (gFlags & (uint) GenerationFlags::AlphaTest)
    if (true)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
            IgnoreHit();
    }
}

//Use anyhit for alpha test if activated
[shader("anyhit")]
void shadowAnyHit(inout ShadowData shadowData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    //if (gFlags & (uint) GenerationFlags::AlphaTest)
    if (true)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
            IgnoreHit();
    }
}

uint pixelToPathIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * gMaxRecursion;
}

bool isShadowed(const LightSample ls)
{
    Ray ray = ls.getVisibilityRay();
    RayDesc shadowRay = ray.toRayDesc();

    ShadowData shadowData;
    shadowData.shadowed = true;

    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, rayTypeCount, 1 /* missIdx */, shadowRay, shadowData);
    return shadowData.shadowed;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    uint index = pixelToPathIndex(launchIndex, launchDim);

    // Prepare ray payload.
    RayData rayData = RayData.create();

    Ray cameraRay = gScene.camera.computeRayPinhole(launchIndex, launchDim, false);
    RayDesc ray = cameraRay.toRayDesc();

    PathData pathData = PathData.create();
    pathData.sg = SampleGenerator(launchIndex, gFrameCount + 7 * 17);
    pathData.direction = ray.Direction;
    pathData.origin = ray.Origin;

    uint rayFlags = 0;
    uint i = 0;
    for (; i < gMaxRecursion && rayData.terminated == 0; ++i)
    {
        CameraPathVertex vertex;
        vertex.thp = pathData.thp;
        vertex.direction = pathData.direction;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
        handleHit(rayData, pathData);
        //Stop if a light source was hit
        if (pathData.isEmissive || rayData.terminated > 0)
            break;

        vertex.posW = pathData.origin;
        vertex.hit = rayData.hit;
        vertex.luminance = pathData.directLightLuminance; 
        gCameraPaths[index + i] = vertex;

        ray.Origin = pathData.origin;
        ray.Direction = pathData.direction;
    }
    for (; i < gMaxRecursion; ++i)
    {
        CameraPathVertex cpv;
        cpv.luminance = float3(0);
        cpv.thp = float3(0);
        cpv.posW = float3(0);
        cpv.direction = float3(0);
        cpv.hit = PackedHitInfo(uint4(0));
        gCameraPaths[index + i] = cpv;
    }
}
