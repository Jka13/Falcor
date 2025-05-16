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

cbuffer PerFrame
{
    uint gFrameCount; // Frame count since scene was loaded.
    uint2 gFrameDim;
}

cbuffer CB
{
    uint gMaxRecursion; //Max photon recursion depths
    uint gFlags; //Flags for the pass. See GenerationFlags enum
};

// inputs
RWBuffer<float4> gCameraPaths;

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
    bool terminated; ///< Set to true when path is terminated.

    static RayData create()
    {
        RayData d;
        d.hit = {};
        d.terminated = false;
        return d;
    }
};

struct PathData
{
    float3 thp; ///< Current path throughput. This is updated at each path vertex.
    uint packedNormal; ///< Face normal encoded in 16 bit polar coordinates
    float3 origin;     ///< Next path segment origin.
    float3 direction;  ///< Next path segment direction.
    float roughness;
    float reprojection_pdf;
    bool diffuseHit; ///< saves if the his is diffuse
    bool isEmissive; ///<If the hit is emissive

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    static PathData create()
    {
        PathData d;
        d.packedNormal = 0;
        d.thp = float3(1);
        d.origin = float3(0);
        d.direction = float3(0);
        d.reprojection_pdf = 0.f;
        d.isEmissive = false;
        return d;
    }
}

struct ShadowData
{
    bool shadowed;
};

float3 fromWordlToLocal(const in float3 normal, float3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return float3(dot(bitangent, w), dot(tangent, w), dot(normal, w));
}

//transforms direction from local to world space
float3 fromLocalToWorld(const in float3 normal, float3 w)
{
    //transform from local space -> random dir vector in world space
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return w.x * tangent + w.y * bitangent + w.z * normal;
}

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
        rayData.terminated = true;
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
    pathData.roughness = bsdfProperties.roughness;

    bool hasDiffuseParts = (gFlags & (uint) GenerationFlags::DeltaRejReqDiff) ?
    luminance(bsdfProperties.diffuseReflectionAlbedo) > 0 : true;

    bool valid = bsdf.sample(sd, pathData.sg, bsdfSample, true /*ImportanceSampling*/);
    
    bool transmission = bsdfSample.isLobe(LobeType::Transmission);
    float3 rayOrigin ; //false is for transmissions
    if (isCurveHit) {
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else {
        rayOrigin = sd.computeNewRayOrigin(!transmission);
    }
    pathData.origin = rayOrigin;
    pathData.thp *= valid ? bsdfSample.weight : float3(0);
    //if throughput is 0, return
    if (!valid || all(pathData.thp <= 0))
        rayData.terminated |= kRayFlagTerminatedHit;
    pathData.packedNormal = encodeNormal2x16(sd.faceN);
    pathData.direction = bsdfSample.wo;
    pathData.isEmissive = luminance(bsdfProperties.emission) > 0;
    if (gFlags & (uint) GenerationFlags::DeltaRejection)
        pathData.diffuseHit = bsdfSample.isLobe(LobeType::NonDeltaReflection) && hasDiffuseParts;
    else
        pathData.diffuseHit = bsdfSample.isLobe(LobeType::Diffuse) && !transmission;
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

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Prepare ray payload.
    RayData rayData = RayData.create();
    PathData pathData = PathData.create();

    Ray cameraRay = gScene.camera.computeRayPinhole(launchIndex, launchDim, true);
    RayDesc ray = cameraRay.toRayDesc();

    pathData.direction = ray.Direction;
    pathData.origin = ray.Origin;
    
    uint rayFlags = 0;

    for (uint i = 0; i < gMaxRecursion && !rayData.terminated; i++)
    {
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
        handleHit(rayData, pathData);

         //Stop if a light source was hit
        if (pathData.isEmissive || (rayData.terminated & kRayFlagTerminatedMiss > 0))
            break;

        uint index = pixelToPathIndex(launchIndex, launchDim);
        gCameraPaths[index + i] = float4(1.0);

        //Russian Roulette
        const float rrVal = luminance(pathData.thp);
        const float prob = max(0.f, 1.f - rrVal);
        float rnd = (rayData.terminated & kRayFlagTerminatedHit) ? -1.0 : sampleNext1D(pathData.sg);
        if (rnd < prob)
        {
            break; //Photon is absorbed
        }
        else
        {
            pathData.thp /= (1.f - prob);
        }
                        
        //update ray for next iteration
        ray.Origin = pathData.origin;
        ray.Direction = pathData.direction;
    }
}
