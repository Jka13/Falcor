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

#ifndef PATH_LENGTH
#define PATH_LENGTH 1
#endif

cbuffer PerFrame
{
    uint gFrameCount; // Frame count since scene was loaded.
    uint2 gFrameDim;
}

cbuffer CB
{
    const uint gMaxRecursion; //Max photon recursion depths
    const uint gFlags; //Flags for the pass. See GenerationFlags enum
};

struct LightPathVertex
{
    float3 flux;
    float3 direction;
    PackedHitInfo hit;
};

struct CameraPathVertex
{
    float3 direction;
    PackedHitInfo hit;
}

// inputs
RWStructuredBuffer<LightPathVertex> gLightPaths;
RWStructuredBuffer<CameraPathVertex> gCameraPaths;

// outputs
RWTexture2D<float4> gColor;

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
    float3 reprojection_brdf;
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
        d.reprojection_brdf = float3(0);
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

ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
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
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
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

bool isShadowed(float3 lightVertexPosW, float3 cameraVertexPosW)
{
    float3 direction = cameraVertexPosW - lightVertexPosW;
    RayDesc shadowRay;
    shadowRay.Origin = lightVertexPosW;
    shadowRay.TMin = 0;
    shadowRay.TMax = length(direction);
    shadowRay.Direction = normalize(direction);

    ShadowData shadowData;
    shadowData.shadowed = true;

    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, rayTypeCount, 1 /* missIdx */, shadowRay, shadowData);
    return shadowData.shadowed;
}

struct UnpackedVertex
{
    bool isConnectible;
    float3 posW;
    ShadingData sd;
    IMaterialInstance mi;
    BSDFProperties miProperties;
}

UnpackedVertex unpackHit(PackedHitInfo hit, float3 hitDirection)
{
    UnpackedVertex uv;
    // Evaluate Falcor's material parameters at the hit point.
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(hit);
    if (!hit.isValid())
    {
        uv.isConnectible= false;
        return uv;
    }

    ShadingData sd = loadShadingData(hit, hitDirection, lod);

    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;

    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    if (bsdfProperties.roughness < 0.2)
    {
        uv.isConnectible = false;
        return uv;
    }
    uv.isConnectible = true;
    uv.posW = sd.posW;
    uv.mi = bsdf;
    uv.miProperties = bsdfProperties;
    uv.sd = sd;
    return uv;
}

float3 evaluateBRDF(ShadingData sd, SampleGenerator sg, float3 w_O)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    let miProperties = mi.getProperties(sd);
    float NdotV = saturate(dot(miProperties.guideNormal, sd.V));
    float NdotL = saturate(dot(miProperties.guideNormal, -w_O));
    float3 brdf = (mi.eval(sd, -w_O, sg) / NdotL) * NdotV;
    float3 reprojection_pdf = mi.evalPdf(sd, -w_O);
    float3 reprojection_brdf = max(0, brdf); // Valid due to Helmholtz reciprocity
    return reprojection_brdf;
}

float3 connectVertices(uint lightPathIndex, uint cameraPathIndex, SampleGenerator sg)
{

    UnpackedVertex lightVertex = unpackHit(gLightPaths[lightPathIndex].hit, gLightPaths[lightPathIndex].direction);
    UnpackedVertex cameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, gCameraPaths[cameraPathIndex].direction);
    float3 luminance = gLightPaths[lightPathIndex].flux;
    if (any(luminance == float3(0)) || !lightVertex.isConnectible || !cameraVertex.isConnectible || isShadowed(lightVertex.posW, cameraVertex.posW))
    {
        return float3(0);
    }
    float3 connectionDir = normalize(cameraVertex.posW - lightVertex.posW);
    float3 lightBRDF = evaluateBRDF(lightVertex.sd, sg, connectionDir);
    float3 cameraBRDF = float3(0);
    luminance *= lightBRDF;
    for (; cameraPathIndex > 0; --cameraPathIndex)
    {
        UnpackedVertex prevCameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, gCameraPaths[cameraPathIndex].direction);
        connectionDir = normalize(prevCameraVertex.posW - cameraVertex.posW);
        cameraBRDF = evaluateBRDF(cameraVertex.sd, sg, connectionDir);
        luminance *= cameraBRDF;
        --cameraPathIndex;
        cameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, gCameraPaths[cameraPathIndex].direction);
    }
    float3 cameraPos = gScene.camera.data.posW;
    connectionDir = normalize(cameraPos - cameraVertex.posW);
    cameraBRDF = evaluateBRDF(cameraVertex.sd, sg, connectionDir);
    return luminance * cameraBRDF;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    uint index = pixelToPathIndex(launchIndex, launchDim);

    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount + 5 * 13); // Sample Generator for BRDF evaluation

    float3 luminances[PATH_LENGTH * 2 + 1];
    // max recursion = max path length
    for (uint cameraPathOffset = 0; cameraPathOffset < gMaxRecursion; cameraPathOffset++)
    {
        uint pathLength = cameraPathOffset + 1; 
        for (uint lightPathOffset = 0; lightPathOffset < gMaxRecursion; lightPathOffset++)
        {
            pathLength += lightPathOffset + 1;
            uint lightPathIndex = index + lightPathOffset;
            uint cameraPathIndex = index + cameraPathOffset;
            luminances[pathLength + 1] += connectVertices(lightPathIndex, cameraPathIndex, sg);
        }
    }
    for (uint i = 0; i < PATH_LENGTH * 2 + 1; ++i)
    {
        luminances[i] /= i + 1;
        gColor[launchIndex] += float4(luminances[i], 1.0);
    }
}
