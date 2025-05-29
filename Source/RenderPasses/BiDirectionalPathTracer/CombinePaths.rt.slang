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
    float3 posW;
    float3 direction;
    PackedHitInfo hit;
};

struct CameraPathVertex
{
    float3 posW;
    float3 direction;
    PackedHitInfo hit;
}

// inputs
RWStructuredBuffer<LightPathVertex> gLightPaths;
RWStructuredBuffer<CameraPathVertex> gCameraPaths;
RWStructuredBuffer<float4> gPathData;

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

struct BrdfData
{
    float3 brdf;
    float3 pdf;
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

uint pixelToPathDataIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (2 * gMaxRecursion - 1);
}

bool isShadowed(float3 lightVertexPosW, float3 cameraVertexPosW)
{
    float3 direction = cameraVertexPosW - lightVertexPosW;
    RayDesc shadowRay;
    shadowRay.Origin = lightVertexPosW;
    shadowRay.Direction = normalize(direction);
    shadowRay.TMin = 0;
    shadowRay.TMax = length(direction);

    ShadowData shadowData;
    shadowData.shadowed = true;

    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, rayTypeCount, 1 /* missIdx */, shadowRay, shadowData);
    return shadowData.shadowed;
}

struct UnpackedVertex
{
    bool isConnectible;
    ShadingData sd;
}

UnpackedVertex unpackHit(PackedHitInfo packedHit, float3 hitDirection)
{
    UnpackedVertex uv;
    // Evaluate Falcor's material parameters at the hit point.
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(packedHit);
    if (!hit.isValid())
    {
        uv.isConnectible= false;
        return uv;
    }

    ShadingData sd = loadShadingData(hit, hitDirection, lod);

    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;

    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    if (bsdfProperties.roughness <= 0.08)
    {
        uv.isConnectible = false;
        return uv;
    }
    uv.isConnectible = true;
    uv.sd = sd;
    return uv;
}

float3 evaluateReprojectionBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    let miProperties = mi.getProperties(sd);
    float NdotV = saturate(dot(miProperties.guideNormal, sd.V));
    float NdotL = saturate(dot(miProperties.guideNormal, -w_O));
    float3 brdf = (mi.eval(sd, -w_O, sg) / NdotL) * NdotV;
    float3 pdf = mi.evalPdf(sd, -w_O);
    return max(0, brdf);
}

BrdfData evaluateBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O)
{
    BrdfData brdfData;
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    let miProperties = mi.getProperties(sd);
    brdfData.brdf = max(mi.eval(sd, -w_O, sg), 0);
    brdfData.pdf = mi.evalPdf(sd, -w_O);
    return brdfData;
}

float3 connectVertices(uint lightPathIndex, uint cameraPathIndex, uint startIndex, inout SampleGenerator sg)
{

    UnpackedVertex lightVertex = unpackHit(gLightPaths[lightPathIndex].hit, gLightPaths[lightPathIndex].direction);
    if (!lightVertex.isConnectible)
    {
        return float3(1, 0, 0);
    }
    UnpackedVertex cameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, gCameraPaths[cameraPathIndex].direction);
    if (!cameraVertex.isConnectible)
    {
        return float3(1, 0, 0);
    }
    //if (isShadowed(gLightPaths[lightPathIndex].posW, gLightPaths[lightPathIndex].posW))
    //{
    //    return float3(0, 0, 0);
    //}
    float3 luminance = gLightPaths[lightPathIndex].flux;
    if (all(luminance == float3(0)))
    {
        return float3(0, 1, 0);
    }
    float3 connectionDir = normalize(gLightPaths[lightPathIndex].posW - gLightPaths[lightPathIndex].posW);
    float3 lightBRDF = evaluateReprojectionBRDF(lightVertex.sd, sg, connectionDir);
    float3 cameraBRDF = float3(0);
    float3 distance = length(gLightPaths[lightPathIndex].posW - gLightPaths[lightPathIndex].posW);
    luminance *= lightBRDF / (distance * distance);
    for (; cameraPathIndex - startIndex > 0; --cameraPathIndex)
    {
        UnpackedVertex prevCameraVertex = unpackHit(gCameraPaths[cameraPathIndex - 1].hit, gCameraPaths[cameraPathIndex - 1].direction);
        //connectionDir = normalize(prevCameraVertex.posW - gLightPaths[lightPathIndex].posW);
        cameraBRDF = evaluateReprojectionBRDF(cameraVertex.sd, sg, connectionDir);
        luminance *= cameraBRDF;
        cameraVertex = prevCameraVertex;
    }
    cameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, gCameraPaths[cameraPathIndex].direction);
    float3 cameraPos = gScene.camera.getPosition();
    connectionDir = normalize(cameraPos - gLightPaths[lightPathIndex].posW);
    cameraBRDF = evaluateReprojectionBRDF(cameraVertex.sd, sg, connectionDir);
    return cameraBRDF;
}

float3 debugCameraBRDF(uint lightPathIndex, uint cameraPathIndex, uint startIndex, inout SampleGenerator sg)
{
    UnpackedVertex lightVertex = unpackHit(gLightPaths[lightPathIndex].hit, gLightPaths[lightPathIndex].direction);
    if (!lightVertex.isConnectible)
    {
        return float3(0, 0, 0);
    }
    float3 connectionDir = gCameraPaths[cameraPathIndex].posW - gLightPaths[lightPathIndex].posW;
    connectionDir = normalize(connectionDir);
    UnpackedVertex cameraVertex = unpackHit(gCameraPaths[cameraPathIndex].hit, connectionDir);
    if (!cameraVertex.isConnectible)
    {
        return float3(0, 0, 0);
    }
    if (isShadowed(gLightPaths[lightPathIndex].posW, gCameraPaths[cameraPathIndex].posW))
    {
        return float3(0, 0, 0);
    }
    float3 luminance = gLightPaths[lightPathIndex].flux;
    if (all(luminance == float3(0)))
    {
        return float3(0, 0, 0);
    }
    BrdfData lightBRDFData = evaluateBRDF(lightVertex.sd, sg, -connectionDir);
    float3 lightBRDF = lightBRDFData.brdf;
    float3 lightPDF = lightBRDFData.pdf;
    float3 distance = length(gCameraPaths[cameraPathIndex].posW - gLightPaths[lightPathIndex].posW);
    float3 cameraPos = gScene.camera.getPosition();
    connectionDir = normalize(gLightPaths[lightPathIndex].posW - gCameraPaths[cameraPathIndex].posW);
    BrdfData cameraBRDFData = evaluateBRDF(cameraVertex.sd, sg, gCameraPaths[cameraPathIndex].direction);
    float3 cameraBRDF = cameraBRDFData.brdf; 
    float cameraDistance = length(gCameraPaths[cameraPathIndex].posW - cameraPos);
    // return cameraBRDF;
    return luminance * lightBRDF / (distance * distance) * cameraBRDF;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    uint index = pixelToPathIndex(launchIndex, launchDim);
    uint dataIndex = pixelToPathDataIndex(launchIndex, launchDim);

    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount + 5 * 13); // Sample Generator for BRDF evaluation

    for (uint cameraPathOffset = 0; cameraPathOffset < PATH_LENGTH; cameraPathOffset++)
    {
        uint pathLength = cameraPathOffset; 
        for (uint lightPathOffset = 0; lightPathOffset < PATH_LENGTH; lightPathOffset++)
        {
            pathLength += lightPathOffset;
            uint lightPathIndex = index + lightPathOffset;
            uint cameraPathIndex = index + cameraPathOffset;
            gPathData[dataIndex + pathLength] += float4(debugCameraBRDF(lightPathIndex, cameraPathIndex, index, sg), 0.0);
        }
    }
}
