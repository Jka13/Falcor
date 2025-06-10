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
    float forwardPdf;
    float backwardPdf;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;
};

struct CameraPathVertex
{
    float3 luminance;
    float forwardPdf;
    float backwardPdf;
    float3 thp;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;
};

struct PathData
{
    float4 luminance;
    uint count;
}

struct ListNode {
    float3 luminance;
    uint pathDepth;
    int next;

    __init() {
        luminance = float3(0);
        pathDepth = 0;
        next = -1;
    }
}

// inputs
RWStructuredBuffer<LightPathVertex> gLightPaths;
RWStructuredBuffer<CameraPathVertex> gCameraPaths;
RWStructuredBuffer<PathData> gPathData;
RWStructuredBuffer<ListNode> gLinkedList;
RWTexture2D<uint> gHeadCounter;

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

uint pixelToLightpathIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (gMaxRecursion + 1);
}

uint pixelToCamerapathIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * gMaxRecursion;
}

uint pixelToPathDataIndex(uint2 pixel, uint2 dimensions)
{
    return (pixel.y * dimensions.x + pixel.x) * (2 * gMaxRecursion);
}

bool isShadowed(float3 lightVertexPosW, float3 cameraVertexPosW)
{
    float3 direction = lightVertexPosW - cameraVertexPosW;
    RayDesc shadowRay;
    shadowRay.Origin = cameraVertexPosW;
    shadowRay.Direction = normalize(direction);
    shadowRay.TMin = 0;
    shadowRay.TMax = length(direction) * 0.99;

    ShadowData shadowData;
    shadowData.shadowed = true;

    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff /* instanceInclusionMask */, 1 /* hitIdx */, rayTypeCount, 1 /* missIdx */, shadowRay, shadowData);
    return shadowData.shadowed;
}

bool unpackHit(PackedHitInfo packedHit, float3 hitDirection, out ShadingData sd)
{
    // Evaluate Falcor's material parameters at the hit point.
    sd = {};
    let lod = ExplicitLodTextureSampler(0.f);
    HitInfo hit = HitInfo(packedHit);
    if (!hit.isValid())
    {
        return false;
    }
    sd = loadShadingData(hit, hitDirection, lod);
    return true;
}

float3 evaluateReprojectionBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    let miProperties = mi.getProperties(sd);
    float NdotV = saturate(dot(miProperties.guideNormal, sd.V));
    float NdotL = saturate(dot(miProperties.guideNormal, w_O));
    float3 brdf = (mi.eval(sd, w_O, sg) / NdotL) * NdotV;
    float3 pdf = mi.evalPdf(sd, w_O);
    return max(0, brdf);
}

float3 evaluateBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    return max(mi.eval(sd, w_O, sg), 0);
}

float3 connectVertices(uint lightPathIndex, uint cameraPathIndex, inout SampleGenerator sg)
{
    LightPathVertex lpv = gLightPaths[lightPathIndex];
    float3 luminance = lpv.flux;
    if (all(luminance == float3(0)) || lpv.isConnectible == 0)
    {
        return float3(0);
    }
    CameraPathVertex cpv = gCameraPaths[cameraPathIndex];
    if (cpv.isConnectible == 0)
    {
        return float3(0);
    }
    ShadingData lightVertex; 
    if (!unpackHit(lpv.hit, lpv.direction, lightVertex))
    {
        return float3(0);
    }
    float3 connectionDir = normalize(lpv.posW - cpv.posW);
    float cosTheta = -dot(connectionDir, lightVertex.getOrientedFaceNormal());
    if (cosTheta < 0) 
    {
        return float3(0);
    }
    ShadingData cameraVertex;
    if (!unpackHit(cpv.hit, cpv.direction, cameraVertex))
    {
        return float3(0);
    }
    if (isShadowed(lpv.posW, cpv.posW))
    {
        return float3(0);
    }
    //float3 lightBRDF = evaluateBRDF(lightVertex, sg, connectionDir);
    float3 distance = length(cpv.posW - lpv.posW);
    float3 cameraBRDF = evaluateBRDF(cameraVertex, sg, connectionDir);
    return luminance * cameraBRDF * cosTheta * cpv.thp / (distance * distance);
}

void addReprojectionLuminance(uint dataIndex)
{
    uint2 headTextureIndex = DispatchRaysIndex().xy; 
    uint linkedListIndex = gHeadCounter[headTextureIndex];
    while (linkedListIndex != uint(-1)) {
        ListNode currentNode = gLinkedList[linkedListIndex];
        linkedListIndex = currentNode.next;
        uint pathDepth = currentNode.pathDepth;
        float3 luminance = currentNode.luminance; 
        gPathData[dataIndex + pathDepth].luminance += float4(luminance, 0.0);
        gPathData[dataIndex + pathDepth].count += any(luminance) > 0 ? gPathData[dataIndex + pathDepth].count == 0 ? 1 : 0 : 0;
    }
}

float3 connectCameraVertexToLightSource(uint cameraPathIndex, uint lightPathIndex, SampleGenerator sg)
{
    LightPathVertex lpv = gLightPaths[lightPathIndex];
    float3 luminance = lpv.flux;
    if (all(luminance == float3(0)) || lpv.isConnectible == 0)
    {
        return float3(0);
    }
    CameraPathVertex cpv = gCameraPaths[cameraPathIndex];
    if (cpv.isConnectible == 0)
    {
        return float3(0);
    }
    float3 connectionDir = normalize(lpv.posW - cpv.posW);
    float cosTheta = -dot(connectionDir, lpv.direction);
    if (cosTheta < 0) //direction for the first light path entry stores the normal of the light source
    {
        return float3(0);
    }
    ShadingData cameraVertex;
    if (!unpackHit(cpv.hit, cpv.direction, cameraVertex))
    {
        return float3(0);
    }
    if (isShadowed(lpv.posW, cpv.posW))
    {
        return float3(0);
    }
    float3 distance = length(cpv.posW - lpv.posW);
    float3 cameraBRDF = evaluateBRDF(cameraVertex, sg, connectionDir);
    return luminance * cameraBRDF * cosTheta * cpv.thp / (distance * distance);
}

float evaluateLightPathConnectionStrategies(uint lightPath, int lightPathConnectionIndex)
{
    return 0.0;
    float inverseMISWeight = 0.0;
    LightPathVertex nextLpv =  gLightPaths[lightPath + lightPathConnectionIndex > 0 ? lightPathConnectionIndex - 1 : 0];
    LightPathVertex currentLpv;
    float r_i = 1.0;
    for (int lightPathIndex = lightPathConnectionIndex - 1; lightPathIndex > 0; --lightPathIndex)
    {
        currentLpv = nextLpv;
        nextLpv = gLightPaths[lightPath + lightPathIndex - 1];
        r_i *= currentLpv.forwardPdf / currentLpv.backwardPdf;
        if (currentLpv.isConnectible == 1 && nextLpv.isConnectible == 1)
            inverseMISWeight += r_i;
    }
    r_i *= nextLpv.forwardPdf / nextLpv.backwardPdf;
    return inverseMISWeight + r_i;
}

float evaluateCameraPathConnectionStrategies(uint cameraPath, int cameraPathConnectionIndex)
{
    return 0.0;
    float inverseMISWeight = 0.0;
    CameraPathVertex nextCpv =  gCameraPaths[cameraPath + cameraPathConnectionIndex > 0 ? cameraPathConnectionIndex - 1 : 0];
    CameraPathVertex currentCpv;
    float r_i = 1.0;
    for (int cameraPathIndex = cameraPathConnectionIndex - 1; cameraPathIndex > 0; --cameraPathIndex)
    {
        currentCpv = nextCpv;
        nextCpv = gCameraPaths[cameraPath + cameraPathIndex - 1];
        r_i *= currentCpv.forwardPdf / currentCpv.backwardPdf;
        if (currentCpv.isConnectible == 1 && nextCpv.isConnectible == 1)
            inverseMISWeight += r_i;
    }
    if (nextCpv.isConnectible == 1)
        return inverseMISWeight;
    r_i *= nextCpv.forwardPdf / nextCpv.backwardPdf;
    return inverseMISWeight + r_i;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    uint cameraIndex = pixelToCamerapathIndex(launchIndex, launchDim);
    uint lightIndex = pixelToLightpathIndex(launchIndex, launchDim);

    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount + 11 * 19); // Sample Generator for BRDF evaluation

    #if 0
    uint dataIndex = pixelToPathDataIndex(launchIndex, launchDim);
    //addReprojectionLuminance(dataIndex);
    for (uint cameraPathOffset = 0; cameraPathOffset < PATH_LENGTH; cameraPathOffset++)
    {
        uint pathLength = cameraPathOffset;
        uint cameraPathIndex = cameraIndex + cameraPathOffset;
        // float3 luminance = float3(0);
        float3 luminance = misWeight * gCameraPaths[cameraPathIndex].luminance * gCameraPaths[cameraPathIndex].thp;
        gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);
        gPathData[dataIndex + pathLength].count += any(luminance) > 0 ? 1 : 0;
        luminance = connectCameraVertexToLightSource(cameraPathIndex, lightIndex, sg);    
        gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);
        gPathData[dataIndex + pathLength].count += any(luminance) > 0 ? 1 : 0;
        for (uint lightPathOffset = 1; lightPathOffset < PATH_LENGTH + 1; lightPathOffset++)
        {
            pathLength += lightPathOffset;
            uint lightPathIndex = lightIndex + lightPathOffset;
            luminance = connectVertices(lightPathIndex, cameraPathIndex, sg);
            gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);
            gPathData[dataIndex + pathLength].count += any(luminance) > 0 ? 1 : 0;
        }
    }
    #endif

#if 1
    float3 luminance = float3(0);
    for (uint cameraPathOffset = 0; cameraPathOffset < PATH_LENGTH; cameraPathOffset++)
    {
        uint pathLength = cameraPathOffset;
        uint cameraPathIndex = cameraIndex + cameraPathOffset;
        float misWeight = 1 / (1 + evaluateCameraPathConnectionStrategies(cameraIndex, cameraPathOffset));
        luminance += misWeight * gCameraPaths[cameraPathIndex].luminance * gCameraPaths[cameraPathIndex].thp;
        misWeight = 1 / (1 + evaluateCameraPathConnectionStrategies(cameraIndex, cameraPathOffset)
                           + evaluateLightPathConnectionStrategies(lightIndex, 0));
        luminance += misWeight * connectCameraVertexToLightSource(cameraPathIndex, lightIndex, sg);    
        for (uint lightPathOffset = 1; lightPathOffset < PATH_LENGTH + 1; lightPathOffset++)
        {
            pathLength += lightPathOffset;
            uint lightPathIndex = lightIndex + lightPathOffset;
            misWeight = 1 / (1 + evaluateCameraPathConnectionStrategies(cameraIndex, cameraPathOffset)
                               + evaluateLightPathConnectionStrategies(lightIndex, lightPathOffset));
            luminance += misWeight * connectVertices(lightPathIndex, cameraPathIndex, sg);
        }
    }
    gColor[launchIndex] = float4(luminance, 0.0);
#endif
}
