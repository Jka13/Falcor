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
    const uint gMaxRecursion; // Max photon recursion depths
    const uint gSelectedLightPathVertex;
    const uint gSelectedCameraPathVertex;
    const uint gFlags; //Flags for the pass. See GenerationFlags enum
};

struct LightPathVertex
{
    float3 flux;
    float forwardPdf;
    float reversePdf;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;

    __init()
    {
        flux = float3(0);
        forwardPdf = 0.0;
        reversePdf = 0.0;
        posW = float3(0);
        direction = float3(0);
        isConnectible = 0;
        hit = PackedHitInfo(uint4(0));
    }
};

struct CameraPathVertex
{
    float3 luminance;
    float forwardPdf;
    float reversePdf;
    float3 thp;
    float3 posW;
    float3 direction;
    uint isConnectible;
    PackedHitInfo hit;

    __init()
    {
        luminance = float3(0);
        forwardPdf = 0.0;
        reversePdf = 0.0;
        thp = float3(0);
        posW = float3(0);
        direction = float3(0);
        isConnectible = 0;
        hit = PackedHitInfo(uint4(0));
    }
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

struct ConnectionData
{
    float3 luminance;
    float3 connectionDirection;
    float connectionForwardPdf;
    float connectionBackwardPdf;

    __init()
    {
        luminance = float3(0);
        connectionDirection = float3(0);
        connectionForwardPdf = 0.0;
        connectionBackwardPdf = 0.0;
    }
};

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

float3 evaluateBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O, out float pdf)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    pdf = mi.evalPdf(sd, w_O);
    return max(mi.eval(sd, w_O, sg), 0);
}

float3 evaluateBRDF(in ShadingData sd, inout SampleGenerator sg, float3 w_O)
{
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    return max(mi.eval(sd, w_O, sg), 0);
}

#if MODE == 0
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
        return float3(-1);
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
        return float3(-1);
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
#elif MODE == 1
ConnectionData connectVertices(uint lightPathIndex, uint cameraPathIndex, inout SampleGenerator sg)
{
    ConnectionData cd;
    LightPathVertex lpv = gLightPaths[lightPathIndex];
    float3 luminance = lpv.flux;
    if (all(luminance == float3(0)) || lpv.isConnectible == 0)
    {
        return cd;
    }
    CameraPathVertex cpv = gCameraPaths[cameraPathIndex];
    if (cpv.isConnectible == 0)
    {
        return cd;
    }
    ShadingData lightVertex; 
    if (!unpackHit(lpv.hit, lpv.direction, lightVertex))
    {
        return cd;
    }
    float3 connectionDir = normalize(lpv.posW - cpv.posW);
    float cosTheta = -dot(connectionDir, lightVertex.getOrientedFaceNormal());
    if (cosTheta < 0) 
    {
        return cd;
    }
    ShadingData cameraVertex;
    if (!unpackHit(cpv.hit, cpv.direction, cameraVertex))
    {
        return cd;
    }
    if (isShadowed(lpv.posW, cpv.posW))
    {
        return cd;
    }
    float3 lightBRDF = evaluateBRDF(lightVertex, sg, -connectionDir, cd.connectionBackwardPdf);
    float3 distance = length(cpv.posW - lpv.posW);
    float3 cameraBRDF = evaluateBRDF(cameraVertex, sg, connectionDir, cd.connectionForwardPdf);
    cd.luminance = luminance * cameraBRDF * cosTheta * cpv.thp / (distance * distance);
    cd.connectionDirection = connectionDir;
    return cd;
}
#endif

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
        gPathData[dataIndex + pathDepth].count += gPathData[dataIndex + pathDepth].count == 0 ? 1 : 0;
    }
}

float3 showReprojectionPathLengthX(uint pathLength)
{
    float3 luminance = float3(0);
    uint2 headTextureIndex = DispatchRaysIndex().xy; 
    uint linkedListIndex = gHeadCounter[headTextureIndex];
    while (linkedListIndex != uint(-1)) {
        ListNode currentNode = gLinkedList[linkedListIndex];
        linkedListIndex = currentNode.next;
        uint pathDepth = currentNode.pathDepth;
        luminance += pathDepth == pathLength ? currentNode.luminance : float3(0); 
    }
    return luminance;
}

float3 showCameraPathIlluminationLengthX(uint cameraPathIndex)
{
    CameraPathVertex cpv = gCameraPaths[cameraPathIndex];
    return cpv.luminance * cpv.thp;
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
    if (cosTheta < 0)
    {
        return float3(0);
    }
    ShadingData cameraVertex;
    if (!unpackHit(cpv.hit, cpv.direction, cameraVertex))
    {
        return float3(-1);
    }
    if (isShadowed(lpv.posW, cpv.posW))
    {
        return float3(0);
    }
    float3 distance = length(cpv.posW - lpv.posW);
    float3 cameraBRDF = evaluateBRDF(cameraVertex, sg, connectionDir);
    return luminance * cameraBRDF * cosTheta * cpv.thp / (distance * distance);
}

float updateLightPathReversePdf(uint lightPath, uint connectionIndex, float3 connectionDir)
{
    float reversePdf = 1.0;
    LightPathVertex lpv = gLightPaths[lightPath + connectionIndex]; 
    ShadingData sd;
    if (!unpackHit(lpv.hit, connectionDir, sd));
    {
        return reversePdf;
    }
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    reversePdf = mi.evalPdf(sd, -lpv.direction); 
    return  reversePdf > 0 ? reversePdf : 1.0;
}

float updateCameraPathReversePdf(uint cameraPath, uint connectionIndex, float3 connectionDir)
{
    float reversePdf = 1.0;
    CameraPathVertex cpv = gCameraPaths[cameraPath + connectionIndex]; 
    ShadingData sd;
    if (!unpackHit(cpv.hit, -connectionDir, sd));
    {
        return reversePdf;
    }
    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
    reversePdf = mi.evalPdf(sd, -cpv.direction); 
    return  reversePdf > 0 ? reversePdf : 1.0;
}

float evaluateLightPathConnectionStrategies(uint lightPath, int connectionIndex, float updatedReversePdf)
{
    float inverseMISWeight = 0.0;
    if (connectionIndex == 0)
        return inverseMISWeight;
    float r_i = 1.0;
    //unroll first loop iteration to use the updated reverse pdf
    LightPathVertex currentLpv = gLightPaths[lightPath + connectionIndex - 1];
    LightPathVertex connectionLpv = gLightPaths[lightPath + connectionIndex];
    r_i *= updatedReversePdf / currentLpv.forwardPdf;
    if (currentLpv.isConnectible == 1 && connectionLpv.isConnectible == 1)
        inverseMISWeight += r_i;
    for (int lightPathIndex = connectionIndex - 1; lightPathIndex > 0; --lightPathIndex)
    {
        connectionLpv = currentLpv;
        currentLpv = gLightPaths[lightPath + lightPathIndex - 1];
        r_i *= currentLpv.reversePdf / currentLpv.forwardPdf;
        if (currentLpv.isConnectible == 1 && connectionLpv.isConnectible == 1)
            inverseMISWeight += r_i;
    }
    return inverseMISWeight;
}

float evaluateCameraPathConnectionStrategies(uint cameraPath, int connectionIndex, float connectionForwardPdf, float connectionReversePdf, float updatedReversePdf)
{
    float inverseMISWeight = 0.0;
    if (connectionIndex == 0)
        return inverseMISWeight;
    float r_i = 1.0;
    //unroll first two iterations for the original connection and the updated reverse pdf
    r_i *= connectionReversePdf / connectionReversePdf;
    // connectible between the actual connection vertices has been satisfied already
    inverseMISWeight += r_i;
    CameraPathVertex connectionCpv = gCameraPaths[cameraPath + connectionIndex];
    CameraPathVertex currentCpv = gCameraPaths[cameraPath + connectionIndex - 1];
    r_i *= updatedReversePdf / currentCpv.forwardPdf;
    if (currentCpv.isConnectible == 1 && connectionCpv.isConnectible == 1)
        inverseMISWeight += r_i;
    for (int cameraPathIndex = connectionIndex - 1; cameraPathIndex > 0; --cameraPathIndex)
    {
        connectionCpv = currentCpv;
        currentCpv = gCameraPaths[cameraPath + cameraPathIndex - 1];
        r_i *= currentCpv.reversePdf / currentCpv.forwardPdf;
        if (currentCpv.isConnectible == 1 && connectionCpv.isConnectible == 1)
            inverseMISWeight += r_i;
    }
    return inverseMISWeight;
}

float getMISWeight(ConnectionData cd, uint cameraPath, uint cameraPathConnectionIndex, uint lightPath, uint lightPathConnectionIndex)
{
    float updatedCameraReversePdf = updateCameraPathReversePdf(cameraPath, cameraPathConnectionIndex, cd.connectionDirection);
    float updatedLightReversePdf = updateLightPathReversePdf(lightPath, lightPathConnectionIndex, cd.connectionDirection);
    float cameraPathsInverseWeight = evaluateCameraPathConnectionStrategies(cameraPath, cameraPathConnectionIndex, cd.connectionForwardPdf, cd.connectionBackwardPdf, updatedCameraReversePdf);
    float lightPathsInverseWeight = evaluateLightPathConnectionStrategies(lightPath, lightPathConnectionIndex, updatedLightReversePdf);
    return 1 / (1 + cameraPathsInverseWeight + lightPathsInverseWeight);
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    uint dataIndex = pixelToPathDataIndex(launchIndex, launchDim);
    uint cameraIndex = pixelToCamerapathIndex(launchIndex, launchDim);
    uint lightIndex = pixelToLightpathIndex(launchIndex, launchDim);

    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount + 11 * 19); // Sample Generator for BRDF evaluation

    float3 luminance = float3(0);
    float misWeight = 0.0;
    ConnectionData cd;
    #if PATH_SELECTION_ENABLED == 0
    #if MODE == 0
    addReprojectionLuminance(dataIndex);
    #endif
    for (uint cameraPathOffset = 0; cameraPathOffset < PATH_LENGTH; cameraPathOffset++)
    {
        uint pathLength = cameraPathOffset;
        uint cameraPathIndex = cameraIndex + cameraPathOffset;

        #if MODE == 0 //Average
        float3 luminance = gCameraPaths[cameraPathIndex].luminance * gCameraPaths[cameraPathIndex].thp;
        gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);
        gPathData[dataIndex + pathLength].count += 1;
        luminance = connectCameraVertexToLightSource(cameraPathIndex, lightIndex, sg);
        gPathData[dataIndex + pathLength].count += any(luminance) < 0 ? 0 : 1;
        luminance = any(luminance) < 0 ? float3(0) : luminance;
        gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);

        #elif MODE == 1 //MIS
        //luminance += misWeight * gCameraPaths[cameraPathIndex].luminance * gCameraPaths[cameraPathIndex].thp;
        //misWeight = getMisWeight();
        //luminance += misWeight * connectCameraVertexToLightSource(cameraPathIndex, lightIndex, sg);    
        #endif
        for (uint lightPathOffset = 1; lightPathOffset < PATH_LENGTH + 1; lightPathOffset++)
        {
            pathLength += lightPathOffset;
            uint lightPathIndex = lightIndex + lightPathOffset;

            #if MODE == 0 //Average
            luminance = connectVertices(lightPathIndex, cameraPathIndex, sg);
            gPathData[dataIndex + pathLength].count += any(luminance) < 0 ? 0 : 1;
            luminance = any(luminance) < 0 ? float3(0) : luminance;
            gPathData[dataIndex + pathLength].luminance += float4(luminance, 0.0);
            pathLength += lightPathOffset;

            #elif MODE == 1 //MIS
            cd = connectVertices(lightPathIndex, cameraPathIndex, sg);
            misWeight = getMISWeight(cd, cameraIndex, cameraPathOffset, lightIndex, lightPathOffset);
            luminance += misWeight * cd.luminance;
            #endif
        }
    }
#else
    uint lightPathIndex = lightIndex + gSelectedLightPathVertex;
    uint cameraPathIndex = cameraIndex + gSelectedCameraPathVertex;
    uint pathLength = gSelectedCameraPathVertex + gSelectedLightPathVertex;
    #if MODE == 0
    if (gSelectedLightPathVertex == -1 && gSelectedCameraPathVertex == -1)
        luminance = float3(0);
    else if (gSelectedLightPathVertex == -1)
        luminance = showCameraPathIlluminationLengthX(cameraPathIndex);
    else if (gSelectedCameraPathVertex == -1)
        luminance = showReprojectionPathLengthX(gSelectedLightPathVertex);
    else if (gSelectedLightPathVertex == 0)
        luminance = connectCameraVertexToLightSource(cameraPathIndex, lightPathIndex, sg);
    else
        luminance = connectVertices(lightPathIndex, cameraPathIndex, sg);
    #elif MODE == 1 //MIS
    if (gSelectedLightPathVertex == -1 && gSelectedCameraPathVertex == -1)
        luminance = float3(0);
    else if (gSelectedLightPathVertex == -1)
        luminance = showCameraPathIlluminationLengthX(cameraPathIndex);
    else if (gSelectedCameraPathVertex == -1)
        luminance = showReprojectionPathLengthX(gSelectedLightPathVertex);
    else if (gSelectedLightPathVertex == 0)
        luminance = connectCameraVertexToLightSource(cameraPathIndex, lightPathIndex, sg);
    else
        cd = connectVertices(lightPathIndex, cameraPathIndex, sg);
        misWeight = getMISWeight(cd, cameraIndex, gSelectedCameraPathVertex, lightIndex, gSelectedLightPathVertex);
        luminance += misWeight * cd.luminance;
    #endif
    #endif
    #if MODE == 1 || PATH_SELECTION_ENABLED == 1
    gColor[launchIndex] = float4(luminance, 1.0);
    #endif
}
