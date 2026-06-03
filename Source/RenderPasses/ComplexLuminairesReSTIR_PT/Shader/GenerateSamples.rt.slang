#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.Microfacet;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import Scene.RaytracingInline;
import Scene.Raytracing;

import Helpers.SceneDataHandler;
import Helpers.VPLSampler;

/** Payload for scatter ray (80B).
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

cbuffer PerFrame
{
    uint gFrameCount;
    uint2 gFrameDim;
}

cbuffer CB
{
    uint gMaxRecursion;
    uint gFlags;
    float gAABBSize;
    float gNormalizedPixelArea;
    float2 gPixelWidthHeight;
    float gEmissiveLightProb;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

RWStructuredBuffer<vplSampler.DirectVPL> gDirectVPLBuffer;
RWStructuredBuffer<vplSampler.IndirectVPL> gIndirectVPLBuffer;
RWStructuredBuffer<AABB> gPhotonAABBs;
RWStructuredBuffer<ListNode> gLinkedList;
RWTexture2D<uint> gHeadCounter;

RWBuffer<uint> gPhotonCounter;
RWBuffer<uint> gLinkedListCounter;
RWTexture2D<float4> gOutDebug;

struct LightSample
{
    float3 posW;
    float3 normal;
    float3 dirW;
    float3 flux;
    float selectionPDF;

    __init()
    {
        posW = float3(0);
        normal = float3(0);
        dirW = float3(0);
        flux = float3(0);
        selectionPDF = 0.0;
    }
};

struct ListNode
{
    float3  luminance;
    float   pathPDF;
    uint    pathLength;
    int     next;

    __init()
    {
        luminance = float3(0);
        pathPDF = 0.0;
        pathLength = 0;
        next = -1;
    }
};

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

static const uint kPhotonBufferSize = uint(PHOTON_BUFFER_SIZE_GLOBAL);
static const bool kUseDirectPointLights = USE_DIRECT_POINT_LIGHTS;
static const bool kUseIndirectPointLights = USE_INDIRECT_POINT_LIGHTS;

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
    rayData.materialID = gScene.getMaterialID(triangleHit.instanceID);
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    HitInfo hitInfo = {};
    rayData.hit = hitInfo.getData();
    rayData.materialID = (uint) -1;
}

//transforms direction from local to world space
float3 fromLocalToWorld(const in float3 normal, float3 w)
{
    //transform from local space -> random dir vector in world space
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return w.x * tangent + w.y * bitangent + w.z * normal;
}

bool generateLightSample(bool isEmissive, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

    float3 woLocal = float3(0);

    if (isEmissive)
    {
        const uint lightCount = gScene.lightCollection.triangleCount;
        const uint selectedLight = min(uint(floor(lightCount * sampleNext1D(sg))), lightCount - 1);
        // Get emissive triangle data
        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(selectedLight);
        const float3 barycentrics = sample_triangle(sampleNext2D(sg)); // Sample random triangle position
        ls.posW = tri.getPosition(barycentrics);
        ls.normal = tri.normal;

        // Sample emissive material of the triangle
        float2 uv = tri.getTexCoord(barycentrics);
        ls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

        ls.posW = computeRayOrigin(ls.posW, ls.normal); // Offset as this light is real geometry

        // Sample random direction in a cosine lobe
        float lightDirPDF = 1.0;
        woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);
        if (lightDirPDF <= 0.) // Invalid direction
            return false;
        ls.flux *= M_PI;               // Lambert Emitter (or reduced cos/pdf)
        ls.flux *= lightCount;         // Inv light select pdf
        //ls.flux /= gEmissiveLightProb; //Probability to select an emissive light
        ls.selectionPDF = lightDirPDF / lightCount;
    } else
    {
        const uint lightCount = gScene.getLightCount();
        const uint selectedLight = min(uint(floor(lightCount * sampleNext1D(sg))), lightCount - 1);
        LightData lightData = gScene.getLight(selectedLight);
        // Only Point (and Spot) lights are supported
        if (lightData.type != uint(LightType::Point))
            return false;

        float spotAngle = lightData.cosOpeningAngle;
        float2 rndDir = sampleNext2D(sg);   //Random direction variable
        // Depending on the opening angle, the light is a spot or point light
        if (lightData.openingAngle < M_2PI) // Spot Light
        {
            woLocal = sample_cone(rndDir, spotAngle);
        } else // Point Light
        {
            spotAngle = 0; // Just to be sure
            woLocal = sample_sphere(rndDir);
        }
        float invLightPdf = M_2PI * (1.0 - spotAngle);
        ls.normal = lightData.dirW;
        ls.posW = lightData.posW;
        ls.flux = lightData.intensity * invLightPdf;
        ls.flux *= lightCount;               //Inv light select pdf
        //ls.flux /= 1.0 - gEmissiveLightProb; //Probability to select an analytic light
    }

    // Local to world direction
    //  Convert direction local -> world
    float3 bitangent = perp_stark(ls.normal);
    float3 tangent = cross(bitangent, ls.normal);
    ls.dirW = woLocal.x * tangent + woLocal.y * bitangent + woLocal.z * ls.normal;
    return true;
}

void backprojectPhoton(const ShadingData sd, const vplSampler.Photon photon, const float pathPDF, const uint pathLength, inout SampleGenerator sg)
{
    // Check if position is in camera view
    float4 screenPos = mul(gScene.camera.data.viewProjMatNoJitter, float4(sd.posW, 1));
    screenPos /= screenPos.w;
    screenPos.y *= -1;
    screenPos.xy = screenPos.xy * 0.5 + 0.5; // UV from [-1,1] to [0,1]
    if (any(screenPos.xy < 0) || any(screenPos.xy >= 1) || screenPos.z < 0 || screenPos.z > 1)
        return;

    float3 directionToCamera = gScene.camera.getPosition() - sd.posW;
    float distanceToCamera = length(directionToCamera);
    directionToCamera /= distanceToCamera;
    Ray shadowRay = Ray(photon.posW, directionToCamera, 0, distanceToCamera);
    const bool reprojectionVisible = sdh.dispatchShadowRay(shadowRay);
    if(!reprojectionVisible)
        return;

    uint2 currentPixel = uint2(floor(screenPos.xy * gFrameDim));
    //Store in Linked List
    uint photonIndex = 0;
    InterlockedAdd(gLinkedListCounter[0], 1u, photonIndex);
    ListNode node = ListNode();

    let lod = ExplicitLodTextureSampler(0.f);
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;

    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);

    const float3 camForward = normalize(gScene.camera.data.cameraW);
    const float cosAtCamera = abs(dot(-directionToCamera, camForward));
    float cameraPDF = distanceToCamera * distanceToCamera * gNormalizedPixelArea * pow(cosAtCamera, 3);
    const float3 bsdf = max(0, mi.eval(sd, directionToCamera, sg));
    const float3 Lr = (photon.flux * bsdf) /  cameraPDF;
    node.luminance = Lr / vplSampler.gPhotonCount;
    node.pathPDF = pathPDF * cameraPDF;
    node.pathLength = pathLength;
    InterlockedExchange(gHeadCounter[currentPixel], photonIndex, /*out*/ node.next); 
    gLinkedList[photonIndex] = node; 
}

//Load in shading data from hit and direction
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, rayDir, lod);
    
    return sd;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dispatchDims = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Generate a light sample
    bool isEmissive = (dispatchDims.x * gEmissiveLightProb) > pixel.x; // Use x dims for selection
    isEmissive = true;
    LightSample pls;
    if (!generateLightSample(isEmissive, sg, pls))
        return;

    vplSampler.Photon photon;
    photon.flux = pls.flux;
    photon.posW = pls.posW;
    photon.origin = pls.posW;
    photon.w_i = pls.dirW;
    photon.w_o = pls.dirW;
    photon.faceN = pls.normal;
    photon.originFaceN = pls.normal;

    RayData rayData = RayData();
    RayDesc ray = {};
    ray.Origin = pls.posW;
    ray.Direction = pls.dirW;
    ray.TMin = 0;
    ray.TMax = FLT_MAX;

    float3 thp = float3(1.0);
    float pathPDF = pls.selectionPDF;
    bool isLuminaireHit = true; 
    for (uint i = 0; isLuminaireHit && i < gMaxRecursion; ++i)
    {
        TraceRay(gScene.rtAccel, 0 /*Flags*/, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        photon.w_i = ray.Direction;
        photon.origin = ray.Origin;
        photon.originFaceN = photon.faceN;

        HitInfo hit = HitInfo(rayData.hit);
        if (!hit.isValid())
            return;

        let lod = ExplicitLodTextureSampler(0);
        ShadingData sd = loadShadingData(hit, -ray.Direction, lod);

        uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        //Hit with emissive materials are undefined
        if (any(bsdfProperties.emission > 0))
            return;

        BSDFSample bsdfSample;
        if(!mi.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/))
            return;

        photon.flux *= thp;
        thp = bsdfSample.weight;
        if (all(thp <= 0))
            return;

        const bool isDelta = bsdfSample.isLobe(LobeType::Delta);
        if(!isDelta)
            pathPDF *= bsdfSample.pdf;

        //Compute new ray origin to avoid self intersection
        const bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        const bool isCurveHit = hit.getType() == HitType::Curve;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            ray.Origin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
          ray.Origin = sd.computeNewRayOrigin(!transmission);
        }
        ray.Direction = bsdfSample.wo;

        photon.posW = ray.Origin;
        photon.w_o = ray.Direction;
        photon.faceN = sd.faceN;

        isLuminaireHit = gScene.materials.isCastShadow(rayData.materialID) == 0;

        #if USE_BACKPROJECTION
            backprojectPhoton(sd, photon, pathPDF, i, sg);
        #endif
    }

    if (!isLuminaireHit)
    {
        uint photonIndex = 0;
        InterlockedAdd(gPhotonCounter[0], 1u, photonIndex);
    #if USE_DIRECT_POINT_LIGHTS
        vplSampler.DirectVPL dVPL = vplSampler.DirectVPL(photon);
        float cosTheta = dot(photon.originFaceN, photon.w_i);
        float angleToSurface = sqrt(1 - cosTheta * cosTheta);
        dVPL.cosOpeningAngle = max(vplSampler.gCosOpeningAngle, angleToSurface);

        gDirectVPLBuffer[photonIndex] = dVPL;
        AABB directVPLAABB = AABB(float3(dVPL.posW - gAABBSize), float3(dVPL.posW + gAABBSize));
        gPhotonAABBs[photonIndex] = directVPLAABB;
    #endif
    #if USE_INDIRECT_POINT_LIGHTS
        vplSampler.IndirectVPL lp = vplSampler.IndirectVPL(photon);
        lp.hit = rayData.hit;
        gIndirectVPLBuffer[photonIndex] = lp;
        AABB indirectVPLAABB = AABB(float3(lp.origin - gAABBSize), float3(lp.origin + gAABBSize));
        gPhotonAABBs[photonIndex] = indirectVPLAABB;
    #endif
    }
}
