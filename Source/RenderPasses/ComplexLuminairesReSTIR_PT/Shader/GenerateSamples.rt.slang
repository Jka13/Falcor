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

struct PhotonLightSample
{
    float3 posW;
    float3 normal;
    float3 flux;
    float lightSampleSelectionPDF;

    __init()
    {
        posW = float3(0);
        normal = float3(0);
        flux = float3(0);
        lightSampleSelectionPDF = 0.0;
    }
};

struct ListNode {
    float3 luminance;
    int next;

    static ListNode create() {
        ListNode node;
        node.luminance = float3(0);
        node.next = -1;
        return node;
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
}

//transforms direction from local to world space
float3 fromLocalToWorld(const in float3 normal, float3 w)
{
    //transform from local space -> random dir vector in world space
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return w.x * tangent + w.y * bitangent + w.z * normal;
}

#if USE_EMISSIVE_LIGHT // Function is only valid if the emissive sampler is set
bool createLightSample(inout SampleGenerator sg, out PhotonLightSample pls)
{
    pls = {};
    if (gScene.lightCollection.isEmpty())
        return false;

    // Randomly pick a triangle out of the global list with uniform probability
    const uint triangleCount = gScene.lightCollection.triangleCount;
    // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
    uint triangleIndex = min((uint)(sampleNext1D(sg) * triangleCount), triangleCount - 1);

    uint2 packed = gEmissiveLightSampler._emissivePower.triangleAliasTable[triangleIndex];
    float threshold = f16tof32(packed.x >> 16u);
    uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    uint selectBelow = packed.y & 0xFFFFFFu;

    // Test the threshold in the current table entry; pick one of the two options
    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * gEmissiveLightSampler._emissivePower.invWeightsSum;

    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    const float3 barycentrics = sample_triangle(sampleNext2D(sg));
    pls.posW = tri.getPosition(barycentrics);
    pls.normal = tri.normal;

    // Get Flux
    float2 uv = tri.getTexCoord(barycentrics);
    pls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

    // Check if the material is double sided (infinetly thin)
    MaterialHeader mltHeader = gScene.materials.getMaterialHeader(tri.materialID);

    // If the material is double sided, randomly flip the normal
    /*
    if (mltHeader.isDoubleSided())
    {
        pls.flux *= 2;
        if (sampleNext1D(sg) >= 0.5f)
            pls.normal = -tri.normal;
    }
    */

    pls.posW = computeRayOrigin(pls.posW, pls.normal);
    pls.flux /= triangleSelectionPdf;
    pls.lightSampleSelectionPDF = triangleSelectionPdf;

    return true;
}
#endif

bool createLightSampleAnalytic(inout SampleGenerator sg, out PhotonLightSample pls, out float3 sampledDir)
{
    pls = {};
    sampledDir = float3(0);
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return false;
    // Uniformly sample one of the analytic lights
    float rndLightIdx = float(lightCount) * sampleNext1D(sg);
    uint lightIdx = min(uint(floor(rndLightIdx)), lightCount - 1);
    LightData lightData = gScene.getLight(lightIdx);

    // Only Point(+ Spot) lights are supported (due to RTXDI)
    if (lightData.type != uint(LightType::Point))
        return false;
    float invLightPdf;
    float3 woLocal;
    float3 rnd = sampleNext3D(sg);
    rnd.z = 0;
    float spotAngle = lightData.cosOpeningAngle;
    if (lightData.openingAngle < M_2PI) // If the opening angle is smaller than 2PI it is a spotlight
    {
        spotAngle = cos(lightData.openingAngle - lightData.penumbraAngle * rnd.z);
        woLocal = sample_cone(rnd.xy, spotAngle);
    }
    else // Pointlight
    {
        woLocal = sample_sphere(rnd.xy);
    }
    invLightPdf = M_2PI * (1.f - spotAngle);

    if (invLightPdf <= 0.f)
        return false;

    invLightPdf *= lightCount; // Take the select pdf into account

    // Fill the photon light struct
    pls.posW = lightData.posW;
    pls.normal = lightData.dirW;
    pls.flux = lightData.intensity * invLightPdf;

    sampledDir = fromLocalToWorld(pls.normal, woLocal);
    return true;
}

//float3 getReprojectionBRDF(const RayData rayData, float3 cameraDirection, float3 reprojectedDirection, const vplSampler.Photon photon, SampleGenerator sg) {
//    let lod = ExplicitLodTextureSampler(0.f);
//    HitInfo hit = HitInfo(rayData.hit);
//
//    //ShadingData sd = sdh.getShadingData(hit, cameraDirection, lod);  //Shading data for view from camera
//
//    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
//    let mi = gScene.materials.getMaterialInstance(sd, lod, hints); // Material instance is created for the camera hit
//    let miProperties = mi.getProperties(sd);
//    float NdotV = saturate(dot(miProperties.guideNormal, sd.V));
//    float NdotL = saturate(dot(miProperties.guideNormal, -reprojectedDirection));
//    float3 brdf = (mi.eval(sd, -reprojectedDirection, sg) / NdotL) * NdotV;
//    return max(0, brdf); // Valid due to Helmholtz reciprocity
//}

void backprojectPhoton(const ShadingData sd, const vplSampler.Photon photon, inout SampleGenerator sg)
{
    //Check if the point is visible for the camera
    float4 cameraPos = mul(gScene.camera.getViewProj(), float4(photon.posW, 1));
    cameraPos /= cameraPos.w;
    cameraPos.y *= -1;

    float3 cameraDir = mul(gScene.camera.data.viewMat, float4(photon.posW, 1)).xyz;
    cameraDir.y *= -1;
    cameraDir = normalize(cameraDir);

    float2 pixelDir = (floor(cameraPos.xy * 0.5 * gFrameDim) + 0.5) * gPixelWidthHeight;
    cameraPos.xy = cameraPos.xy * 0.5 + 0.5; //UV from [-1,1] to [0,1]
    bool validReprojection = true;
    if (any(cameraPos.xy < 0) || any(cameraPos.xy >= 1) || cameraPos.z < 0 || cameraPos.z > 1)
        validReprojection = false;

    bool reprojectionVisible = false;
    float3 pointToCamera = gScene.camera.getPosition() - photon.posW;
    float distToCamera = length(pointToCamera);
    pointToCamera /= distToCamera; //Normalize

    //Do a visibility test
    if(validReprojection){
       
        Ray shadowRay = Ray(photon.posW, pointToCamera, 0, distToCamera);
        reprojectionVisible = sdh.dispatchShadowRay(shadowRay);
    }
    //Store in Linked List
    if(validReprojection && reprojectionVisible){
        uint photonIndex = 0;
        InterlockedAdd(gLinkedListCounter[0], 1u, photonIndex);
        if (photonIndex < kPhotonBufferSize)
        {
            uint2 currentPixel = cameraPos.xy * gFrameDim;
            ListNode node = ListNode.create();
            let lod = ExplicitLodTextureSampler(0.f);
            uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            float3 brdf = max(mi.eval(sd, pointToCamera, sg), 0.0);
            //brdf = float3(1);
            float cosTheta = dot(cameraDir, float3(0, 0, -1));
            cosTheta = cosTheta * cosTheta * cosTheta;
            node.luminance = photon.flux * brdf / (distToCamera * distToCamera * gNormalizedPixelArea * cosTheta);
            InterlockedExchange(gHeadCounter[currentPixel], photonIndex, /*out*/ node.next); 
            gLinkedList[photonIndex] = node; 
        }
    }
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    RayDesc ray;
    PhotonLightSample pls;
    vplSampler.LightSample directSample;

    #if USE_EMISSIVE_LIGHT
        if (gFlags & (uint)GenerationFlags::AnalyticLight)
    #endif
        { // Analytic light sample
            float3 sampleDir;
            if (!createLightSampleAnalytic(sg, pls, sampleDir))
                return;

            ray.Direction = sampleDir;
        }
    #if USE_EMISSIVE_LIGHT
        else { // Emissive light sample
            if (!createLightSample(sg, pls))
                return;
            directSample.flux = pls.flux;
            float lightDirPDF = 1.0;
            float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);

            if (lightDirPDF <= 0.f)
                return;

            ray.Direction = fromLocalToWorld(pls.normal, woLocal);
            //pls.flux *= M_PI; // Lambert Emitter
            pls.flux *= dot(pls.normal, ray.Direction) / lightDirPDF;
        }
    #endif

    vplSampler.Photon photon;
    photon.flux = pls.flux;
    photon.posW = pls.posW;
    photon.origin = pls.posW;
    photon.w_i = ray.Direction;
    photon.w_o = ray.Direction;
    photon.faceN = pls.normal;
    photon.originFaceN = pls.normal;

    directSample.posW = pls.posW;
    directSample.faceN = pls.normal;

    ray.Origin = pls.posW;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    uint rayFlags = 0;
    RayData rayData = RayData();
    float3 thp = float3(1.0);
    bool isLuminaireHit = true; 
    uint i = 0;
    ShadingData sd = {};
    for (;isLuminaireHit && i < gMaxRecursion; ++i)
    {
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        photon.flux *= thp;
        photon.w_i = ray.Direction;
        photon.origin = ray.Origin;
        photon.originFaceN = photon.faceN;

        ray.TMin = 0;
        let lod = ExplicitLodTextureSampler(0.f);
        HitInfo hit = HitInfo(rayData.hit);
        if (!hit.isValid())
            return;

        if(!sdh.getShadingData(hit, -ray.Direction, sd))
            return;

        uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
        hints = 0;
        
        BSDFSample bsdfSample;
        let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = bsdf.getProperties(sd);
        if(!bsdf.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/))
            return;

        thp = bsdfSample.weight;
        if (all(thp <= 0))
            return;

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
            backprojectPhoton(sd, photon, sg);
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
