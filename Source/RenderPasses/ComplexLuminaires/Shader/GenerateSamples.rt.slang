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
}

cbuffer CB
{
    uint gMaxRecursion;
    uint gFlags;
    float gAABBSize;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

RWStructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<AABB> gPhotonAABBs;
RWBuffer<uint> gPhotonCounter;

struct PhotonLightSample
{
    float3 posW;
    float3 normal;
    float3 flux;
};

struct Photon
{
    float3 posW;
    float3 direction;
    float3 flux;

    __init()
    {
        posW = float3(0);
        direction = float3(0);
        flux = float3(0);
    }
}

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

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    RayDesc ray;
    PhotonLightSample pls;
    Photon photon;

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

            float lightDirPDF = 1.0;
            float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);

            if (lightDirPDF <= 0.f)
                return;

            ray.Direction = fromLocalToWorld(pls.normal, woLocal);
            pls.flux *= M_PI; // Lambert Emitter
        }
    #endif

    photon.flux = pls.flux;

    ray.Origin = pls.posW;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;

    uint rayFlags = 0;
    RayData rayData = RayData();
    float3 thp = float3(1.0);
    bool isLuminaireHit = true; //only save photons that do not directly leave the luminaire as those are handled by direct light
    bool valid = true;
    uint i = 0;
    for (; i < gMaxRecursion; ++i)
    {
        #if MODE == 1
        if (!isLuminaireHit)
            break;
        #endif
        photon.flux *= thp;

        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
        ray.TMin = 0;
        let lod = ExplicitLodTextureSampler(0.f);
        HitInfo hit = HitInfo(rayData.hit);
        if (!hit.isValid())
            return;
        isLuminaireHit = gScene.materials.isCastShadow(rayData.materialID) == 0;

        #if MODE != 1
        if (!isLuminaireHit)
            break;
        #endif

        ShadingData sd = {};
        valid = sdh.getShadingData(hit, -ray.Direction, sd);
        if (!valid)
            return;

        //Adjust shading normals if desired
        uint hints = 0;
        //if (gFlags & (uint) GenerationFlags::AdjShadingNormals)
        {
            hints |= (uint) MaterialInstanceHints::AdjustShadingNormal;
        }
        
        BSDFSample bsdfSample;
        let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = bsdf.getProperties(sd);
        valid = bsdf.sample(sd, sg, bsdfSample, true /*ImportanceSampling*/);
        if (!valid)
            return;

        thp = bsdfSample.weight;
        if (all(thp <= 0))
            return;

        #if MODE == 1
        photon.direction = ray.Direction;
        #endif 

        //Compute new ray origin to avoid self intersection
        bool transmission = bsdfSample.isLobe(LobeType::Transmission);
        ray.Origin = sd.computeNewRayOrigin(!transmission);
        ray.Direction = bsdfSample.wo;

        photon.posW = ray.Origin;
        #if MODE != 1
        photon.direction = ray.Direction;
        #endif
    }

    uint photonIndex = 0;
    InterlockedAdd(gPhotonCounter[0], 1u, photonIndex);
    if (i > 0 && !isLuminaireHit)
    {
        gPhotonBuffer[photonIndex] = photon;
        AABB photonAABB = AABB(float3(photon.posW - gAABBSize), float3(photon.posW + gAABBSize));
        gPhotonAABBs[photonIndex] = photonAABB;
    }
}
