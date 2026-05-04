#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"


import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import Scene.RaytracingInline;
import Scene.Raytracing;
import Scene.Intersection;
import Scene.Material.ShadingUtils;

import Rendering.Lights.LightHelpers;

import Helpers.SceneDataHandler;

/** Payload for scatter ray (80B).
*/
struct RayData
{
    float3 radiance;
    float3 color;
    PackedHitInfo hit;

    SampleGenerator sg;
    /** Create ray payload with default parameters.
    */
    __init()
    {
        radiance = float3(0);
        color = float3(0);
        hit = uint4(0);
    }
};

struct SphereAttribs
{
    float radius;
};

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer CB
{
    uint gFlags;
    uint gPhotonCount;
    float gPhotonRadius;
}

//Acceleration Structure
RaytracingAccelerationStructure gPhotonAccelerationStructure;

StructuredBuffer<AABB> gPhotonAABBs;
StructuredBuffer<Photon> gPhotonBuffer;
//Ouput
RWTexture2D<float4> gOutColor;

struct Photon
{
    float3  posW;
    float3  direction;
    float3  flux;
    float3  lightSamplePosW;
    float   lightSampleSelectionPDF;

    __init()
    {
        posW = float3(0);
        direction = float3(0);
        flux = float3(0);
        lightSamplePosW = float3(0);
        lightSampleSelectionPDF = 0.0;
    }
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    //Empty. Is needed for compilation
}


[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, SphereAttribs attribs : SV_IntersectionAttributes)
{
    //Get Photon data
    const uint primIndex = PrimitiveIndex();
    Photon pd = gPhotonBuffer[primIndex];

    //Get hit data from payload
    const HitInfo hit = HitInfo(rayData.hit);
    let lod = ExplicitLodTextureSampler(0.f);
    // World Direction is set to the viewDir we get from the vBuffer
    ShadingData sd = {};
    bool valid = sdh.getShadingData(hit, WorldRayDirection(), sd);

    //Check for different surfaces
    //if (dot(pd.direction, sd.faceN) < 0.6)
    //{
    //    return;
    //}

    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    float3 f_r = bsdf.eval(sd, pd.direction, rayData.sg);
    float NdotL = saturate(dot(bsdfProperties.guideNormal, pd.direction));
    if (NdotL > 0)
        rayData.radiance += max(0.f, (f_r * pd.flux) / (NdotL * attribs.radius * attribs.radius));  //Take out the NdotL that is included in the eval function
    IgnoreHit();
}

//Checks if the ray start point is inside the sphere.
bool hitSphere(const float3 center, const float radius, const float3 p)
{
    float3 radiusTest = p - center;
    radiusTest = radiusTest * radiusTest;
    float radiusTestF = radiusTest.x + radiusTest.y + radiusTest.z;
    if (radiusTestF < radius * radius)
        return true;
    return false;
}

[shader("intersection")]
void intersection()
{
    //Attribs are needed by the api. Because we have no need for additional attributes it is empty const float3 origin = ObjectRayOrigin();
    const float3 origin = ObjectRayOrigin();
    const uint primIndex = PrimitiveIndex();

    SphereAttribs attribs;
    AABB photonAABB = gPhotonAABBs[primIndex];
    float radius = (photonAABB.maxPoint.x - photonAABB.minPoint.x) / 2;
    attribs.radius = radius;
    if (hitSphere(photonAABB.center(), radius, origin))
        ReportHit(RayTCurrent(), 0, attribs);
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    RayData rayData = RayData();

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);

    let lod = ExplicitLodTextureSampler(0.0);
    uint hints = 0;

    BSDFSample bsdfSample;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);

    rayData.hit = sdh.gVBuffer[pixel];
    rayData.sg = sg;

    //Ray description
    RayDesc ray;
    ray.Direction = -sdh.getRayDir(pixel);
    ray.Origin = sd.posW;
    ray.TMin = 0.0;
    ray.TMax = 1e-7;
    float3 radiance = float3(0);

    if (valid)
    {
        //Closest hit shader can be skipped
        uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES;
        TraceRay(gPhotonAccelerationStructure, rayFlags, 0xFF /* instanceInclusionMask */, 0 /* hitIdx */, 0 /* rayType count */, 0 /* missIdx */, ray, rayData);
        float w = 1.0 / M_PI;
        radiance = w * rayData.radiance; 
    }
    gOutColor[pixel] = float4(bsdfProperties.emission + radiance / float(gPhotonCount), 1.0);
}
