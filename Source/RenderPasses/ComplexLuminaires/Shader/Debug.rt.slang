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
    float3 color;
    /** Create ray payload with default parameters.
    */
    __init()
    {
        color = float3(0);
    }
};

struct SphereAttribs
{
    float2 pad;
};

cbuffer PerFrame
{
    uint gFrameCount;
}

cbuffer CB
{
    uint gFlags;
}

//Acceleration Structure
RaytracingAccelerationStructure gPhotonAccelerationStructure;

RWStructuredBuffer<AABB> gPhotonAABBs;
//Ouput
RWTexture2D<float4> gOutColor;

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
    // float4 L = gLightTraceData[primIndex];
    //
    // rayData.radiance += L.xyz;
    rayData.color = float3(1.0, 0.0, 0.0);
    IgnoreHit();
}

[shader("intersection")]
void intersection()
{
    //Attribs are needed by the api. Because we have no need for additional attributes it is empty const float3 origin = ObjectRayOrigin();
    SphereAttribs attribs;
    attribs.pad = float2(0);
    ReportHit(RayTCurrent(), 0, attribs);
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    ShadingData sd = {};
    bool valid = sdh.getShadingData(pixel, sd);

    let lod = ExplicitLodTextureSampler(0.0);
    uint hints = 0;

    BSDFSample bsdfSample;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);

    RayData rayData = RayData();
    rayData.color = bsdfProperties.diffuseReflectionAlbedo * 0.1;

    //Ray description 
    Ray cameraRay = gScene.camera.computeRayPinhole(pixel, launchDim, true);
    RayDesc ray = cameraRay.toRayDesc();

    //Closest hit shader can be skipped
    uint rayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES;
    TraceRay(gPhotonAccelerationStructure, rayFlags, 0xFF /* instanceInclusionMask */, 0 /* hitIdx */, 0 /* rayType count */, 0 /* missIdx */, ray, rayData);

    gOutColor[pixel] = float4(rayData.color, 1.0);
}
