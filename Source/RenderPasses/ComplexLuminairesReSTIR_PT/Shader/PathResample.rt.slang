#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import Scene.RaytracingInline;
import Scene.Raytracing;

import Helpers.SceneDataHandler;
import Helpers.PathReservoirHandler;
import Helpers.VPLSampler;

cbuffer UI
{
    uint gPixelRadius;
    float gMinConnectionDistance;
    float gRoughnessThreshold;
}

cbuffer PerFrame
{
    uint gFrameCount;
}

RWStructuredBuffer<prh.CausticReservoir> gCausticReservoir;
RWStructuredBuffer<prh.CausticReservoir> gCausticReservoirPrev;

RWStructuredBuffer<prh.PathReservoir> gPathReservoir;
RWStructuredBuffer<prh.PathReservoir> gPathReservoirPrev;

RWTexture2D<float4> gOutDebug;
RWTexture2D<float4> gOutColor;

/** Payload for scatter ray (up to 72B).
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

float3 resampleCaustic(const uint index, const uint prevIndex, const uint2 pixel, inout SampleGenerator sg)
{
    prh.CausticReservoir reservoir = gCausticReservoir[index];
    prh.CausticReservoir temporalReservoir = gCausticReservoirPrev[prevIndex];
    prh.CausticSample temporalSample = temporalReservoir.sample;
    const uint c = temporalReservoir.confidenceWeight;
    float temporalWeight = temporalSample.w;

    //bool applyMIS = temporalSample.pathLength == reservoir.sample.pathLength;
    //gOutDebug[pixel] = float4(temporalSample.pathLength, reservoir.sample.pathLength, 0, 1);
    //if (applyMIS)
    //{
    //    float misSum = reservoir.confidenceWeight * reservoir.sample.pathPDF + c * temporalSample.pathPDF;
    //    float mis = 0.0;
    //    float temporalMIS = 0.0;

    //    if (misSum > 0)
    //    {
    //        mis = reservoir.confidenceWeight * reservoir.sample.pathPDF / misSum;

    //        temporalMIS = c * temporalSample.pathPDF / misSum;
    //        reservoir.w_sum *= mis;
    //    }
    //    temporalWeight *= temporalMIS;
    //}
    //reservoir.w_sum *= reservoir.sample.targetFunction;
    //reservoir.addSample(temporalSample, c, temporalWeight, sg);
    //reservoir.finalizeSample();

    float3 color = reservoir.sample.pathContribution * reservoir.w_sum;
    //gCausticReservoir[index] = reservoir;
    return color;
}

void hybridShift(uint2 pixel, inout prh.PathReservoir reservoir)
{
}

float3 resamplePath(const uint index, const uint prevIndex, const uint2 pixel, inout SampleGenerator sg)
{
    prh.PathReservoir reservoir = gPathReservoir[index];
    prh.PathReservoir temporalReservoir = gPathReservoirPrev[prevIndex];
    prh.PathSample temporalSample = temporalReservoir.sample;
    const uint c = temporalReservoir.confidenceWeight;
    float temporalWeight = temporalSample.w;

    gOutDebug[pixel] = float4(reservoir.sample.rd.pathLength);

    float3 color = reservoir.sample.pathContribution * reservoir.w_sum;
    //gPathReservoir[index] = reservoir;
    return color;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void miss(inout RayData rayData)
{
    //write invalid hit
    HitInfo hitInfo = {};
    rayData.hit = hitInfo.getData();
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.hit = HitInfo(triangleHit).getData();
    rayData.materialID = gScene.getMaterialID(triangleHit.instanceID);
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    sdh.copyVBuffer(pixel);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    uint index = prh.getIndex(pixel);

    uint2 prevPixel = pixel;
    //float2 rnd = sampleNext2D(sg);
    //bool valid = sdh.applyMotionVector(prevPixel, rnd);
    uint prevIndex = prh.getIndex(prevPixel);

    float3 color = float3(0);

    color += resampleCaustic(index, prevIndex, pixel, sg);
    color += resamplePath(index, prevIndex, pixel, sg);

    gOutColor[pixel] = float4(color, 1);
}
