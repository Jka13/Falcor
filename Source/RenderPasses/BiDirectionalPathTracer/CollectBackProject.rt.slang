#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Material.ShadingUtils;
import Rendering.Lights.LightHelpers;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;

#ifndef NUM_PIXEL
#define NUM_PIXEL 1
#endif

cbuffer PerFrame
{
    uint gFrameCount; // Frame count since scene was loaded.
    float2 gPhotonRadius; // x->Global Radius, y-> Caustic Radius
    uint _pad;
}

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

// Inputs
Buffer<uint> gLightCounter;
RWStructuredBuffer<float4> gLightTraceData;
RWStructuredBuffer<ListNode> gLinkedList;
RWTexture2D<uint> gHeadCounter;
RWBuffer<float4> gLightPaths;
RWBuffer<float4> gCameraPaths;

// Outputs
RWTexture2D<float4> gColor;

/** Payload for ray (48B).
*/
struct RayData
{
    float3 radiance; ///< Accumulated outgoing radiance from path.
    bool _pad; ///< free space.
    PackedHitInfo packedHitInfo; ///< Hit info from vBuffer; Up to 16B

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).
  
    __init(){
        this.radiance = float3(0);
        this._pad = false;
    }
};

struct SphereAttribs
{
    float2 pad;
};

//Load in shading data from hit and direction
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    
    return sd;
}

uint index2Dto1D(uint2 idx, uint width)
{
    return idx.x + width * idx.y;
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
    float4 L = gLightTraceData[primIndex];
            
    rayData.radiance += L.xyz;
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
}

float3 getLuminance()
{
    uint2 headTextureIndex = DispatchRaysIndex().xy; 
    uint linkedListIndex = gHeadCounter[headTextureIndex];
    float3 luminance = float3(0.0); 
    while (linkedListIndex != uint(-1)) {
        ListNode currentNode = gLinkedList[linkedListIndex];    
        luminance += currentNode.luminance;
        linkedListIndex = currentNode.next;
    }
    return luminance;
}


[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    #if MODE == 0
    gColor[launchIndex] = float4(getLuminance(), 1.f); 
    #endif
}
