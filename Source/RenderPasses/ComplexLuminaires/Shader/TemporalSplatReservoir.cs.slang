#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Rendering.RTXDI.RTXDI;

import Helpers.SceneDataHandler;
import Helpers.ReservoirHandler;

cbuffer CB {
    uint2 gFrameDim;
}

RWStructuredBuffer<uint> gCellCounter;
RWStructuredBuffer<uint> gGlobalCounter;
RWStructuredBuffer<uint4> gSplatSortData;

RWTexture2D<PackedHitInfo> gSplattingHits;

RWTexture2D<float4> gOutDebug;
RWTexture2D<float4> gOutDebug1;

uint index2Dto1D(uint2 index, uint frameDimX) {
    return index.x + index.y * frameDimX;
}

// Load in shading data from hit and previous hit position
ShadingData loadShadingDataWithPrevHit(const HitInfo hit, const float3 cameraOrigin, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    float3 dir = normalize(cameraOrigin - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, dir, lod);
    return sd;
}

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    let textureSampler = ExplicitLodTextureSampler(0);
    gOutDebug1[pixel] = float4(0);

    const HitInfo hit = HitInfo(gSplattingHits[pixel]);
    if (!hit.isValid())
        return;

    ShadingData sd = loadShadingDataWithPrevHit(hit, gScene.camera.getPosition(), textureSampler);

    //Reproject postion in current camera view
    float4 screenPos = mul(gScene.camera.getViewProj(), float4(sd.posW, 1));
    screenPos /= screenPos.w;
    screenPos.y *= -1;
    screenPos.xy = screenPos.xy * 0.5 + 0.5; // UV from [-1,1] to [0,1]
    if (any(screenPos.xy < 0) || any(screenPos.xy >= 1) || screenPos.z < 0 || screenPos.z > 1)
        return;

    //Visibility test (from point to camera)
    float3 directionToCamera = gScene.camera.getPosition() - sd.posW;
    float distanceToCamera = length(directionToCamera);
    directionToCamera /= distanceToCamera;

    SceneRayQuery<1> rayQuery;
    const Ray ray = Ray(sd.computeNewRayOrigin(), directionToCamera, 0.f, distanceToCamera);
    if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff))
        return;

    uint2 newPixel = uint2(floor(screenPos.xy * gFrameDim));
    uint linPixel = index2Dto1D(newPixel, gFrameDim.x);
    uint index = 0;
    InterlockedAdd(gGlobalCounter[kSplattingGlobalCounterElement], 1, index);

    uint cellIndex = 0;
    InterlockedAdd(gCellCounter[linPixel], 1, cellIndex);

    gSplatSortData[index] = uint4(linPixel, cellIndex, pixel); 
}
