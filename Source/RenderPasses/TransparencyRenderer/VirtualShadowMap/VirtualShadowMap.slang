#pragma once

//import Rendering.ShadowMaps.ShadowTestFunctions;
//import Rendering.ShadowMaps.ShadowMapHelpers;
//__exported import Scene.Raytracing;
//__exported import Scene.Intersection;
import VirtualShadowMapData;

#ifndef NUM_CLIPMAPS
#define NUM_CLIPMAPS 1
#endif

#define data gVirtualShadowMapData

struct VirtualShadowMapShader
{
    /** Caculates the shadow value by evaluating the shadow acceleration structure
     * \param[in] index of the current light
     * \param[in] world position of the current hit
     * \param[in] is the light directional (ortho projection)
     * \param[in] Importance. Is used to weight the access texture. Should be in (0, 1]
     * \return visibility in [0,1]
     */
    float3 eval(uint lightIdx, float3 posW, bool isDirectional = false, float importance = 1.0) {
        uint clipMapLevel = 0;
        uint clipMapIdx = data.posWToClipMapIdx(posW);
        if (clipMapIdx == NUM_CLIPMAPS)
            return float3(1);
        float projectedPositionZ = mul(data.gViewProjection[clipMapIdx], float4(posW, 1.0)).z;
        float compareDepth = data.readShadowValue(posW) + 0.0001;
        //float compareDepth = data.onlyReadClipMapX(posW, clipMapLevel) + 0.01;
        if (compareDepth > projectedPositionZ)
            return float3(1);
        else
            return float3(0);
    }
} gVirtualShadowMapEval;
