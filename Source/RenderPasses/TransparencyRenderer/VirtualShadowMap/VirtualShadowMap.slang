#pragma once

//import Rendering.ShadowMaps.ShadowTestFunctions;
//import Rendering.ShadowMaps.ShadowMapHelpers;
//__exported import Scene.Raytracing;
//__exported import Scene.Intersection;
import VirtualShadowMapData;

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
        float4 projectedPosition = mul(data.gViewProjection, float4(posW, 1.0));
        float compareDepth = data.readShadowValue(posW) + 0.001;
        //float compareDepth = data.onlyReadClipMapX(posW, 0) + 0.01;
        if (compareDepth > projectedPosition.z)
            return float3(1);
        else
            return float3(0);
    }
} gVirtualShadowMapEval;
