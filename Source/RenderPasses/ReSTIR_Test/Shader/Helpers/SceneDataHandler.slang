#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

import Scene.RaytracingInline;

struct SceneDataHandler
{

    Texture2D<PackedHitInfo> gVBuffer;
    Texture2D<float4> gView;

    bool getShadingData(HitInfo hit, float3 rayDir, out ShadingData sd)
    {
        let textureSampler = ExplicitLodTextureSampler(0);
        sd = {};
        if (!hit.isValid())
            return false;
        TriangleHit triangleHit = hit.getTriangleHit();
        VertexData vertexData = gScene.getVertexData(triangleHit);
        uint materialId = gScene.getMaterialID(triangleHit.instanceID);

        sd = gScene.materials.prepareShadingData(vertexData, materialId, rayDir, textureSampler);
        return true;
    }

    float3 getRayDir(uint2 pixel)
    {
        return gView[pixel].xyz;
    }

    HitInfo getHitInfo(uint2 pixel)
    {
        return HitInfo(gVBuffer[pixel]);
    }

} sdh;
