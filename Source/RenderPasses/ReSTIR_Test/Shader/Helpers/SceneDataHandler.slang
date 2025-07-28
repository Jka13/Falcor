#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

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

    float3 getRayDir(uint2 pixel)
    {
        return gView[pixel].xyz;
    }

    HitInfo getHitInfo(uint2 pixel)
    {
        return HitInfo(gVBuffer[pixel]);
    }

    bool getShadingData(uint2 pixel, out ShadingData sd)
    {
        float3 rayDir = gView[pixel].xyz;
        HitInfo hit = HitInfo(gVBuffer[pixel]);
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

    float evalApproximateBRDF(ShadingData sd, BSDFProperties bsdfProperties, float3 toLight)
    {
        const float3 N = bsdfProperties.guideNormal;
        const float3 H = normalize(sd.V + toLight);
        const float NdotV = saturate(dot(N, sd.V));
        const float NdotL = saturate(dot(N, toLight));
        const float NdotH = saturate(dot(N, H));
        const float LdotH = saturate(dot(toLight, H));
        const float diffuse = luminance(bsdfProperties.diffuseReflectionAlbedo);
        const float specular = luminance(bsdfProperties.specularReflectionAlbedo);

        // BRDF terms for target function (DGF Microfaset BRDF)
        float D = evalNdfGGX(bsdfProperties.roughness * bsdfProperties.roughness, NdotH);
        float G = evalMaskingSmithGGXSeparable(bsdfProperties.roughness, NdotL, NdotV);

        // float F = specular < 1e-8f ? 0.f :  evalFresnelSchlick(specular, 1.f, LdotH);     //Right way
        float F = specular < 1e-8f ? 0.f : evalFresnelSchlick(specular, 1.f, LdotH) / specular; // Not "right" but less noisier

        // eval brdf terms
        float diffuseBRDF = NdotL * M_1_PI;
        float specularBRDF = D * G * F / (4.f * NdotV);

        // Mix diffuse and specular BRDF
        float diffuseProb = diffuse / (diffuse + specular);
        float brdf = max(0.f, lerp(specularBRDF, diffuseBRDF, diffuseProb));
        return max(0.f, brdf);
    }


} sdh;
