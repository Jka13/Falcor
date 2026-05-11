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
import ReservoirHandler;

static const float kShadowRayEpsilon = 0.99;
static const uint kSplattingGlobalCounterElement = 0; //Index for the platting global counter, that counts the total number of elements
static const uint kSplattingGlobalCounterPrefixSum = 1; //Index for the platting global counter, used for the prefix sum to determine the offsets

struct SceneDataHandler
{

    Texture2D<PackedHitInfo> gVBuffer;
    Texture2D<float4> gView;
    Texture2D<float2> gMVec;

    Texture2D<PackedHitInfo> gPrevVBuffer;
    Texture2D<float4> gPrevView;
    RWTexture2D<PackedHitInfo> gPrevVBufferWrite;
    RWTexture2D<float4> gPrevViewWrite;

    void copyVBuffer(uint2 pixel)
    {
        gPrevVBufferWrite[pixel] = gVBuffer[pixel];
        gPrevViewWrite[pixel] = gView[pixel];
    }

    float3 getRayDir(uint2 pixel)
    {
        return gView[pixel].xyz;
    }

    HitInfo getHitInfo(uint2 pixel)
    {
        return HitInfo(gVBuffer[pixel]);
    }

    bool applyMotionVector(inout uint2 pixel, float2 rndOffset)
    {
        float2 offset = gMVec[pixel] * rh.gFrameDim;
        float2 pixelF = float2(pixel) + offset + rndOffset;
        bool isValid = all(pixelF >= 0) && all(pixelF < rh.gFrameDim);
        pixel = int2(pixelF);
        return isValid;  
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

    bool getShadingData(uint2 pixel, float3 rayDir, out ShadingData sd)
    {
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

    bool getPrevShadingData(uint2 pixel, out ShadingData sd)
    {
        float3 rayDir = gPrevView[pixel].xyz;
        HitInfo hit = HitInfo(gPrevVBuffer[pixel]);
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

    bool getPrevShadingData(uint2 pixel, float3 rayDir, out ShadingData sd)
    {
        HitInfo hit = HitInfo(gPrevVBuffer[pixel]);
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


    bool dispatchShadowRay(const ShadingData sd, const float3 toLight, const float distanceToLight, const bool reflect = true) {
        // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
        SceneRayQuery<1> rayQuery;
        const Ray ray = Ray(sd.computeNewRayOrigin(reflect), toLight, 0.0, distanceToLight * kShadowRayEpsilon);
        return rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

    bool dispatchShadowRay(const float3 origin, const float3 toLight, const float distanceToLight, const bool applyShadowRayEpsilon = true) {
        // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
        SceneRayQuery<1> rayQuery;
        float distance = applyShadowRayEpsilon ? distanceToLight * kShadowRayEpsilon : distanceToLight;
        const Ray ray = Ray(origin, toLight, 0.0, distance);
        return rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

} sdh;
