#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;
import Rendering.Materials.Microfacet;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import Scene.RaytracingInline;
import Scene.Raytracing;

import Helpers.SceneDataHandler;

cbuffer CB
{
    uint gFrameCount;        // Frame count since scene was loaded.
    uint gLuminaireSampleCount;
    uint gPhotonCount;
    float gConeExponent;
    float gCosOpeningAngle;
    float gPenumbraAngle;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

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

StructuredBuffer<Photon> gPhotonBuffer;

// Outputs
RWTexture2D<float4> gOutColor;
RWTexture2D<float4> gOutDebug;

// Static configuration based on defines set from the host.

static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseVPLs = USE_VIRTUAL_POINT_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

/** Payload for scatter ray (up to 72B).
*/
struct RayData
{
    PackedHitInfo hit;
    uint materialID;
    bool miss;

    /** Create ray payload with default parameters.
    */
    __init()
    {
        HitInfo hitInfo = {};
        hit = hitInfo.getData();
        materialID = 0;
        miss = false;
    }
};

struct PathData
{
    float3  radiance;       ///< Accumulated outgoing radiance from path.
    bool    terminated;     ///< Set to true when path is terminated.
    float3  thp;            ///< Current path throughput. This is updated at each path vertex.
    uint    pathLength;     ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3  origin;         ///< Next path segment origin.
    float3  direction;      ///< Next path segment direction.
    float3  neeSample;      ///< NEE sample from last frame stored for MIS with the BSDF sample of the current frame
    float3  neeSamplePosW;  ///< Position of the light sample on the emissive light source
    float   neeSamplePDF;   ///< Probability of the nee sample

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Initializes ray payload with default parameters.
    */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.neeSample = float3(0, 0, 0);
        this.neeSamplePosW = float3(0, 0, 0);
        this.neeSamplePDF = 0.0;
        this.sg = sg;
    }
};

/** Traces a scatter ray based on ray parameters stored in the ray payload.
    \param[in] rayData Describes the ray parameters. The struct is modified based on the result.
*/
void traceScatterRay(inout RayData rayData, inout PathData pathData)
{
    RayDesc ray;
    ray.Origin = pathData.origin;
    ray.Direction = pathData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0;      // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/** Evaluates the direct illumination from analytic lights.
    This function samples Falcor's light list uniformly with one shadow ray.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in,out] sg SampleGenerator object.
    \return Outgoing radiance in view direction.
*/
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls)) return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Test visibility by tracing a shadow ray.
    bool V = sdh.dispatchShadowRay(sd, ls.dir, ls.distance);
    if (!V) return float3(0.f);
       
    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
}

/** Evaluates the direct illumination from analytic lights.
    This function samples Falcor's light list uniformly with one shadow ray.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in,out] sg SampleGenerator object.
    \return Outgoing radiance in view direction.
*/
float3 samplePointLight(const float3 shadingPosW, const Photon light)
{
    float kMinLightDistSqr = 1e-9f;
    // Compute direction and distance to light.
    // The distance is clamped to a small epsilon to avoid div-by-zero below.
    float3 toLight = light.posW - shadingPosW;
    float distSqr = dot(toLight, toLight);
    distSqr =max(distSqr, kMinLightDistSqr); 
    float distance = sqrt(distSqr);
    float3 dir = toLight / distance;

    // Calculate the falloff for spot-lights.
    float cosTheta = -dot(dir, light.direction);
    float falloff = 1.f;
    float openingAngle = acos(gCosOpeningAngle); //light property
    if (cosTheta < gCosOpeningAngle)
    {
        falloff = 0.f;
    }
    else if (gPenumbraAngle > 0.f)
    {
        float deltaAngle = openingAngle - acos(cosTheta);
        falloff = smoothstep(0.f, gPenumbraAngle, deltaAngle);
    }
    float radius = tan(openingAngle);
    float3 normalizationFactor = 1 / (M_2PI * (1 - gCosOpeningAngle));
    // Compute incident radiance at shading point.
    return normalizationFactor * light.flux * falloff / distSqr;
}

/** Generate a new scatter ray or terminate.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in] isCurveHit True if on curve hit.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] rayData Ray payload.
    \return True if the path continues.
*/
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout PathData pathData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, pathData.sg, bsdfSample, kUseImportanceSampling))
    {
        pathData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            pathData.origin = sd.computeNewRayOrigin(false);
        }
        pathData.direction = bsdfSample.wo;
        pathData.thp *= bsdfSample.weight;
        return any(pathData.thp > 0.f);
    }

    return false;
}

/** Process a hit.
    Loads the shading data, samples analytic lights and samples a new scatter ray.
    Terminates the path if maximum number of bounces is reached.
    \param[in] hit Hit info.
    \param[in,out] rayData Ray payload.

*/
void handleHit(in RayData rayData, inout PathData pathData)
{

    // Add contribution from distant light (env map) in this direction.
    // TODO: add MIS weight for last NEE sample
    if (kUseEnvLight && (kComputeDirect || pathData.pathLength > 0) && rayData.miss)
    {
        pathData.radiance += pathData.neeSample;
        //float3 Le = gScene.envMap.eval(WorldRayDirection());
        //pathData.radiance += pathData.thp * Le;
        pathData.terminated = true;
        return;
    }
    HitInfo hit = HitInfo(rayData.hit);

    const bool isCurveHit = hit.getType() == HitType::Curve;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = {};
    if (!sdh.getShadingData(hit, -pathData.direction, sd))
    {
        pathData.radiance += pathData.neeSample;
        pathData.terminated = true;
        return;
    }

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    float3 selfEmission = mi.getProperties(sd).emission;
    float3 bsdfSample = float3(0);
    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || pathData.pathLength > 0) && any(selfEmission > 0))
    {
        //bsdfSample = pathData.thp * selfEmission;
        pathData.radiance = pathData.thp * selfEmission;
        pathData.terminated = true;

        //TriangleHit triangleHit = hit.getTriangleHit();
        //TriangleLightHit tlh;
        //tlh.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
        //tlh.posW = lightSd.posW;
        //tlh.normalW = lightSd.getOrientedFaceNormal();

        //float lightPDF = luminance(gEmissiveLightSampler.evalPdf(sd.posW, sd.faceN, true, tlh));
    }
    if (any(bsdfSample > 0) && any(pathData.neeSample > 0))
    {

    }
    else
    {
    }

    // Check whether to terminate based on max depth.
    if (pathData.pathLength >= kMaxBounces)
    {
        pathData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeNewRayOrigin();
    }

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, pathData.sg);
        pathData.radiance += pathData.thp * Lr;
    }

    //Add contribution of direct light from VPLs.
    if (kUseVPLs)
    {
        uint photonIndex = sampleNext1D(pathData.sg) * gLuminaireSampleCount;
        Photon lightSample = gPhotonBuffer[photonIndex]; 
        float3 toLight = lightSample.posW - sd.posW;
        float dist = length(toLight);
        toLight /= dist;
        bool isVisible = sdh.dispatchShadowRay(sd, toLight, dist);
        if (isVisible)
        {
            float3 brdf = max(mi.eval(sd, toLight, pathData.sg), 0);
            float3 Lr = samplePointLight(sd.posW, lightSample);
            pathData.neeSample = pathData.thp * brdf * Lr * float(gLuminaireSampleCount) / float(gPhotonCount);
        }
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, pathData))
    {
        pathData.terminated = true;
        return;
    }

    pathData.pathLength++;
}

/** This is the main entry point for the minimal path tracer.

    One path per pixel is generated, which is traced into the scene.
    The path tracer is written as a for-loop over path segments.

    Built-in light sources (point, directional) are sampled explicitly at each
    path vertex. The contributions from area lights (env map and mesh lights)
    are explicitly added by the scatter ray hit/miss shaders.

    \param[in] pixel Pixel to trace a path for.
    \param[in] frameDim Dimension of the frame in pixels.
    \return Returns the estimated color (radiance).
*/
float3 tracePath(const uint2 pixel, const uint2 frameDim)
{
    float3 outColor = float3(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = sdh.getRayDir(pixel);

    const HitInfo hit = sdh.getHitInfo(pixel);
    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Compute its contribution.
        const bool isCurveHit = hit.getType() == HitType::Curve;
        let lod = ExplicitLodTextureSampler(0.f);

        // Load shading data.
        ShadingData sd = {};
        sdh.getShadingData(hit, primaryRayDir, sd);


        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeNewRayOrigin();
        }

        PathData pathData = PathData(sg);

        if (kComputeDirect)
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            outColor += mi.getProperties(sd).emission;

            // Add contribution of direct light from analytic lights.
            // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
            //Add contribution of direct light from VPLs.
            if (kUseVPLs)
            {
                uint photonIndex = sampleNext1D(sg) * gLuminaireSampleCount;
                Photon lightSample = gPhotonBuffer[photonIndex]; 
                float3 toLight = lightSample.posW - sd.posW;
                float dist = length(toLight);
                toLight /= dist;
                bool isVisible = sdh.dispatchShadowRay(sd, toLight, dist);
                if (isVisible)
                {
                    float3 brdf = max(mi.eval(sd, toLight, sg), 0);
                    float3 Lr = samplePointLight(sd.posW, lightSample);
                    pathData.neeSample = brdf * Lr * float(gLuminaireSampleCount) / float(gPhotonCount);
                }
            }
        }

        RayData rayData = RayData();

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, pathData)) pathData.terminated = true;

        // Follow path into the scene and compute its total contribution.
        for (uint depth = 0; depth <= kMaxBounces && !pathData.terminated; depth++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData, pathData);
            handleHit(rayData, pathData);
        }
        // Store contribution from scatter ray.
        outColor += pathData.radiance;
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    return outColor;
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
    rayData.miss = true;
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

    float3 color = tracePath(pixel, frameDim);

    gOutColor[pixel] = float4(color, 1.f);
}
