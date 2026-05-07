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
import Helpers.VPLSampler;

cbuffer CB
{
    uint gFrameCount;        // Frame count since scene was loaded.
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

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
static const bool kUseBSDFSamples = USE_BSDF_SAMPLES;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;
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
    float3  neeSamplePosW;  ///< Position of the light sample on the emissive light source
    float   neeSamplePDF;   ///< Probability of the NEE sample
    float   bsdfSamplePDF;  ///< Probability of the BSDF sample
    float3  lastHitPosW;    ///< World position of the last hit 
    float3  lastHitFaceN;   ///< Face normal of the last hit
    bool    isDelta;        ///< Set to false if the current shading point has delta distribution 


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
        this.neeSamplePosW = float3(0, 0, 0);
        this.neeSamplePDF = 0.0;
        this.bsdfSamplePDF = 0.0;
        this.lastHitPosW = float3(0,0,0);
        this.lastHitFaceN = float3(0,0,0);
        this.sg = sg;
    }
};

/** Describes a light sample.
*/
struct LightSample
{
    float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float   pdf;        ///< Pdf with respect to solid angle at the shading point.
    float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3  toLight;    ///< Ray direction for visibility evaluation (normalized).
    uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).
    float3  faceN;      ///< Normal of the sampled triangle for emissive lights.
};

/** Get position with offset applied in direction of the geometry normal to avoid self-intersection
    for visibility rays.
    \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
    \return Returns the offseted position.
*/
float3 getRayOrigin(float3 posW, float3 faceN, float3 lightPos)
{
    float3 rayDir = lightPos - posW;
    return computeRayOrigin(posW, dot(faceN, rayDir) >= 0 ? faceN : -faceN);
}

float evalMIS(float p0, float p1)
{
    // Balance heuristic
    float misSum = p0 + p1;
    return misSum > 0.0 ? p0 / misSum : 0.0;
}

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

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample(const ShadingData sd, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    TriangleLightSample tls;
    if (!gEmissiveLightSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = getRayOrigin(sd.posW, sd.faceN, lightPos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.toLight = toLight / ls.distance;
    ls.faceN = tls.normalW;

    return any(ls.Li > 0.f);
}

/** Generate a new scatter ray or terminate.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in] isCurveHit True if on curve hit.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] rayData Ray payload.
    \return True if the path continues.
*/
bool generateScatterRay(const ShadingData sd, const BSDFSample bsdfSample, bool isCurveHit, float3 rayOrigin, inout PathData pathData)
{
    pathData.origin = rayOrigin;
    if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
    {
        pathData.origin = sd.computeNewRayOrigin(false);
    }
    pathData.direction = bsdfSample.wo;
    pathData.thp *= bsdfSample.weight;
    pathData.bsdfSamplePDF = bsdfSample.pdf;
    // Handle delta events.
    return any(pathData.thp > 0.f);
}

/** Process a hit.
    Loads the shading data, samples analytic lights and samples a new scatter ray.
    Terminates the path if maximum number of bounces is reached.
    \param[in] hit Hit info.
    \param[in,out] rayData Ray payload.

*/
void handleHit(in RayData rayData, inout PathData pathData, uint2 pixel)
{

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || pathData.pathLength > 0) && rayData.miss)
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        pathData.radiance += pathData.thp * Le;
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
        pathData.terminated = true;
        return;
    }

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    float3 selfEmission = mi.getProperties(sd).emission;

    // Add emitted light.
    if (any(selfEmission > 0) && kUseBSDFSamples && kUseEmissiveLights && (kComputeDirect || pathData.pathLength > 0))
    {
        if(kUseMIS && !pathData.isDelta)
        {
            TriangleHit triangleHit = hit.getTriangleHit();
            TriangleLightHit tlh;
            tlh.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            tlh.posW = sd.posW;
            tlh.normalW = sd.getOrientedFaceNormal();

            //Calculate MIS weight for the BSDF sample
            float lightPDF = luminance(gEmissiveLightSampler.evalPdf(pathData.origin, pathData.lastHitFaceN, true, tlh));
            selfEmission *= evalMIS(pathData.bsdfSamplePDF, lightPDF);
        }
        pathData.radiance += pathData.thp * selfEmission;
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
    BSDFSample bsdfSample;
    pathData.terminated = !mi.sample(sd, pathData.sg, bsdfSample, kUseImportanceSampling);
    pathData.isDelta = bsdfSample.isLobe(LobeType::Delta);

    //Add contribution of direct light from VPLs.
    if (kUseVPLs)
    {
        vplSampler.Photon lightSample = vplSampler.sampleVPLsUniform(pathData.sg);
        pathData.neeSamplePosW = lightSample.lightSamplePosW;
        pathData.neeSamplePDF = lightSample.lightSampleSelectionPDF;
        float3 toLight = lightSample.posW - sd.posW;
        float dist = length(toLight);
        toLight /= dist;
        float cosTheta = dot(lightSample.faceN, -toLight); 
        bool isFrontfacing = cosTheta > 0.0; 
        if (isFrontfacing)
        {
            bool isVisible = sdh.dispatchShadowRay(sd, toLight, dist);
            if (isVisible)
            {
                float3 brdf = max(mi.eval(sd, toLight, pathData.sg), 0);
                float3 Lr = vplSampler.samplePointLight(sd.posW, lightSample);
                // Apply MIS weight
                if (kUseMIS)
                {
                    float3 toLight = normalize(pathData.neeSamplePosW - sd.posW);
                    float bsdfPDF = mi.evalPdf(sd, toLight, kUseImportanceSampling);
                    Lr *= evalMIS(pathData.neeSamplePDF, bsdfPDF);
                }
                pathData.radiance += pathData.thp * brdf * Lr;
            }
        }
    }

    //Add contribution of direct light from NEE
    if(kUseNEE && kUseEmissiveLights)
    {
        LightSample ls = {};
        ls.pdf = 0.0;
        if(generateEmissiveSample(sd, true, pathData.sg, ls))
        {
            // Apply MIS weight.
            if (kUseMIS)
            {
                float scatterPdf = mi.evalPdf(sd, ls.toLight, kUseImportanceSampling);

                float cosTheta = dot(ls.faceN, -ls.toLight);
                scatterPdf *= ls.distance * ls.distance / cosTheta;

                ls.Li *= evalMIS(ls.pdf, scatterPdf);
            }
            float3 brdf = max(mi.eval(sd, ls.toLight, pathData.sg), 0);
            float3 Lr = pathData.thp * brdf * ls.Li;
            if (any(Lr > 0.0))
            {
                bool isVisible = sdh.dispatchShadowRay(sd, ls.toLight, ls.distance);
                if(isVisible)
                {
                    pathData.radiance += Lr;
                }
            }
        }
    }

    // Generate scatter ray for the next path segment.
    // Update isDelta based on the BSDF sample lobe.
    // The raygen shader will continue the path based on the returned payload.
    pathData.terminated = !generateScatterRay(sd, bsdfSample, isCurveHit, rayOrigin, pathData);
    pathData.lastHitFaceN = sd.getOrientedFaceNormal();
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
        if(!sdh.getShadingData(hit, primaryRayDir, sd))
        {
            return outColor;
        }

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
        // Initialize path data
        PathData pathData = PathData(sg);

        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod);


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

        pathData.lastHitFaceN = sd.getOrientedFaceNormal(); 
        RayData rayData = RayData();

        if (kComputeDirect)
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            outColor += mi.getProperties(sd).emission;

            // Add contribution of direct light from analytic lights.
            // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
        }

        BSDFSample bsdfSample;
        pathData.terminated = !mi.sample(sd, pathData.sg, bsdfSample, kUseImportanceSampling);
        pathData.isDelta = bsdfSample.isLobe(LobeType::Delta);

        //Add contribution of direct light from VPLs.
        if (kUseVPLs)
        {
            vplSampler.Photon lightSample = vplSampler.sampleVPLsUniform(pathData.sg);
            pathData.neeSamplePosW = lightSample.lightSamplePosW;
            pathData.neeSamplePDF = lightSample.lightSampleSelectionPDF;
            float3 toLight = lightSample.posW - sd.posW;
            float dist = length(toLight);
            toLight /= dist;
            float cosTheta = dot(lightSample.faceN, -toLight); 
            bool isFrontfacing = cosTheta > 0.0; 
            if (isFrontfacing)
            {
                bool isVisible = sdh.dispatchShadowRay(sd, toLight, dist);
                if (isVisible)
                {
                    float3 brdf = max(mi.eval(sd, toLight, pathData.sg), 0);
                    float3 Lr = vplSampler.samplePointLight(sd.posW, lightSample);

                    float neeSampleMIS = 1.0;
                    if (kUseMIS)
                    {
                        //Calculate MIS weight for the NEE sample
                        float3 toLight = normalize(pathData.neeSamplePosW - sd.posW);
                        float bsdfPDF = luminance(mi.evalPdf(sd, toLight, true));
                        float misSum = bsdfPDF + pathData.neeSamplePDF;
                        neeSampleMIS = misSum > 0.0 ? pathData.neeSamplePDF / misSum : 0.0; 
                    }
                    pathData.radiance += neeSampleMIS * pathData.thp * brdf * Lr;
                }
            }
        }

        //Add contribution of direct light from NEE
        if(kUseNEE && kUseEmissiveLights)
        {
            LightSample ls = {};
            if(generateEmissiveSample(sd, true, pathData.sg, ls))
            {
                // Apply MIS weight.
                if (kUseMIS)
                {
                    float scatterPdf = mi.evalPdf(sd, ls.toLight, kUseImportanceSampling);
                    ls.Li *= evalMIS(ls.pdf, scatterPdf);
                }
                float3 brdf = max(mi.eval(sd, ls.toLight, pathData.sg), 0);
                float3 Lr = brdf * ls.Li;
                if (any(Lr > 0.0))
                {
                    bool isVisible = sdh.dispatchShadowRay(sd, ls.toLight, ls.distance);
                    if(isVisible)
                    {
                        pathData.radiance += Lr;
                    }
                }
            }
        }

        // Generate scatter ray.
        pathData.terminated = !generateScatterRay(sd, bsdfSample, isCurveHit, rayOrigin, pathData);

        // Follow path into the scene and compute its total contribution.
        for (uint depth = 0; depth <= kMaxBounces && !pathData.terminated; depth++)
        {
            gOutDebug[pixel] = float4(0);
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData, pathData);
            handleHit(rayData, pathData, pixel);
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
