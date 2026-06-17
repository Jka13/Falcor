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

cbuffer CB
{
    uint gFrameCount;        // Frame count since scene was loaded.
    float gRoughnessThreshold;
    float gMinConnectionDistance;
}

cbuffer LightCB
{
    EmissiveLightSampler gEmissiveLightSampler;
}

//Debug
RWStructuredBuffer<prh.PathDebugData> gPathDebugBuffer;
RWTexture2D<float4> gOutDebug;

// Outputs
RWTexture2D<float4> gOutColor;

// Reprojection Data
RWTexture2D<uint> gHeadCounter;
RWStructuredBuffer<ListNode> gLinkedList;

// Reservoir Buffer
RWStructuredBuffer<prh.CausticReservoir> gCausticReservoir;
RWStructuredBuffer<prh.PathReservoir> gPathReservoir;

// ReconnectionData
RWStructuredBuffer<prh.ReconnectionData> gReconnectionData;
RWStructuredBuffer<vplSampler.DirectVPL> gNEESampleBuffer;

// Static configuration based on defines set from the host.

static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseVPLs = USE_VIRTUAL_POINT_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const bool kUseDirectVPLs = USE_DIRECT_POINT_LIGHTS;
static const bool kUseIndirectVPLs = USE_INDIRECT_POINT_LIGHTS;
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

    /** Create ray payload with default parameters.
    */
    __init()
    {
        HitInfo hitInfo = {};
        hit = hitInfo.getData();
        materialID = 0;
    }
};

struct ListNode
{
    float3  luminance;
    float   pathPDF;
    uint    pathLength;
    int     next;

    __init()
    {
        luminance = float3(0);
        pathPDF = 0.0;
        next = -1;
    }
};

struct PathData
{
    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).
    SampleGenerator pathSG; ///< Distinct sample generator for path generation.
    float3  radiance;           ///< Accumulated outgoing radiance from path.
    float   pdf;                ///< Path probability
    float3  brdf;               ///< BRDF of last hit to sample 
    uint    pathLength;         ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3  thp;                ///< Current path throughput. This is updated at each path vertex.
    float   bsdfSamplePDF;      ///< Probability of the BSDF sample
    float3  reconnectionThp;
    float   lastHitRoughness;
    float3  origin;             ///< Next path segment origin.
    uint    prevLobeType;       ///< acitve lobe type flags of previous bsdf sample
    float3  direction;          ///< Next path segment direction.
    float   lastCos;
    float3  lastOrigin;         ///< Origin of the last ray
    float   pad2;
    float3  lastDirection;      ///< Direction of the last ray
    float   pad3;
    float3  lastHitFaceN;       ///< Face normal of the last hit
    bool    isDelta;            ///< Set to true if the last shading point has delta distribution 
    bool    isSpecular;         ///< Set to true if the last shading point was below roughness threshold
    bool    isLightSampled;     ///< Does an NEE sample of the current path length exist?
    bool    applyBSDFSample; 
    float3  lastHitThp;
    bool    hasReconnectionHit;
    bool    terminated;         ///< Set to true when path is terminated.
    bool    pad4;
    bool    pad5;

    /** Initializes ray payload with default parameters.
    */
    __init(SampleGenerator sg)
    {
        this.prevLobeType = 0;
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0);
        this.brdf = float3(0);
        this.pdf = 1.0;
        this.thp = float3(1);
        this.reconnectionThp = float3(1);
        this.origin = float3(0);
        this.direction = float3(0);
        this.bsdfSamplePDF = 1.0;
        this.lastOrigin = float3(0);
        this.lastHitFaceN = float3(0);
        this.isDelta = true;
        this.isSpecular = true;
        this.isLightSampled = false;
        this.applyBSDFSample = true;
        this.hasReconnectionHit = false;
        this.lastHitRoughness = 0.0;
        this.lastHitThp = float3(1);
        this.lastCos = 0.0;
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
    uint    index;

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

    Ray getVisibilityRay() { return Ray(origin, toLight, 0.f, distance); }
};


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
bool generateEmissiveSample(const ShadingData sd, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls, uint2 pixel)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    TriangleLightSample tls = {};
    if (!gEmissiveLightSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;
    
    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;

    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = ls.getRayOrigin(sd.posW, sd.faceN, lightPos);

    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.toLight = toLight / ls.distance;

    return any(ls.Li > 0.0);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateVPLSample(const float3 posW, const float3 faceN, const bool hasDirectLight, inout SampleGenerator sg, out LightSample ls, uint2 pixel)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseVPLs) return false;

    vplSampler.VPLSample vpl = {};
    bool sampleDirectVPL = (!hasDirectLight || !kUseIndirectVPLs) && kUseDirectVPLs;

    if (!vplSampler.sampleLight(posW, sg, sampleDirectVPL, vpl, pixel)) return false;
    
    // Setup returned sample.
    ls.Li = vpl.pdf > 0.0 ? vpl.Li / vpl.pdf : float3(0);
    ls.pdf = vpl.pdf;
    ls.origin = vpl.posW;

    // Prepare shadow ray data.
    // Offset shading position to avoid self-intersection (light position is already offset).
    float3 rayOrigin = ls.getRayOrigin(posW, faceN, ls.origin);
    float3 toLight = ls.origin - rayOrigin;
    ls.distance = length(toLight);
    ls.toLight = toLight / ls.distance;
    ls.origin = rayOrigin;
    ls.index = vpl.index;

    return any(ls.Li > 0.0);
}

/** Generate a new scatter ray or terminate.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in] isCurveHit True if on curve hit.
    \param[in] rayOrigin Ray origin for the new ray.
    \param[in,out] rayData Ray payload.
    \return True if the path continues.
*/
void generateScatterRay(const ShadingData sd, const BSDFSample bsdfSample, bool isCurveHit, float3 rayOrigin, inout PathData pathData)
{
    pathData.origin = rayOrigin;
    if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
    {
        pathData.origin = sd.computeNewRayOrigin(false);
    }
    pathData.direction = bsdfSample.wo;
}

void updateReconnectionData(const ShadingData sd, const BSDFSample bsdfSample, const BSDFProperties miProperties, const RayData rayData, inout PathData pathData, inout prh.ReconnectionData rd)
{
    const bool isDelta = bsdfSample.isLobe(LobeType::Delta);

    if (!pathData.hasReconnectionHit && min(miProperties.roughness, pathData.lastHitRoughness) > gRoughnessThreshold && distance(pathData.lastOrigin, pathData.origin) > gMinConnectionDistance)
    {
        rd.reconnectionHit = rayData.hit;
        rd.pathLength = pathData.pathLength;
        rd.posW = pathData.origin;
        rd.w_o = bsdfSample.wo;
        float reconnectionDist = distance(pathData.lastOrigin, pathData.origin);
        rd.reconnectionJacobian = pathData.lastCos * pathData.bsdfSamplePDF * bsdfSample.pdf / (reconnectionDist * reconnectionDist);
        pathData.hasReconnectionHit = true;
    }

    if (pathData.hasReconnectionHit)
    {
        if (!isDelta)
            rd.pathPDF *= bsdfSample.pdf;
        pathData.reconnectionThp *= bsdfSample.weight;
    }
    pathData.lastHitRoughness = miProperties.roughness;
    pathData.lastOrigin = pathData.origin;
    pathData.lastHitThp = bsdfSample.weight;
}

void updatePathData(const ShadingData sd, const BSDFSample bsdfSample, const BSDFProperties miProperties, const RayData rayData, inout PathData pathData)
{
    const bool isDelta = bsdfSample.isLobe(LobeType::Delta);
    bool isLuminaireHit = gScene.materials.isCastShadow(rayData.materialID) == 0;
    pathData.isDelta &= isDelta; 
    pathData.applyBSDFSample &= (pathData.isDelta | isLuminaireHit); 
    pathData.thp *= bsdfSample.weight;
    if (!isDelta)
    {
        pathData.pdf *= bsdfSample.pdf; 
        pathData.bsdfSamplePDF = bsdfSample.pdf;
    }
    else
    {
        pathData.bsdfSamplePDF = 1.0;
    }
    pathData.lastHitFaceN = sd.getOrientedFaceNormal();
    pathData.terminated = !any(pathData.thp > 0.f);
    pathData.prevLobeType = bsdfSample.lobeType;
    pathData.lastCos = dot(sd.V, pathData.lastHitFaceN);
}

/** Process a miss.
    Query environment map.
    \param[in] hit Hit info.
    \param[in,out] rayData Ray payload.

*/
void handleMiss(inout PathData pathData)
{
    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || pathData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        pathData.radiance += pathData.thp * Le;
        pathData.terminated = true;
        return;
    }
}

/** Process a hit.
    Loads the shading data, samples analytic lights and samples a new scatter ray.
    Terminates the path if maximum number of bounces is reached.
    \param[in] hit Hit info.
    \param[in,out] rayData Ray payload.

*/
void handleHit(RayData rayData, inout PathData pathData, const bool hasDirectLight, inout prh.PathReservoir reservoir, inout prh.ReconnectionData rd, uint2 pixel)
{
    HitInfo hit = HitInfo(rayData.hit);
    const bool isCurveHit = hit.getType() == HitType::Curve;
    const bool isTriangleHit = hit.getType() == HitType::Triangle;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = {};
    if (!sdh.getShadingData(hit, -pathData.direction, sd))
    {
        pathData.terminated = true;
        return;
    }

    // Create material instance.
    uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let miProperties = mi.getProperties(sd);

    float3 selfEmission = miProperties.emission;

    // Add emitted light.
    //bool computeEmissive = kUseBSDFSamples && kUseEmissiveLights && (kUseMIS || !pathData.isLightSampled || pathData.isDelta);
    bool computeEmissive = kUseBSDFSamples && kUseEmissiveLights && pathData.applyBSDFSample && !hasDirectLight;
    if (any(selfEmission > 0) && computeEmissive)
    {
        if(kUseMIS && isTriangleHit && !pathData.isDelta && pathData.isLightSampled)
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
        float3 Lr = selfEmission * pathData.thp;
        pathData.radiance += Lr;
        float3 pathContribution = Lr * pathData.pdf;

        float3 reconnectionLr = selfEmission * pathData.reconnectionThp;
        rd.pathContribution = reconnectionLr; 
        prh.PathSample pathSample = prh.PathSample(pathContribution, luminance(Lr), 1.0, pathData.pathLength);
        const bool sampleAccpeted = reservoir.addSample(pathSample, 0, pathSample.w, pathData.sg);
        if(sampleAccpeted)
        {
            rd.bsdfPathLength = pathData.pathLength + 1;
            rd.resampleMode = 2;
            float dist = distance(pathData.origin, sd.posW);
            rd.jacobian = pathData.pathLength > 0 ? pathData.lastCos * pathData.bsdfSamplePDF / (dist * dist) : 1.0;
        }
    }

    // Check whether to terminate based on max depth.
    if (pathData.pathLength > kMaxBounces)
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

    //Sample BSDF for the next scatter ray and check for delta distribution
    BSDFSample bsdfSample;
    pathData.terminated = !mi.sample(sd, pathData.pathSG, bsdfSample, kUseImportanceSampling);

    // Determine if BSDF has non-delta lobes.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

    // Check if we should apply VPLs.
    const bool applyVPLs = kUseVPLs && hasNonDeltaLobes;
    pathData.isLightSampled = false;

    //Add contribution of direct light from VPLs.
    if (applyVPLs)
    {
        LightSample ls = {};
        bool validSample = generateVPLSample(rayOrigin, sd.getOrientedFaceNormal(), hasDirectLight, pathData.sg, ls, pixel);
        if (validSample)
        {
            float3 brdf = max(mi.eval(sd, ls.toLight, pathData.sg), 0.0);
            float3 Lr = pathData.thp * brdf * ls.Li;
            // Apply MIS weight
            //if (kUseMIS)
            //{
            //    float3 toLight = normalize(pathData.neeSamplePosW - sd.posW);
            //    float bsdfPDF = mi.evalPdf(sd, lightSample.direction, kUseImportanceSampling);
            //    Lr *= evalMIS(pathData.neeSamplePDF, bsdfPDF);
            //}
            if (any(Lr > 0.0))
            {
                bool isVisible = sdh.dispatchShadowRay(ls.getVisibilityRay());
                if (isVisible)
                {
                    pathData.radiance += Lr;
                    float pathPDF = pathData.pdf * ls.pdf;
                    float3 contribution = Lr * pathPDF;
                    prh.PathSample pathSample = prh.PathSample(contribution, luminance(Lr), pathPDF, pathData.pathLength);
                    const bool sampleAccepted = reservoir.addSample(pathSample, 0, pathSample.w ,pathData.sg);
                    if (sampleAccepted)
                    {
                        rd.pathPDF *= ls.pdf;
                        float3 reconnectionLr = ls.Li * brdf * pathData.reconnectionThp;
                        rd.pathContribution = reconnectionLr;
                        vplSampler.DirectVPL vpl = vplSampler.gDirectVPLBuffer[ls.index];
                        vpl.pdf = ls.pdf;
                        gNEESampleBuffer[prh.getIndex(pixel)] = vpl;
                        rd.neePathLength = pathData.pathLength + 1;
                        if (!pathData.hasReconnectionHit)
                        {
                            rd.resampleMode = 0;
                            rd.jacobian = dot(sd.V, sd.getOrientedFaceNormal()) * max(mi.evalPdf(sd, ls.toLight, true), 0.0) / (ls.distance * ls.distance);
                        }
                        else
                            rd.resampleMode = 1;
                    }
                }
            }
        }
    }

    //Add contribution of direct light from NEE
    //const bool applyNEE = kUseNEE && hasNonDeltaLobes;
    //if(applyNEE && kUseEmissiveLights)
    //{
    //    LightSample ls = {};
    //    if(generateEmissiveSample(sd, true, pathData.sg, ls, pixel))
    //    {
    //        pathData.isLightSampled = true;
    //        // Apply MIS weight.
    //        if (kUseMIS)
    //        {
    //            float scatterPdf = mi.evalPdf(sd, ls.toLight, kUseImportanceSampling);
    //            ls.Li *= evalMIS(ls.pdf, scatterPdf);
    //        }
    //        float3 brdf = mi.eval(sd, ls.toLight, pathData.sg);
    //        float3 Lr = brdf * ls.Li;
    //        if (any(Lr > 0.0))
    //        {
    //            //don't apply shadow ray epsilon, as end points have been adjusted already
    //            bool isVisible = sdh.dispatchShadowRay(ls.getVisibilityRay()); 
    //            if(isVisible)
    //            {
    //                pathData.radiance += pathData.thp * Lr;
    //            }
    //        }
    //    }
    //}

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.

    pathData.lastDirection = pathData.direction;
    generateScatterRay(sd, bsdfSample, isCurveHit, rayOrigin, pathData);
    updateReconnectionData(sd, bsdfSample, miProperties, rayData, pathData, rd);
    updatePathData(sd, bsdfSample, miProperties, rayData, pathData);

    prh.PathDebugData.Node node = {};
    node.posW = pathData.origin;
    node.hasReconnectionHit = pathData.hasReconnectionHit;
    node.thp = float4(pathData.thp, 1);
    node.reconnectionPosW = float4(pathData.reconnectionThp, 1);
    gPathDebugBuffer[prh.getIndex(pixel)].path[pathData.pathLength] = node;

    pathData.pathLength++;
}

float3 getReprojection(inout prh.CausticReservoir reservoir, SampleGenerator sg, uint2 pixel)
{
    uint2 headTextureIndex = DispatchRaysIndex().xy; 
    uint linkedListIndex = gHeadCounter[headTextureIndex];
    float3 flux = float3(0.0); 
    while (linkedListIndex != uint(-1)) {
        ListNode currentNode = gLinkedList[linkedListIndex];    
        flux += currentNode.luminance;
        float3 pathContribution = currentNode.luminance * currentNode.pathPDF;
        prh.CausticSample pathSample = prh.CausticSample(pathContribution, luminance(currentNode.luminance), currentNode.pathPDF, currentNode.pathLength);
        reservoir.addSample(pathSample, 0, pathSample.w, sg);
        linkedListIndex = currentNode.next;
    }
    return flux;
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
void tracePath(const uint2 pixel, const bool hasDirectLight, inout prh.PathReservoir reservoir, inout SampleGenerator sg)
{
    PathData pathData = PathData(sg);
    pathData.pathSG = SampleGenerator(pixel, gFrameCount);
    prh.ReconnectionData rd = prh.ReconnectionData();
    prh.PathDebugData.Node node = {};
    gPathDebugBuffer[prh.getIndex(pixel)].path = {node, node, node};
    rd.seed = uint4(pixel, gFrameCount, 0);

    pathData.direction = -sdh.getRayDir(pixel);
    RayData rayData = RayData();

    rayData.hit = sdh.getPackedHitInfo(pixel);
    TriangleHit triangleHit = TriangleHit(rayData.hit);
    rayData.materialID = gScene.getMaterialID(triangleHit.instanceID);
    handleHit(rayData, pathData, hasDirectLight, reservoir, rd, pixel);

    // Follow path into the scene and compute its total contribution.
    for (uint depth = 1; depth <= kMaxBounces && !pathData.terminated; depth++)
    {
        traceScatterRay(rayData, pathData);
        handleHit(rayData, pathData, hasDirectLight, reservoir, rd, pixel);
    }
    gReconnectionData[prh.getIndex(pixel)] = rd;
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
    const uint index = prh.getIndex(pixel);

    gOutDebug[pixel] = float4(0);

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    bool hasDirectLight = false;

    //prepare caustic reservoirs
    prh.CausticReservoir causticReservoir = prh.CausticReservoir();
    gCausticReservoir[index] = causticReservoir;

    //add light paths to reservoirs
    float3 reprojection = getReprojection(causticReservoir, sg, pixel);
    hasDirectLight = any(reprojection > 0.0);

    //finalize and write reservoir
    causticReservoir.finalizeSample();
    causticReservoir.confidenceWeight = 1;
    gCausticReservoir[index] = causticReservoir;

    //prepare path reservoirs
    prh.PathReservoir reservoir = prh.PathReservoir();
    gPathReservoir[index] = reservoir;

    vplSampler.DirectVPL vpl = vplSampler.DirectVPL();
    gNEESampleBuffer[prh.getIndex(pixel)] = vpl;

    //add camera paths to reservoirs
    tracePath(pixel, hasDirectLight, reservoir, sg);

    //finalize and write reservoirs
    reservoir.finalizeSample();
    reservoir.confidenceWeight = 1;
    gPathReservoir[index] = reservoir;
}
