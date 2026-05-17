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
import Helpers.SceneDataHandler;


struct VPLSampler
{
    cbuffer SampleBuffer
    {
        float gCosOpeningAngle;
        float gPenumbraAngle;
        uint gPhotonCount;
        float gPointLightRadius;
    }

    struct Photon
    {
        float3  posW;
        float3  direction;
        float3  flux;
        float3  faceN;
        float3  lightSamplePosW;
        float   lightSampleSelectionPDF;
        float   cosOpeningAngle;

        __init()
        {
            posW = float3(0);
            direction = float3(0);
            flux = float3(0);
            faceN = float3(0);
            lightSamplePosW = float3(0);
            lightSampleSelectionPDF = 0.0;
            cosOpeningAngle = 0.0;
        }
    }

    struct VPL
    {
        PackedHitInfo hit;
        float3 flux;
        float3 direction;
        float3 origin;

        __init()
        {
            HitInfo hitInfo = {};
            hit = hitInfo.getData();
            flux = float3(0.0);
        }
    }
    RWTexture2D<float4> gOutDebug;

    StructuredBuffer<Photon> gPhotonBuffer;
    StructuredBuffer<VPL> gVPLBuffer;
    Buffer<uint> gPhotonCounter;
    Buffer<uint> gDirectVPLCounter;
    Buffer<uint> gBRDFVPLCounter;

    struct VPLSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  posW;       ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    };

    //Photon samplePhotonsUniform(inout SampleGenerator sg)
    //{
    //    float uLight = sampleNext1D(sg);
    //    uint idx = min((uint)(uLight * luminaireSampleCount), luminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to luminaireSampleCount even if uLight < 1.0 when luminaireSampleCount is large.
    //    Photon vpl = gPhotonBuffer[idx];

    //    float vplSelectionPDF = 1.f / (float)luminaireSampleCount;
    //    vpl.lightSampleSelectionPDF *= vplSelectionPDF;
    //    return vpl; 
    //}

    //VPL sampleVPLsUniformly(inout SampleGenerator sg)
    //{
    //    float uLight = sampleNext1D(sg);
    //    uint idx = min((uint)(uLight * luminaireSampleCount), luminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to luminaireSampleCount even if uLight < 1.0 when luminaireSampleCount is large.
    //    VPL vpl = gVPLBuffer[idx];

    //    float vplSelectionPDF = 1.f / (float)luminaireSampleCount;
    //    //vpl.lightSampleSelectionPDF *= vplSelectionPDF;
    //    return vpl; 
    //}

    bool sampleLight(const float3 shadingPosW, inout SampleGenerator sg, out VPLSample vplSample, uint2 pixel)
    {
        vplSample = {};
        uint luminaireSampleCount = gPhotonCounter[0];

        float uLight = sampleNext1D(sg);
        uint idx = min((uint)(uLight * luminaireSampleCount), luminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to luminaireSampleCount even if uLight < 1.0 when luminaireSampleCount is large.
        float vplSelectionPDF = 1.f / (float) luminaireSampleCount;

        uint brdfVPLCount = gBRDFVPLCounter[0];

        if (idx < brdfVPLCount)
        {
            VPL vpl = gVPLBuffer[idx];
            HitInfo hit = HitInfo(vpl.hit); 
            ShadingData sd = {};
            if (!sdh.getShadingData(hit, -vpl.direction, sd)) return false;

            float3 toLight = vpl.origin - shadingPosW;
            float distSqr = dot(toLight, toLight);
            float distance = sqrt(distSqr);
            toLight /= distance;

            float cosTheta = dot(-toLight, sd.getOrientedFaceNormal());
            bool backFacing = cosTheta <= 0.0;
            if (backFacing) return false;

            let lod = ExplicitLodTextureSampler(0);
            uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);

            float3 brdf = max(mi.eval(sd, -toLight, sg), 0.0);
            float rSqr = gPointLightRadius * gPointLightRadius;
            float attenuation = 2 / (distSqr + rSqr + distance * sqrt(distSqr + rSqr));

            vplSample.posW = vpl.origin;
            vplSample.pdf = vplSelectionPDF;
            float3 Li = vpl.flux / (float) gPhotonCount; //deferred division through photon count from sample generation
            vplSample.Li = Li * brdf * attenuation;
            return true;
        }
        else 
        {
            Photon vpl = gPhotonBuffer[idx - brdfVPLCount];

            float3 toLight = vpl.posW - shadingPosW;
            float distSqr = dot(toLight, toLight);
            float distance = sqrt(distSqr);
            toLight /= distance;

            float cosTheta = dot(-toLight, vpl.faceN);
            bool backFacing = cosTheta <= 0.0;
            if (backFacing) return false;

            vplSample.posW = vpl.posW;
            float3 Li = vpl.flux / (float) gPhotonCount; //deferred division through photon count from sample generation
            vplSample.pdf = vplSelectionPDF;
            vplSample.Li = Li * cosTheta / distSqr;
            return true;
        }
    }

    //float3 samplePointLight(const float3 shadingPosW, const Photon ls)
    //{
    //    float kMinLightDistSqr = 1e-9f;
    //    float3 toLight = ls.posW - shadingPosW;
    //    float distSqr = dot(toLight, toLight);
    //    distSqr = max(distSqr, kMinLightDistSqr); 
    //    float distance = sqrt(distSqr);
    //    toLight /= distance;

    //    float cosTheta = dot(-toLight, ls.direction);
    //    float falloff = 1.f;
    //    float openingAngle = acos(ls.cosOpeningAngle); //light property
    //    if (cosTheta < ls.cosOpeningAngle)
    //    {
    //        falloff = 0.f;
    //    }
    //    else if (gPenumbraAngle > 0.f)
    //    {
    //        float deltaAngle = openingAngle - acos(cosTheta);
    //        falloff = smoothstep(0.f, gPenumbraAngle, deltaAngle);
    //    }
    //    float3 normalizationFactor = min(FLT_MAX, 1 / (M_2PI * (1 - ls.cosOpeningAngle)));
    //    float samplePDF = float(luminaireSampleCount) / float(gPhotonCount); //deferred division through photon count from sample generation
    //    // Compute incident radiance at shading point.
    //    return normalizationFactor * samplePDF * ls.flux * falloff / distSqr;
    //}
} vplSampler;
