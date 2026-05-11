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
        uint gLuminaireSampleCount;
        uint gPhotonCount;
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

    StructuredBuffer<Photon> gPhotonBuffer;
    StructuredBuffer<VPL> gVPLBuffer;

    struct VPLSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  posW;       ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    };

    Photon samplePhotonsUniform(inout SampleGenerator sg)
    {
        float uLight = sampleNext1D(sg);
        uint idx = min((uint)(uLight * gLuminaireSampleCount), gLuminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to gLuminaireSampleCount even if uLight < 1.0 when gLuminaireSampleCount is large.
        Photon vpl = gPhotonBuffer[idx];

        float vplSelectionPDF = 1.f / (float)gLuminaireSampleCount;
        vpl.lightSampleSelectionPDF *= vplSelectionPDF;
        return vpl; 
    }

    VPL sampleVPLsUniformly(inout SampleGenerator sg)
    {
        float uLight = sampleNext1D(sg);
        uint idx = min((uint)(uLight * gLuminaireSampleCount), gLuminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to gLuminaireSampleCount even if uLight < 1.0 when gLuminaireSampleCount is large.
        VPL vpl = gVPLBuffer[idx];

        float vplSelectionPDF = 1.f / (float)gLuminaireSampleCount;
        //vpl.lightSampleSelectionPDF *= vplSelectionPDF;
        return vpl; 
    }

    bool sampleLight(const float3 shadingPosW, inout SampleGenerator sg, out VPLSample vplSample) //direct light
    {
        vplSample = {};
        float kMinLightDistSqr = 1e-9f;

        //sample vpls uniformly
        float uLight = sampleNext1D(sg);
        // Safety precaution as the result of the multiplication may be rounded to gLuminaireSampleCount even if uLight < 1.0 when gLuminaireSampleCount is large
        uint idx = min((uint)(uLight * gLuminaireSampleCount), gLuminaireSampleCount - 1); 
        Photon vpl = gPhotonBuffer[idx];

        float3 toLight = vpl.posW - shadingPosW;
        float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
        float distance = sqrt(distSqr);
        toLight /= distance;

        float cosTheta = dot(-toLight, vpl.faceN);
        bool backFacing = cosTheta <= 0.0;
        if (backFacing) return false;

        vplSample.posW = vpl.posW;
        float3 Li = vpl.flux / (float)gPhotonCount; //deferred division through photon count from sample generation
        vplSample.pdf = 1.f / (float)gLuminaireSampleCount;
        // Compute incident radiance at shading point.
        vplSample.Li = Li * cosTheta / distSqr;
        return true;
    }

    float3 sampleBRDFVPL(const float3 toLight, inout VPL lp, inout SampleGenerator sg)
    {
        float kMinLightDistSqr = 1e-9f;
        let lod = ExplicitLodTextureSampler(0);

        HitInfo hit = HitInfo(lp.hit); 
        ShadingData sd = {};
        if (!sdh.getShadingData(hit, lp.direction, sd))
            return float3(0);

        uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        float3 brdf = max(mi.eval(sd, toLight, sg), 0);
        float pdf = max(mi.evalPdf(sd, toLight, true), 0);
        float3 thp = pdf > 0.0 ? brdf / pdf : 0.0;

        float samplePDF = float(gLuminaireSampleCount) / float(gPhotonCount); //deferred division through photon count from sample generation
        float cos = dot(-toLight, sd.faceN);
        return cos > 0.0 ? samplePDF * lp.flux * thp : 0.0;
    }

    float3 samplePointLight(const float3 shadingPosW, const Photon ls)
    {
        float kMinLightDistSqr = 1e-9f;
        float3 toLight = ls.posW - shadingPosW;
        float distSqr = dot(toLight, toLight);
        distSqr = max(distSqr, kMinLightDistSqr); 
        float distance = sqrt(distSqr);
        toLight /= distance;

        float cosTheta = dot(-toLight, ls.direction);
        float falloff = 1.f;
        float openingAngle = acos(ls.cosOpeningAngle); //light property
        if (cosTheta < ls.cosOpeningAngle)
        {
            falloff = 0.f;
        }
        else if (gPenumbraAngle > 0.f)
        {
            float deltaAngle = openingAngle - acos(cosTheta);
            falloff = smoothstep(0.f, gPenumbraAngle, deltaAngle);
        }
        float3 normalizationFactor = min(FLT_MAX, 1 / (M_2PI * (1 - ls.cosOpeningAngle)));
        float samplePDF = float(gLuminaireSampleCount) / float(gPhotonCount); //deferred division through photon count from sample generation
        // Compute incident radiance at shading point.
        return normalizationFactor * samplePDF * ls.flux * falloff / distSqr;
    }
} vplSampler;
