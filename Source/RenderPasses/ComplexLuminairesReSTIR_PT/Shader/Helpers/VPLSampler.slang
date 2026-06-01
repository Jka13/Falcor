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
        float3 flux;
        float3 posW;        //current world pos
        float3 origin;      //previous world pos
        float3 faceN;       //face normal of the current surface
        float3 originFaceN; //face noraml of the previous surface
        float3 w_i;         //incoming direction
        float3 w_o;         //outgoing direction

        __init()
        {
            flux = float3(0);
            posW = float3(0);
            origin = float3(0);
            faceN = float3(0);
            originFaceN = float3(0);
            w_i = float3(0);
            w_o = float3(0);
        }
    }

    struct LightSample
    {
        float3 flux;
        float3 posW;
        float3 faceN;

        __init()
        {
            posW = float3(0);
            flux = float3(0);
            faceN = float3(0);
        }

        __init(Photon p)
        {
            flux = p.flux;
            posW = p.posW;
            faceN = p.faceN;
        }
    }

    struct DirectVPL
    {
        float3  flux;
        float3  posW;
        float3  direction;
        float   cosOpeningAngle;

        __init()
        {
            posW = float3(0);
            direction = float3(0);
            flux = float3(0);
            cosOpeningAngle = 0.0;
        }

        __init(Photon p)
        {
            flux = p.flux;
            posW = p.origin;
            direction = p.w_i;
            cosOpeningAngle = 0.0;
        }
    }

    struct IndirectVPL
    {
        PackedHitInfo hit;
        float3 flux;
        float3 direction;
        float3 origin;

        __init()
        {
            HitInfo hitInfo = {};
            hit = hitInfo.getData();
            flux = float3(0);
            direction = float3(0);
            origin = float3(0);
        }

        __init(Photon p)
        {
            HitInfo hitInfo = {};
            hit = hitInfo.getData();
            flux = p.flux;
            direction = p.w_i;
            origin = p.posW;
        }
    }

    RWTexture2D<float4> gOutDebug;

    StructuredBuffer<DirectVPL> gDirectVPLBuffer;
    StructuredBuffer<IndirectVPL> gIndirectVPLBuffer;
    Buffer<uint> gPhotonCounter;
    Buffer<uint> gDirectVPLCounter;
    Buffer<uint> gIndirectVPLCounter;

    struct VPLSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  posW;       ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    };

    bool sampleLight(const float3 shadingPosW, inout SampleGenerator sg, const bool sampleDirectVPL, out VPLSample vplSample, uint2 pixel)
    {
        vplSample = {};
        uint luminaireSampleCount = gPhotonCounter[0];

        float uLight = sampleNext1D(sg);
        uint idx = min((uint)(uLight * luminaireSampleCount), luminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to luminaireSampleCount even if uLight < 1.0 when luminaireSampleCount is large.
        float vplSelectionPDF = 1.f / (float) luminaireSampleCount;

        if (sampleDirectVPL)
        {
            //gOutDebug[pixel] += float4(0.0, 1.0, 0.0, 0.0);
            DirectVPL vpl = gDirectVPLBuffer[idx];

            float3 toLight = vpl.posW - shadingPosW;
            float distSqr = dot(toLight, toLight);
            float distance = sqrt(distSqr);
            toLight /= distance;

            float cosTheta = dot(-toLight, vpl.direction);
            if (cosTheta < vpl.cosOpeningAngle)
            {
                return false;
            }

            float rSqr = gPointLightRadius * gPointLightRadius;
            float attenuation = 2 / (distSqr + rSqr + distance * sqrt(distSqr + rSqr)); //Cem Yuksel point light attenuation formula to avoid infinite attenuation terms for small distances
            float3 normalizationFactor = min(FLT_MAX, 1 / (M_2PI * (1 - vpl.cosOpeningAngle))); //normalization by the cone area to only spread the flux of the photon over the integral of the spotlight

            float3 Li = vpl.flux / gPhotonCount; //deferred division through photon count from sample generation
            vplSample.posW = vpl.posW;
            vplSample.pdf = vplSelectionPDF;
            vplSample.Li = normalizationFactor * Li / distSqr;
            return true;
        }
        else 
        {
            //gOutDebug[pixel] += float4(0.0, 0.0, 1.0, 0.0);
            IndirectVPL vpl = gIndirectVPLBuffer[idx];
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
            float attenuation = 2 / (distSqr + rSqr + distance * sqrt(distSqr + rSqr)); //Cem Yuksel point light attenuation formula to avoid infinite attenuation terms for small distances

            vplSample.posW = vpl.origin;
            vplSample.pdf = vplSelectionPDF;
            float3 Li = vpl.flux / (float) gPhotonCount; //deferred division through photon count from sample generation
            vplSample.Li = Li * brdf * attenuation;
            return true;
        }
        
        //{
        //    Photon vpl = gPhotonBuffer[idx - brdfVPLCount];

        //    float3 toLight = vpl.posW - shadingPosW;
        //    float distSqr = dot(toLight, toLight);
        //    float distance = sqrt(distSqr);
        //    toLight /= distance;

        //    float cosTheta = dot(-toLight, vpl.faceN);
        //    bool backFacing = cosTheta <= 0.0;
        //    if (backFacing) return false;

        //    vplSample.posW = vpl.posW;
        //    float3 Li = vpl.flux / (float) gPhotonCount; //deferred division through photon count from sample generation
        //    vplSample.pdf = vplSelectionPDF;
        //    vplSample.Li = Li * cosTheta / distSqr;
        //    return true;
        //}
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
