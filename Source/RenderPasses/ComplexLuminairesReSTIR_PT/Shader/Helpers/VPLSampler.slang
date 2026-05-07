#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

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

    StructuredBuffer<Photon> gPhotonBuffer;

    Photon sampleVPLsUniform(inout SampleGenerator sg)
    {
        float uLight = sampleNext1D(sg);
        uint idx = min((uint)(uLight * gLuminaireSampleCount), gLuminaireSampleCount - 1); // Safety precaution as the result of the multiplication may be rounded to gLuminaireSampleCount even if uLight < 1.0 when gLuminaireSampleCount is large.
        Photon vpl = gPhotonBuffer[idx];

        float vplSelectionPDF = 1.f / (float)gLuminaireSampleCount;
        vpl.lightSampleSelectionPDF *= vplSelectionPDF;
        return vpl; 
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
