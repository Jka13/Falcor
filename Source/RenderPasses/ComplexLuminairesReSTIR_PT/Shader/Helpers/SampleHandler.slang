#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

struct SampleHandler
{
    cbuffer SampleBuffer
    {
        float gCosOpeningAngle;
        float gPenumbraAngle;
        uint gPhotonCount;
    }

    float3 samplePointLight(const float3 shadingPosW, const float3 samplePosW, const float3 coneDirection, const float3 flux)
    {
        float kMinLightDistSqr = 1e-9f;
        float3 toLight = samplePosW - shadingPosW;
        float distSqr = dot(toLight, toLight);
        distSqr = max(distSqr, kMinLightDistSqr); 
        float distance = sqrt(distSqr);
        float3 dir = toLight / distance;

        float cosTheta = -dot(dir, coneDirection);
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
        float3 normalizationFactor = 1 / (M_2PI * (1 - gCosOpeningAngle));
        // Compute incident radiance at shading point.
        return normalizationFactor * flux * falloff / distSqr;
    }
} sh;
