#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

import Scene.RaytracingInline;

struct PathReservoirHelper
{
    cbuffer CB
    {
        uint2 gFrameDim;
    }

    struct CausticSample
    {
        float3  pathContribution;
        float   targetFunction;
        float   w;
        float   pathPDF;
        uint    pathLength;

        __init()
        {
            pathContribution = float3(0);
            targetFunction = 0;
            w = 0.0;
            pathPDF = 0.0;
            pathLength = 0;
        }

        __init(float3 contribution, float w, float pdf, uint length)
        {
            pathContribution = contribution;
            targetFunction = luminance(contribution);
            this.w = w;
            pathPDF = pdf;
            pathLength = length;
        }
    }

    struct ReconnectionData
    {
        float3          pathContribution;   ///< path contribution from the light sample to the reconnection hit (not including the hit)
        float3          posW;               ///< position of the reconnection hit
        float3          w_o;                ///< outgoing direction from the reconnection hit
        float           pathPDF;            ///< probability of the path after the reconnection hit (not including the pdf of the sample at the hit)
        uint            pathLength;         ///< number of bounces until the reconnection hit (not including the hit)
        PackedHitInfo   reconnectionHit;    ///< first hit from camera satisfying reconnection condintions (> roughness threshold, > min connection distance)
        uint            seed;               ///< seed for a distinct sg generating the bsdf samples

        __init()
        {
            pathContribution = float3(0);
            posW = float3(0);
            w_o = float3(0);
            pathPDF = 1.0;
            pathLength = 0;
            HitInfo hit = {};
            reconnectionHit = hit.getData();
            seed = 0;
        }
    }

    struct PathSample
    {
        float3          pathContribution;
        float           targetFunction;
        float           w;
        float           pathPDF;
        uint            pathLength;
        ReconnectionData rd;

        __init()
        {
            pathContribution = float3(0);
            targetFunction = 0;
            w = 0.0;
            pathPDF = 0.0;
            pathLength = 0;
            rd = ReconnectionData();
        }

        __init(float3 contribution, float w, float pdf, uint length, ReconnectionData rd)
        {
            pathContribution = contribution;
            targetFunction = luminance(contribution);
            this.w = w;
            pathPDF = pdf;
            pathLength = length;
            this.rd = rd;
        }
    }

    struct CausticReservoir
    {
        CausticSample sample;
        float w_sum;
        uint confidenceWeight;

        [mutating] bool addSample(CausticSample sample, uint c, float w, inout SampleGenerator sg)
        {
            confidenceWeight += c;
            w_sum += w;
            float rnd = sampleNext1D(sg);
            bool sampleAccepted = w_sum * rnd < w;
            if (sampleAccepted)
            {
                this.sample = sample;
            }
            return sampleAccepted;
        }

        [mutating] void finalizeSample()
        {
            confidenceWeight = clamp(confidenceWeight, 1, 20);
            if (sample.targetFunction <= 0 || isnan(sample.targetFunction))
            {
                w_sum = 0.0;
                return;
            }
            w_sum /= sample.targetFunction;
            w_sum = isinf(w_sum) ? 0.0 : w_sum;
        }

        [mutating] void updateVisibility(bool isVisible)
        {
            w_sum = isVisible ? w_sum : 0;
        }

        __init()
        {
            sample = CausticSample();
            w_sum = 0.0;
            confidenceWeight = 0;
        }
    }

    struct PathReservoir
    {
        PathSample sample;
        float w_sum;
        uint confidenceWeight;

        [mutating] bool addSample(PathSample sample, uint c, float w, inout SampleGenerator sg)
        {
            confidenceWeight += c;
            w_sum += w;
            float rnd = sampleNext1D(sg);
            bool sampleAccepted = w_sum * rnd < w;
            if (sampleAccepted)
            {
                this.sample = sample;
            }
            return sampleAccepted;
        }

        [mutating] void finalizeSample()
        {
            confidenceWeight = clamp(confidenceWeight, 1, 20);
            if (sample.targetFunction <= 0 || isnan(sample.targetFunction))
            {
                w_sum = 0.0;
                return;
            }
            w_sum /= sample.targetFunction;
            w_sum = isinf(w_sum) ? 0.0 : w_sum;
        }

        [mutating] void updateVisibility(bool isVisible)
        {
            w_sum = isVisible ? w_sum : 0;
        }

        __init()
        {
            sample = PathSample();
            w_sum = 0.0;
            confidenceWeight = 0;
        }
    }

    uint getIndex(uint2 pixel)
    {
        return pixel.x + gFrameDim.x * pixel.y;
    }

} prh;
