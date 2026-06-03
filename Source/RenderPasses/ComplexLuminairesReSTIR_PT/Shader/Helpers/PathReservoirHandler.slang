#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

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

    struct PathSample
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

    struct Reservoir
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
