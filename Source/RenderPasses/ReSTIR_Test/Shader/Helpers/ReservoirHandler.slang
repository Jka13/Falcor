#include "Utils/Math/MathConstants.slangh"

import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

struct ReservoirHelper
{
    cbuffer CB
    {
        uint2 gFrameDim;
    }

    struct Sample
    {
        float3 flux;
        float targetFunction;
        float3 posW;

        __init()
        {
            flux = float3(0);
            targetFunction = 0;
            posW = float3(0);
        }

        __init(float3 fluxIn, float3 posWIn, float targetFunctionIn)
        {
            flux = fluxIn;
            posW = posWIn;
            targetFunction = targetFunctionIn;
        }
    }

    struct Reservoir
    {
        Sample sample;
        float w_sum;
        float3 pad;
        uint confidenceWeight;

        [mutating] bool addSample(Sample sample, float w, uint c, inout SampleGenerator sg)
        {
            this.confidenceWeight += c;
            w_sum = w_sum + w;
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
            confidenceWeight = min(confidenceWeight, 20);
            if (sample.targetFunction == 0)
            {
                w_sum = 0;
                return;
            }
            w_sum /= sample.targetFunction;
        }

        [mutating] void updateVisibility(bool isVisible)
        {
            w_sum = isVisible ? w_sum : 0;
        }

        __init()
        {
            sample = Sample();
            w_sum = 0;
            confidenceWeight = 0;
            pad = float3(0);
        }
    }

    uint getIndex(uint2 pixel)
    {
        return pixel.x + gFrameDim.x * pixel.y;
    }
} rh;
