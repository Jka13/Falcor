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
        float3 posW;
        float p_hat;

        __init()
        {
            flux = float3(0);
            posW = float3(0);
            p_hat = 0;
        }
    }

    struct Reservoir
    {
        Sample sample;
        float w_sum;

        [mutating] void addSample(Sample sample, float w, inout SampleGenerator sg)
        {
            float w_sum = this.w_sum + w;
            float rnd = sampleNext1D(sg);
            bool accepted = false;
            if (w_sum * rnd < w)
            {
                this.sample = sample;
            }
            this.w_sum = w_sum;
        }

        [mutating] void finalizeSample()
        {
            if (sample.p_hat == 0)
            {
                w_sum = 0;
                return;
            }
            float W = w_sum / sample.p_hat;
            w_sum = W;
        }

        __init()
        {
            sample = Sample();
            w_sum = 0;
        }
    }

    uint getIndex(uint2 pixel)
    {
        return pixel.x + gFrameDim.x * pixel.y;
    }
} rh;
