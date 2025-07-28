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

        __init(float3 fluxIn, float3 posWIn, float p_hatIn)
        {
            flux = fluxIn;
            posW = posWIn;
            p_hat = p_hatIn;
        }
    }

    struct Reservoir
    {
        Sample sample;
        float w_sum;
        uint confidenceWeight;

        [mutating] void addSample(Sample sample, float w, uint c, inout SampleGenerator sg)
        {
            this.confidenceWeight += c;
            float w_sum = this.w_sum + w;
            float rnd = sampleNext1D(sg);
            if (w_sum * rnd < w)
            {
                this.sample = sample;
            }
            this.w_sum = w_sum;
        }

        [mutating] void finalizeSample()
        {
            if (this.sample.p_hat == 0)
            {
                this.w_sum = 0;
                return;
            }
            float W = this.w_sum / this.sample.p_hat;
            this.w_sum = W;
            this.confidenceWeight = min(this.confidenceWeight , 20);
        }

        __init()
        {
            sample = Sample();
            w_sum = 0;
            confidenceWeight = 0;
        }
    }

    uint getIndex(uint2 pixel)
    {
        return pixel.x + gFrameDim.x * pixel.y;
    }
} rh;
