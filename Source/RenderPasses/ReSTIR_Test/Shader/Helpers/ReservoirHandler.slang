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

        __init()
        {
            flux = float3(0);
            posW = float3(0);
        }
    }

    struct Reservoir
    {
        Sample sample;
        float w_sum;

        [mutating] bool addSample(Sample sample, float w, inout SampleGenerator sg)
        {
            float w_sum = this.w_sum + w;
            float rnd = sampleNext1D(sg);
            bool accepted = false;
            if (w_sum * rnd < w)
            {
                this.sample = sample;
                accepted = true;
            }
            this.w_sum = w_sum;
            return accepted;
        }

        [mutating] void finalizeSample(float p_hat)
        {
            if (p_hat == 0)
            {
                w_sum = 0;
                return;
            }
            float W = w_sum / p_hat;
            w_sum = W;
        }

        __init()
        {
            sample = Sample();
            w_sum = 0;
        }
    }

    RWStructuredBuffer<Reservoir> gReservoir0;
    RWStructuredBuffer<Reservoir> gReservoir1;

    uint getIndex(uint2 pixel)
    {
        return pixel.x + gFrameDim.x * pixel.y;
    }

    void setReservoir(uint2 pixel, Reservoir reservoir)
    {
        uint index = getIndex(pixel);
        gReservoir0[index] = reservoir;
    }

    Reservoir getReservoir(uint2 pixel)
    {
        uint index = getIndex(pixel);
        return gReservoir0[index];
    }
} rh;
