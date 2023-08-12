#ifndef RESERVOIR_GLSL
#define RESERVOIR_GLSL

#include "host_device.h"

float resvToScalar(vec3 x) {
    return length(x);
}

void resvReset(inout DirectReservoir resv) {
    resv.num = 0;
    resv.weight = 0;
}

void resvReset(inout IndirectReservoir resv) {
    resv.num = 0;
    resv.weight = 0;
    resv.bias = 0;
}

void resvUpdateBias(inout IndirectReservoir resv, float rnd) {
    resv.bias = resv.weight / (float(resv.num) * rnd);
}

bool resvInvalid(DirectReservoir resv) {
    return isnan(resv.weight) || resv.weight < 0.0;
}

bool resvInvalid(IndirectReservoir resv) {
    return isnan(resv.weight) || resv.weight < 0.0;
}

void resvCheckValidity(inout DirectReservoir resv) {
    if (resvInvalid(resv)) {
        resvReset(resv);
    }
}

void resvCheckValidity(inout IndirectReservoir resv) {
    if (resvInvalid(resv)) {
        resvReset(resv);
    }
}

void resvUpdate(inout DirectReservoir resv, LightSample newSample, float newWeight, float r) {
    resv.weight += newWeight;
    resv.num += 1;
    if (r * resv.weight < newWeight) {
        resv.lightSample = newSample;
    }
}

void resvUpdate(inout IndirectReservoir resv, GISample newSample, float newWeight, float r) {
    resv.weight += newWeight;
    resv.num += 1;
    if (r * resv.weight < newWeight) {
        resv.giSample = newSample;
    }
}

void resvMerge(inout IndirectReservoir resv, IndirectReservoir rhs, float rnd, float r) {
    uint num = resv.num;
    resvUpdate(resv, rhs.giSample, rnd * rhs.bias * float(rhs.num), r);
    resv.num = num + rhs.num;
}

void resvMerge(inout DirectReservoir resv, DirectReservoir rhs, float r) {
    resv.weight += rhs.weight;
    resv.num += rhs.num;
    if (r * resv.weight < rhs.weight) {
        resv.lightSample = rhs.lightSample;
    }
}

void resvMerge(inout IndirectReservoir resv, IndirectReservoir rhs, float r) {
    resv.weight += rhs.weight;
    resv.num += rhs.num;
    if (r * resv.weight < rhs.weight) {
        resv.giSample = rhs.giSample;
    }
}

void resvClamp(inout DirectReservoir resv, int clamp) {
    if (resv.num > clamp) {
        resv.weight *= float(clamp) / float(resv.num);
        resv.num = clamp;
    }
}

void resvClamp(inout IndirectReservoir resv, int clamp) {
    if (resv.num > clamp) {
        resv.weight *= float(clamp) / float(resv.num);
        resv.num = clamp;
    }
}

#endif