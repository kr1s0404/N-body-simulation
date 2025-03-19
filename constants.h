#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <cmath>

// FMM parameters
const int maxParticles         = 10000000;   // max number of particles
const int numExpansions        = 10;         // order of expansion in FMM
const int maxP2PInteraction    = 27;         // max of P2P interacting boxes
const int maxM2LInteraction    = 189;        // max of M2L interacting boxes
const int numRelativeBox       = 512;        // max of relative box positioning
const int targetBufferSize     = 200000;     // max of GPU target buffer
const int sourceBufferSize     = 100000;     // max of GPU source buffer
const int threadsPerBlockTypeA = 128;        // size of GPU thread block P2P
const int threadsPerBlockTypeB = 64;         // size of GPU thread block M2L
const float eps                = 1e-6;       // single precision epsilon
const float inv4PI             = 0.25f/M_PI; // Laplace kernel coefficient

// Derived constants
const int numExpansion2        = numExpansions*numExpansions;
const int numExpansion4        = numExpansion2*numExpansion2;
const int numCoefficients      = numExpansions*(numExpansions+1)/2;
const int DnmSize              = (4*numExpansion2*numExpansions-numExpansions)/3;

// Simulation parameters
const float timeStep           = 0.001f;     // simulation time step
const float damping            = 0.995f;     // velocity damping factor
const float softening          = 0.001f;     // softening factor to avoid singularities
const float G                  = 1.0f;       // gravitational constant

// Rendering parameters
const int windowWidth          = 1024;       // window width
const int windowHeight         = 768;        // window height

#endif // __CONSTANTS_H__ 