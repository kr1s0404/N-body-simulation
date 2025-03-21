/*
   Copyright 2023 Hsin-Hung Wu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef FAST_MULTIPOLE_KERNEL_H_
#define FAST_MULTIPOLE_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include "fastMultipoleCuda.cuh"

// Core FMM kernels
__global__ void ResetCellsKernel(Cell *cells, int *mutex, int nCells, int nBodies);
__global__ void ComputeBoundingBoxKernel(Body *bodies, Cell *cells, int *mutex, int nBodies);
__global__ void BuildTreeKernel(Body *bodies, Cell *cells, int *cellCount, int *sortedIndex, int *mutex, int nBodies, int maxDepth);
__global__ void ComputeMultipolesKernel(Body *bodies, Cell *cells, int *sortedIndex, int nCells);
__global__ void TranslateMultipolesKernel(Cell *cells, int nCells);
__global__ void ComputeLocalExpansionsKernel(Cell *cells, int nCells);
__global__ void EvaluateLocalExpansionsKernel(Body *bodies, Cell *cells, int *sortedIndex, int nBodies);
__global__ void DirectEvaluationKernel(Body *bodies, Cell *cells, int nCells, int nBodies);
__global__ void ComputeForcesAndUpdateKernel(Body *bodies, int nBodies);
__global__ void UpdateBodiesKernel(Body *bodies, int nBodies, double dt);

// Helper device functions
__device__ int getQuadrant(Vector position, Vector center);
__device__ void computeMultipoleExpansion(Body *bodies, int start, int count, Complex *multipole, Vector center);
__device__ void translateMultipole(Complex *source, Complex *target, Vector sourceCenter, Vector targetCenter);
__device__ void translateMultipoleToLocal(Complex *multipole, Complex *local, Vector multipoleCenter, Vector localCenter);
__device__ void evaluateLocalExpansion(Complex *local, Vector center, Vector position, Vector *force);
__device__ void computeDirectForce(Body body1, Body body2, Vector *force);
__device__ bool isCollide(Body b1, Body b2);
__device__ bool isCollide(Body b, Vector cm, double totalMass);

// Custom atomic operations for double
__device__ double atomicMin(double* address, double val);
__device__ double atomicMax(double* address, double val);

#endif 