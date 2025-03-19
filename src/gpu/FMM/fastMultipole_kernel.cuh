/*
   Copyright 2023 Your Name

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

// Kernel for building the octree
__global__ void BuildTreeKernel(Pos* pos, int* particleIndices, Cell* cells, 
                               int numParticles, int maxLevel, float domainSize);

// Kernel for computing multipole expansions (P2M)
__global__ void ComputeMultipolesKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                       Complex* multipoles, int numCells, int p);

// Kernel for multipole-to-multipole translations (M2M)
__global__ void M2MKernel(Cell* cells, Complex* multipoles, int numCells, int p, int level);

// Kernel for multipole-to-local translations (M2L)
__global__ void M2LKernel(Cell* cells, Complex* multipoles, Complex* locals, 
                         int numCells, int p, float theta, int level);

// Kernel for local-to-local translations (L2L)
__global__ void L2LKernel(Cell* cells, Complex* locals, int numCells, int p, int level);

// Kernel for evaluating local expansions (L2P)
__global__ void EvaluateLocalsKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                    Complex* locals, Acc* acc, int numParticles, int p);

// Kernel for direct particle-particle interactions (P2P)
__global__ void DirectInteractionsKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                        Acc* acc, int numParticles, float G);

// Kernel for updating particle positions and velocities
__global__ void UpdateParticlesKernel(Pos* pos, Vel* vel, Acc* acc, int numParticles, float dt);

#endif // FAST_MULTIPOLE_KERNEL_H_ 