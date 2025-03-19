#ifndef __FAST_MULTIPOLE_KERNEL_H__
#define __FAST_MULTIPOLE_KERNEL_H__

#include <cuda_runtime.h>
#include "constants.h"

// Kernel functions for FMM algorithm
void computeMortonIndicesKernel(int numParticles, float4* d_pos, int* d_mortonIndex, 
                               float3 boxMin, float boxSize, int maxLevel);

void sortParticlesKernel(int numParticles, float4* d_pos, float3* d_vel, 
                        int* d_mortonIndex, int* d_sortedIndex);

void countBoxesKernel(int numParticles, int* d_mortonIndex, int* d_boxCount, 
                     int* d_boxStart, int maxLevel);

void buildInteractionListsKernel(int numBoxes, int* d_boxIndexFull, int* d_interactionLists, 
                                int* d_numInteractions, int maxLevel);

// P2P kernel (particle to particle)
void p2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, int* d_interactionLists, 
              int* d_numInteractions, float4* d_pos, float3* d_acc);

// P2M kernel (particle to multipole)
void p2mKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, float4* d_pos, 
              float* d_Mnm, float3 boxMin, float boxSize, int maxLevel);

// M2M kernel (multipole to multipole)
void m2mKernel(int numBoxes, int* d_boxParent, float* d_Mnm, float* d_MnmParent, 
              int maxLevel);

// M2L kernel (multipole to local)
void m2lKernel(int numBoxes, int* d_interactionLists, int* d_numInteractions, 
              float* d_Mnm, float* d_Lnm, float3 boxMin, float boxSize, int maxLevel);

// L2L kernel (local to local)
void l2lKernel(int numBoxes, int* d_boxChildren, float* d_Lnm, float* d_LnmChildren, 
              int maxLevel);

// L2P kernel (local to particle)
void l2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, float* d_Lnm, 
              float4* d_pos, float3* d_acc, float3 boxMin, float boxSize, int maxLevel);

// M2P kernel (multipole to particle)
void m2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, int* d_interactionLists, 
              int* d_numInteractions, float* d_Mnm, float4* d_pos, float3* d_acc, 
              float3 boxMin, float boxSize, int maxLevel);

// Update particles kernel
void updateParticlesKernel(int numParticles, float4* d_pos, float3* d_vel, 
                          float3* d_acc, float deltaTime, float damping);

#endif // __FAST_MULTIPOLE_KERNEL_H__ 