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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "fastMultipole_kernel.cuh"
#include "constants.h"

// Helper device functions for complex arithmetic
__device__ Complex complexAdd(Complex a, Complex b) {
    Complex result;
    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;
    return result;
}

__device__ Complex complexMul(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

__device__ Complex complexScale(Complex a, float scale) {
    Complex result;
    result.real = a.real * scale;
    result.imag = a.imag * scale;
    return result;
}

// Helper function to compute cell index from position and level
__device__ int computeCellIndex(float x, float y, float z, int level, float domainSize) {
    float cellSize = domainSize / (1 << level);
    int ix = floor(x / cellSize);
    int iy = floor(y / cellSize);
    int iz = floor(z / cellSize);
    
    // Compute Morton code (Z-order curve)
    unsigned int morton = 0;
    for (int i = 0; i < level; i++) {
        morton |= ((ix & (1 << i)) << (2 * i)) | 
                  ((iy & (1 << i)) << (2 * i + 1)) | 
                  ((iz & (1 << i)) << (2 * i + 2));
    }
    
    // Compute cell index
    return (1 << (3 * level) - 1) / 7 + morton;
}

// Helper function to compute cell center
__device__ void computeCellCenter(int cellIndex, int level, float domainSize, float& cx, float& cy, float& cz) {
    float cellSize = domainSize / (1 << level);
    
    // Extract Morton code
    unsigned int morton = cellIndex - ((1 << (3 * level) - 1) / 7);
    
    // Extract coordinates
    int ix = 0, iy = 0, iz = 0;
    for (int i = 0; i < level; i++) {
        ix |= ((morton >> (3 * i)) & 1) << i;
        iy |= ((morton >> (3 * i + 1)) & 1) << i;
        iz |= ((morton >> (3 * i + 2)) & 1) << i;
    }
    
    // Compute center coordinates
    cx = (ix + 0.5f) * cellSize;
    cy = (iy + 0.5f) * cellSize;
    cz = (iz + 0.5f) * cellSize;
}

// Helper function to compute spherical harmonics
__device__ Complex sphericalHarmonic(int l, int m, float theta, float phi) {
    Complex result;
    
    // Simple implementation for l <= 4
    if (l == 0 && m == 0) {
        result.real = 0.5f * sqrtf(1.0f / M_PI);
        result.imag = 0.0f;
    } else if (l == 1 && m == -1) {
        result.real = 0.5f * sqrtf(3.0f / (2.0f * M_PI)) * sinf(theta) * sinf(phi);
        result.imag = 0.0f;
    } else if (l == 1 && m == 0) {
        result.real = 0.5f * sqrtf(3.0f / M_PI) * cosf(theta);
        result.imag = 0.0f;
    } else if (l == 1 && m == 1) {
        result.real = -0.5f * sqrtf(3.0f / (2.0f * M_PI)) * sinf(theta) * cosf(phi);
        result.imag = 0.0f;
    } else {
        // Higher order terms would be implemented here
        result.real = 0.0f;
        result.imag = 0.0f;
    }
    
    return result;
}

// Kernel for building the octree
__global__ void BuildTreeKernel(Pos* pos, int* particleIndices, Cell* cells, 
                               int numParticles, int maxLevel, float domainSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        float x = pos[i].x;
        float y = pos[i].y;
        float z = pos[i].z;
        
        // Find the leaf cell for this particle
        int cellIndex = computeCellIndex(x, y, z, maxLevel, domainSize);
        
        // Atomically increment the particle count for this cell
        int particleIdx = atomicAdd(&cells[cellIndex].numParticles, 1);
        
        // Store the particle index
        particleIndices[cells[cellIndex].particleOffset + particleIdx] = i;
        
        // Initialize cell properties if this is the first particle
        if (particleIdx == 0) {
            float cx, cy, cz;
            computeCellCenter(cellIndex, maxLevel, domainSize, cx, cy, cz);
            
            cells[cellIndex].x = cx;
            cells[cellIndex].y = cy;
            cells[cellIndex].z = cz;
            cells[cellIndex].size = domainSize / (1 << maxLevel);
            
            // Compute parent index
            if (maxLevel > 0) {
                int parentLevel = maxLevel - 1;
                int parentIndex = computeCellIndex(x, y, z, parentLevel, domainSize);
                cells[cellIndex].parent = parentIndex;
                
                // Add this cell as a child of the parent
                int childIdx = atomicAdd(&cells[parentIndex].numParticles, 1);
                if (childIdx < 8) {
                    cells[parentIndex].children[childIdx] = cellIndex;
                }
            }
        }
    }
}

// Kernel for computing multipole expansions (P2M)
__global__ void ComputeMultipolesKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                       Complex* multipoles, int numCells, int p) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cellIdx < numCells) {
        Cell cell = cells[cellIdx];
        
        // Skip non-leaf cells
        if (cell.children[0] != 0) return;
        
        float cx = cell.x;
        float cy = cell.y;
        float cz = cell.z;
        
        // For each particle in this cell
        for (int i = 0; i < cell.numParticles; i++) {
            int particleIdx = particleIndices[cell.particleOffset + i];
            float mass = pos[particleIdx].w;
            
            // Compute relative position
            float dx = pos[particleIdx].x - cx;
            float dy = pos[particleIdx].y - cy;
            float dz = pos[particleIdx].z - cz;
            
            // Convert to spherical coordinates
            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            float theta = acosf(dz / (r + 1e-10f));
            float phi = atan2f(dy, dx);
            
            // Compute multipole expansion
            for (int l = 0; l <= p; l++) {
                for (int m = -l; m <= l; m++) {
                    int idx = cellIdx * (p+1)*(p+1) + l*(l+1) + m;
                    
                    // Compute spherical harmonic
                    Complex Ylm = sphericalHarmonic(l, m, theta, phi);
                    
                    // Scale by mass and r^l
                    float scale = mass * powf(r, l);
                    Complex contrib = complexScale(Ylm, scale);
                    
                    // Add to multipole expansion
                    atomicAdd(&multipoles[idx].real, contrib.real);
                    atomicAdd(&multipoles[idx].imag, contrib.imag);
                }
            }
        }
    }
}

// Kernel for multipole-to-multipole translations (M2M)
__global__ void M2MKernel(Cell* cells, Complex* multipoles, int numCells, int p, int level) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int levelOffset = ((1 << (3 * level)) - 1) / 7;
    int numCellsLevel = 1 << (3 * level);
    
    if (cellIdx < numCellsLevel) {
        int globalCellIdx = levelOffset + cellIdx;
        
        // Skip leaf cells
        if (cells[globalCellIdx].children[0] == 0) return;
        
        // For each child
        for (int childIdx = 0; childIdx < 8; childIdx++) {
            int childCellIdx = cells[globalCellIdx].children[childIdx];
            if (childCellIdx == 0) continue;
            
            // Compute relative position
            float dx = cells[childCellIdx].x - cells[globalCellIdx].x;
            float dy = cells[childCellIdx].y - cells[globalCellIdx].y;
            float dz = cells[childCellIdx].z - cells[globalCellIdx].z;
            
            // Convert to spherical coordinates
            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            float theta = acosf(dz / (r + 1e-10f));
            float phi = atan2f(dy, dx);
            
            // For each multipole coefficient
            for (int l = 0; l <= p; l++) {
                for (int m = -l; m <= l; m++) {
                    int parentIdx = globalCellIdx * (p+1)*(p+1) + l*(l+1) + m;
                    int childIdx = childCellIdx * (p+1)*(p+1) + l*(l+1) + m;
                    
                    // Translate multipole
                    Complex Ylm = sphericalHarmonic(l, m, theta, phi);
                    Complex childMultipole = multipoles[childIdx];
                    Complex translated = complexMul(childMultipole, Ylm);
                    
                    // Add to parent's multipole
                    atomicAdd(&multipoles[parentIdx].real, translated.real);
                    atomicAdd(&multipoles[parentIdx].imag, translated.imag);
                }
            }
        }
    }
}

// Kernel for multipole-to-local translations (M2L)
__global__ void M2LKernel(Cell* cells, Complex* multipoles, Complex* locals, 
                         int numCells, int p, float theta, int level) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int levelOffset = ((1 << (3 * level)) - 1) / 7;
    int numCellsLevel = 1 << (3 * level);
    
    if (cellIdx < numCellsLevel) {
        int globalCellIdx = levelOffset + cellIdx;
        Cell cell = cells[globalCellIdx];
        
        // For each other cell at this level
        for (int otherCellIdx = levelOffset; otherCellIdx < levelOffset + numCellsLevel; otherCellIdx++) {
            if (otherCellIdx == globalCellIdx) continue;
            
            Cell otherCell = cells[otherCellIdx];
            
            // Compute distance between cells
            float dx = otherCell.x - cell.x;
            float dy = otherCell.y - cell.y;
            float dz = otherCell.z - cell.z;
            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            
            // Check if cells are well-separated
            if (r > cell.size / theta) {
                // Convert to spherical coordinates
                float theta = acosf(dz / (r + 1e-10f));
                float phi = atan2f(dy, dx);
                
                // For each multipole coefficient
                for (int l = 0; l <= p; l++) {
                    for (int m = -l; m <= l; m++) {
                        int sourceIdx = otherCellIdx * (p+1)*(p+1) + l*(l+1) + m;
                        int targetIdx = globalCellIdx * (p+1)*(p+1) + l*(l+1) + m;
                        
                        // Compute translation operator
                        Complex Ylm = sphericalHarmonic(l, m, theta, phi);
                        
                        // Scale by 1/r^(l+1)
                        float scale = 1.0f / powf(r, l + 1);
                        Complex operator_lm = complexScale(Ylm, scale);
                        
                        // Translate multipole to local
                        Complex multipole = multipoles[sourceIdx];
                        Complex translated = complexMul(multipole, operator_lm);
                        
                        // Add to local expansion
                        atomicAdd(&locals[targetIdx].real, translated.real);
                        atomicAdd(&locals[targetIdx].imag, translated.imag);
                    }
                }
            }
        }
    }
}

// Kernel for local-to-local translations (L2L)
__global__ void L2LKernel(Cell* cells, Complex* locals, int numCells, int p, int level) {
    int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int levelOffset = ((1 << (3 * level)) - 1) / 7;
    int numCellsLevel = 1 << (3 * level);
    
    if (cellIdx < numCellsLevel) {
        int globalCellIdx = levelOffset + cellIdx;
        Cell cell = cells[globalCellIdx];
        
        // Skip cells without a parent
        if (level == 0) return;
        
        int parentIdx = cell.parent;
        
        // Compute relative position
        float dx = cell.x - cells[parentIdx].x;
        float dy = cell.y - cells[parentIdx].y;
        float dz = cell.z - cells[parentIdx].z;
        
        // Convert to spherical coordinates
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        float theta = acosf(dz / (r + 1e-10f));
        float phi = atan2f(dy, dx);
        
        // For each local coefficient
        for (int l = 0; l <= p; l++) {
            for (int m = -l; m <= l; m++) {
                int childIdx = globalCellIdx * (p+1)*(p+1) + l*(l+1) + m;
                int parentLocalIdx = parentIdx * (p+1)*(p+1) + l*(l+1) + m;
                
                // Compute translation operator
                Complex Ylm = sphericalHarmonic(l, m, theta, phi);
                
                // Translate local expansion
                Complex parentLocal = locals[parentLocalIdx];
                Complex translated = complexMul(parentLocal, Ylm);
                
                // Add to child's local expansion
                atomicAdd(&locals[childIdx].real, translated.real);
                atomicAdd(&locals[childIdx].imag, translated.imag);
            }
        }
    }
}

// Kernel for evaluating local expansions (L2P)
__global__ void EvaluateLocalsKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                    Complex* locals, Acc* acc, int numParticles, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        float x = pos[i].x;
        float y = pos[i].y;
        float z = pos[i].z;
        
        // Find the leaf cell for this particle
        int cellIdx = computeCellIndex(x, y, z, p, 10.0f); // Assuming domain size is 10.0
        Cell cell = cells[cellIdx];
        
        // Compute relative position
        float dx = x - cell.x;
        float dy = y - cell.y;
        float dz = z - cell.z;
        
        // Convert to spherical coordinates
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        float theta = acosf(dz / (r + 1e-10f));
        float phi = atan2f(dy, dx);
        
        // Initialize acceleration
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        
        // Evaluate local expansion
        for (int l = 0; l <= p; l++) {
            for (int m = -l; m <= l; m++) {
                int idx = cellIdx * (p+1)*(p+1) + l*(l+1) + m;
                
                // Compute spherical harmonic
                Complex Ylm = sphericalHarmonic(l, m, theta, phi);
                
                // Scale by r^l
                float scale = powf(r, l);
                Complex term = complexScale(Ylm, scale);
                
                // Multiply by local coefficient
                Complex local = locals[idx];
                Complex result = complexMul(term, local);
                
                // Add to acceleration components
                // This is a simplification - in a real implementation, 
                // we would compute the gradient of the potential
                ax += result.real;
                ay += result.imag;
                az += (result.real + result.imag) * 0.5f;
            }
        }
        
        // Store acceleration
        acc[i].x = ax;
        acc[i].y = ay;
        acc[i].z = az;
    }
}

// Kernel for direct particle-particle interactions (P2P)
__global__ void DirectInteractionsKernel(Pos* pos, int* particleIndices, Cell* cells, 
                                        Acc* acc, int numParticles, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        float xi = pos[i].x;
        float yi = pos[i].y;
        float zi = pos[i].z;
        float mi = pos[i].w;
        
        // Find the leaf cell for this particle
        int cellIdx = computeCellIndex(xi, yi, zi, 8, 10.0f); // Assuming max level is 8 and domain size is 10.0
        Cell cell = cells[cellIdx];
        
        // Initialize acceleration
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        
        // For each neighboring cell
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    // Compute neighbor cell index
                    float nx = cell.x + dx * cell.size;
                    float ny = cell.y + dy * cell.size;
                    float nz = cell.z + dz * cell.size;
                    
                    int neighborIdx = computeCellIndex(nx, ny, nz, 8, 10.0f);
                    Cell neighbor = cells[neighborIdx];
                    
                    // For each particle in the neighbor cell
                    for (int j = 0; j < neighbor.numParticles; j++) {
                        int particleIdx = particleIndices[neighbor.particleOffset + j];
                        
                        // Skip self-interaction
                        if (particleIdx == i) continue;
                        
                        float xj = pos[particleIdx].x;
                        float yj = pos[particleIdx].y;
                        float zj = pos[particleIdx].z;
                        float mj = pos[particleIdx].w;
                        
                        // Compute distance
                        float dx = xj - xi;
                        float dy = yj - yi;
                        float dz = zj - zi;
                        float r2 = dx*dx + dy*dy + dz*dz;
                        
                        // Avoid division by zero
                        if (r2 < 1e-10f) continue;
                        
                        float r = sqrtf(r2);
                        float r3 = r2 * r;
                        
                        // Compute gravitational force
                        float f = G * mi * mj / r3;
                        
                        // Add to acceleration
                        ax += f * dx;
                        ay += f * dy;
                        az += f * dz;
                    }
                }
            }
        }
        
        // Store acceleration
        acc[i].x = ax;
        acc[i].y = ay;
        acc[i].z = az;
    }
}

// Kernel for updating particle positions and velocities
__global__ void UpdateParticlesKernel(Pos* pos, Vel* vel, Acc* acc, int numParticles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        // Update velocity
        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;
        vel[i].z += acc[i].z * dt;
        
        // Update position
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
} 