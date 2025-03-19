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

#include <iostream>
#include <cmath>
#include "fastMultipole_kernel.cuh"
#include "constants.h"
#include "err.h"

// Constants for FMM
#define MAX_LEVEL 8
#define MAX_CELLS (1 << (3 * MAX_LEVEL + 3))
#define BLOCK_SIZE 256
#define G 6.67430e-11f  // Gravitational constant

// Helper function to calculate the number of cells at a given level
int numCellsAtLevel(int level) {
    return 1 << (3 * level);
}

// Helper function to calculate the total number of cells up to a given level
int totalCellsUpToLevel(int level) {
    return (1 << (3 * (level + 1))) / 7;
}

// Create an FMM system
FMMSystem* createFMMSystem(int numParticles, Pos* positions, Vel* velocities) {
    return new FMMSystem(numParticles, positions, velocities);
}

// Destroy an FMM system
void destroyFMMSystem(FMMSystem* system) {
    delete system;
}

// FMMSystem constructor
FMMSystem::FMMSystem(int numParticles, Pos* positions, Vel* velocities) : numParticles(numParticles) {
    // Set default parameters
    maxLevel = MAX_LEVEL;
    domainSize = 10.0f;
    p = 4;  // Default multipole expansion order
    theta = 0.5f;  // Default multipole acceptance criterion
    
    // Calculate number of cells
    numCells = totalCellsUpToLevel(maxLevel);
    
    // Allocate host memory
    h_pos = new Pos[numParticles];
    h_vel = new Vel[numParticles];
    h_acc = new Acc[numParticles];
    h_cells = new Cell[numCells];
    h_particleIndices = new int[numParticles];
    
    // Calculate number of multipole coefficients: (p+1)^2 for each cell
    int numCoefficients = numCells * (p+1) * (p+1);
    h_multipoles = new Complex[numCoefficients];
    h_locals = new Complex[numCoefficients];
    
    // Copy input data
    memcpy(h_pos, positions, numParticles * sizeof(Pos));
    memcpy(h_vel, velocities, numParticles * sizeof(Vel));
    
    // Initialize particle indices
    for (int i = 0; i < numParticles; i++) {
        h_particleIndices[i] = i;
    }
    
    // Initialize accelerations to zero
    memset(h_acc, 0, numParticles * sizeof(Acc));
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pos, numParticles * sizeof(Pos)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_vel, numParticles * sizeof(Vel)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_acc, numParticles * sizeof(Acc)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cells, numCells * sizeof(Cell)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_particleIndices, numParticles * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_multipoles, numCoefficients * sizeof(Complex)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_locals, numCoefficients * sizeof(Complex)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_pos, h_pos, numParticles * sizeof(Pos), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vel, h_vel, numParticles * sizeof(Vel), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_acc, h_acc, numParticles * sizeof(Acc), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_particleIndices, h_particleIndices, numParticles * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize multipoles and locals to zero
    CHECK_CUDA_ERROR(cudaMemset(d_multipoles, 0, numCoefficients * sizeof(Complex)));
    CHECK_CUDA_ERROR(cudaMemset(d_locals, 0, numCoefficients * sizeof(Complex)));
}

// FMMSystem destructor
FMMSystem::~FMMSystem() {
    // Free host memory
    delete[] h_pos;
    delete[] h_vel;
    delete[] h_acc;
    delete[] h_cells;
    delete[] h_particleIndices;
    delete[] h_multipoles;
    delete[] h_locals;
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_pos));
    CHECK_CUDA_ERROR(cudaFree(d_vel));
    CHECK_CUDA_ERROR(cudaFree(d_acc));
    CHECK_CUDA_ERROR(cudaFree(d_cells));
    CHECK_CUDA_ERROR(cudaFree(d_particleIndices));
    CHECK_CUDA_ERROR(cudaFree(d_multipoles));
    CHECK_CUDA_ERROR(cudaFree(d_locals));
}

// Set domain size
void FMMSystem::setDomainSize(float size) {
    domainSize = size;
}

// Set multipole expansion order
void FMMSystem::setMultipoleOrder(int order) {
    p = order;
}

// Set theta parameter
void FMMSystem::setTheta(float t) {
    theta = t;
}

// Build the octree
void FMMSystem::buildTree() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    BuildTreeKernel<<<gridSize, blockSize>>>(d_pos, d_particleIndices, d_cells, 
                                           numParticles, maxLevel, domainSize);
    CHECK_LAST_CUDA_ERROR();
}

// Compute multipole expansions (P2M)
void FMMSystem::computeMultipoles() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (numCells + blockSize - 1) / blockSize;
    
    // Reset multipoles to zero
    int numCoefficients = numCells * (p+1) * (p+1);
    CHECK_CUDA_ERROR(cudaMemset(d_multipoles, 0, numCoefficients * sizeof(Complex)));
    
    ComputeMultipolesKernel<<<gridSize, blockSize>>>(d_pos, d_particleIndices, d_cells, 
                                                   d_multipoles, numCells, p);
    CHECK_LAST_CUDA_ERROR();
    
    // Perform M2M translations level by level, starting from the bottom
    for (int level = maxLevel; level > 0; level--) {
        int numCellsLevel = numCellsAtLevel(level);
        gridSize = (numCellsLevel + blockSize - 1) / blockSize;
        
        M2MKernel<<<gridSize, blockSize>>>(d_cells, d_multipoles, numCells, p, level);
        CHECK_LAST_CUDA_ERROR();
    }
}

// Translate multipoles to locals (M2L)
void FMMSystem::translateMultipoles() {
    int blockSize = BLOCK_SIZE;
    
    // Reset locals to zero
    int numCoefficients = numCells * (p+1) * (p+1);
    CHECK_CUDA_ERROR(cudaMemset(d_locals, 0, numCoefficients * sizeof(Complex)));
    
    // Perform M2L translations level by level
    for (int level = 2; level <= maxLevel; level++) {
        int numCellsLevel = numCellsAtLevel(level);
        int gridSize = (numCellsLevel + blockSize - 1) / blockSize;
        
        M2LKernel<<<gridSize, blockSize>>>(d_cells, d_multipoles, d_locals, 
                                         numCells, p, theta, level);
        CHECK_LAST_CUDA_ERROR();
    }
}

// Compute local expansions (L2L)
void FMMSystem::computeLocalExpansions() {
    int blockSize = BLOCK_SIZE;
    
    // Perform L2L translations level by level, starting from the top
    for (int level = 2; level < maxLevel; level++) {
        int numCellsLevel = numCellsAtLevel(level);
        int gridSize = (numCellsLevel + blockSize - 1) / blockSize;
        
        L2LKernel<<<gridSize, blockSize>>>(d_cells, d_locals, numCells, p, level);
        CHECK_LAST_CUDA_ERROR();
    }
}

// Evaluate local expansions (L2P)
void FMMSystem::evaluateLocalExpansions() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    // Reset accelerations to zero
    CHECK_CUDA_ERROR(cudaMemset(d_acc, 0, numParticles * sizeof(Acc)));
    
    EvaluateLocalsKernel<<<gridSize, blockSize>>>(d_pos, d_particleIndices, d_cells, 
                                                d_locals, d_acc, numParticles, p);
    CHECK_LAST_CUDA_ERROR();
}

// Direct particle-particle interactions (P2P)
void FMMSystem::directInteractions() {
    int blockSize = BLOCK_SIZE;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    DirectInteractionsKernel<<<gridSize, blockSize>>>(d_pos, d_particleIndices, d_cells, 
                                                    d_acc, numParticles, G);
    CHECK_LAST_CUDA_ERROR();
}

// Update particle positions and velocities
void FMMSystem::updateParticles(float dt) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    UpdateParticlesKernel<<<gridSize, blockSize>>>(d_pos, d_vel, d_acc, numParticles, dt);
    CHECK_LAST_CUDA_ERROR();
}

// Perform a simulation step
void FMMSystem::step(float dt) {
    // Build the octree
    buildTree();
    
    // Compute multipole expansions
    computeMultipoles();
    
    // Translate multipoles to locals
    translateMultipoles();
    
    // Compute local expansions
    computeLocalExpansions();
    
    // Evaluate local expansions
    evaluateLocalExpansions();
    
    // Compute direct interactions for nearby particles
    directInteractions();
    
    // Update particle positions and velocities
    updateParticles(dt);
}

// Get current particle positions
void FMMSystem::getPositions(Pos* positions) {
    CHECK_CUDA_ERROR(cudaMemcpy(positions, d_pos, numParticles * sizeof(Pos), cudaMemcpyDeviceToHost));
}

// Get current particle velocities
void FMMSystem::getVelocities(Vel* velocities) {
    CHECK_CUDA_ERROR(cudaMemcpy(velocities, d_vel, numParticles * sizeof(Vel), cudaMemcpyDeviceToHost));
} 