#include "fastMultipoleCuda.cuh"
#include "fastMultipole_kernel.cuh"
#include "err.h"
#include <stdio.h>
#include <stdlib.h>

// Constructor
FMMSystem::FMMSystem() : 
    numParticles(0),
    d_pos(nullptr),
    d_vel(nullptr),
    d_acc(nullptr),
    maxLevel(0),
    numBoxIndexFull(0),
    numBoxIndexLeaf(0),
    numBoxIndexTotal(0),
    d_boxIndexMask(nullptr),
    d_boxIndexFull(nullptr),
    d_levelOffset(nullptr),
    d_mortonIndex(nullptr),
    d_particleOffset(nullptr),
    d_numInteraction(nullptr),
    d_interactionList(nullptr),
    d_Mnm(nullptr),
    d_Lnm(nullptr),
    rootBoxSize(2.0f)
{
    boxMin = make_float3(-1.0f, -1.0f, -1.0f);
}

// Destructor
FMMSystem::~FMMSystem() {
    freeMemory();
}

// Initialize the FMM system
void FMMSystem::initialize(int numParticles, Pos* h_pos, Vel* h_vel) {
    this->numParticles = numParticles;
    
    // Determine optimal tree depth based on particle count
    setOptimumLevel(numParticles);
    
    // Allocate memory for particle and tree data
    allocateMemory();
    
    // Copy particle data to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_pos, h_pos, numParticles * sizeof(Pos), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_vel, h_vel, numParticles * sizeof(Vel), cudaMemcpyHostToDevice));
    
    // Initialize acceleration to zero
    CUDA_CHECK_ERROR(cudaMemset(d_acc, 0, numParticles * sizeof(Acc)));
    
    // Build the tree structure
    buildTree();
}

// Reset the FMM system
void FMMSystem::reset() {
    // Clear accelerations
    CUDA_CHECK_ERROR(cudaMemset(d_acc, 0, numParticles * sizeof(Acc)));
}

// Main simulation step
void FMMSystem::step(float deltaTime) {
    // Reset accelerations
    reset();
    
    // Rebuild the tree with updated particle positions
    buildTree();
    
    // Compute multipole expansions
    computeMultipoles();
    
    // Compute forces using FMM
    computeForces();
    
    // Update particle positions and velocities
    updateParticles(deltaTime);
}

// Get particle positions from device
void FMMSystem::getPositions(Pos* h_pos) {
    CUDA_CHECK_ERROR(cudaMemcpy(h_pos, d_pos, numParticles * sizeof(Pos), cudaMemcpyDeviceToHost));
}

// Get particle velocities from device
void FMMSystem::getVelocities(Vel* h_vel) {
    CUDA_CHECK_ERROR(cudaMemcpy(h_vel, d_vel, numParticles * sizeof(Vel), cudaMemcpyDeviceToHost));
}

// Set the domain size
void FMMSystem::setDomainSize(float size) {
    rootBoxSize = size;
    boxMin = make_float3(-size/2, -size/2, -size/2);
}

// Set optimal tree depth based on particle count
void FMMSystem::setOptimumLevel(int numParticles) {
    // Determine optimal tree depth based on number of particles
    // These thresholds are based on performance testing
    float level_switch[6] = {1e5, 7e5, 7e6, 5e7, 3e8, 2e9}; // gpu-fmm thresholds
    
    maxLevel = 1;
    if (numParticles < level_switch[0]) {
        maxLevel += 1;
    } else if (numParticles < level_switch[1]) {
        maxLevel += 2;
    } else if (numParticles < level_switch[2]) {
        maxLevel += 3;
    } else if (numParticles < level_switch[3]) {
        maxLevel += 4;
    } else if (numParticles < level_switch[4]) {
        maxLevel += 5;
    } else if (numParticles < level_switch[5]) {
        maxLevel += 6;
    } else {
        maxLevel += 7;
    }
    
    printf("FMM tree level: %d\n", maxLevel);
    numBoxIndexFull = 1 << (3 * maxLevel);
}

// Allocate memory for FMM data structures
void FMMSystem::allocateMemory() {
    // Allocate particle data
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pos, numParticles * sizeof(Pos)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_vel, numParticles * sizeof(Vel)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_acc, numParticles * sizeof(Acc)));
    
    // Allocate tree data
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_mortonIndex, numParticles * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxIndexMask, numBoxIndexFull * sizeof(int)));
    
    // We'll allocate the rest of the memory after we know how many non-empty boxes we have
}

// Free allocated memory
void FMMSystem::freeMemory() {
    // Free particle data
    if (d_pos) cudaFree(d_pos);
    if (d_vel) cudaFree(d_vel);
    if (d_acc) cudaFree(d_acc);
    
    // Free tree data
    if (d_mortonIndex) cudaFree(d_mortonIndex);
    if (d_boxIndexMask) cudaFree(d_boxIndexMask);
    if (d_boxIndexFull) cudaFree(d_boxIndexFull);
    if (d_levelOffset) cudaFree(d_levelOffset);
    if (d_particleOffset) cudaFree(d_particleOffset);
    if (d_numInteraction) cudaFree(d_numInteraction);
    if (d_interactionList) cudaFree(d_interactionList);
    
    // Free multipole data
    if (d_Mnm) cudaFree(d_Mnm);
    if (d_Lnm) cudaFree(d_Lnm);
    
    // Reset pointers
    d_pos = nullptr;
    d_vel = nullptr;
    d_acc = nullptr;
    d_mortonIndex = nullptr;
    d_boxIndexMask = nullptr;
    d_boxIndexFull = nullptr;
    d_levelOffset = nullptr;
    d_particleOffset = nullptr;
    d_numInteraction = nullptr;
    d_interactionList = nullptr;
    d_Mnm = nullptr;
    d_Lnm = nullptr;
}

// Build the octree structure
void FMMSystem::buildTree() {
    // Compute Morton indices for all particles
    computeMortonIndicesKernel(numParticles, d_pos, d_mortonIndex, boxMin, rootBoxSize, maxLevel);
    
    // Sort particles by Morton index
    sortParticlesKernel(numParticles, d_pos, d_vel, d_mortonIndex, nullptr);
    
    // Count non-empty boxes and set up box data structures
    int* d_boxCount;
    int* d_boxStart;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxCount, numBoxIndexFull * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxStart, numBoxIndexFull * sizeof(int)));
    
    countBoxesKernel(numParticles, d_mortonIndex, d_boxCount, d_boxStart, maxLevel);
    
    // Get the number of non-empty boxes
    int* h_boxCount = new int[1];
    CUDA_CHECK_ERROR(cudaMemcpy(h_boxCount, d_boxCount, sizeof(int), cudaMemcpyDeviceToHost));
    numBoxIndexLeaf = h_boxCount[0];
    delete[] h_boxCount;
    
    // Allocate memory for box data now that we know the count
    if (d_boxIndexFull) cudaFree(d_boxIndexFull);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxIndexFull, numBoxIndexLeaf * sizeof(int)));
    
    if (d_particleOffset) cudaFree(d_particleOffset);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_particleOffset, 2 * numBoxIndexLeaf * sizeof(int)));
    
    // Set up level offsets
    if (d_levelOffset) cudaFree(d_levelOffset);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_levelOffset, (maxLevel + 1) * sizeof(int)));
    
    // Compute interaction lists
    if (d_numInteraction) cudaFree(d_numInteraction);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_numInteraction, numBoxIndexLeaf * sizeof(int)));
    
    if (d_interactionList) cudaFree(d_interactionList);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_interactionList, numBoxIndexLeaf * maxM2LInteraction * sizeof(int)));
    
    buildInteractionListsKernel(numBoxIndexLeaf, d_boxIndexFull, d_interactionList, d_numInteraction, maxLevel);
    
    // Allocate memory for multipole expansions
    numBoxIndexTotal = numBoxIndexLeaf;  // This will be updated to include all levels
    
    if (d_Mnm) cudaFree(d_Mnm);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_Mnm, 2 * numBoxIndexTotal * numCoefficients * sizeof(float)));
    
    if (d_Lnm) cudaFree(d_Lnm);
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_Lnm, 2 * numBoxIndexTotal * numCoefficients * sizeof(float)));
    
    // Clean up temporary memory
    cudaFree(d_boxCount);
    cudaFree(d_boxStart);
}

// Compute multipole expansions
void FMMSystem::computeMultipoles() {
    // P2M: Particle to Multipole
    p2mKernel(numBoxIndexLeaf, d_particleOffset, d_particleOffset + numBoxIndexLeaf, 
             d_pos, d_Mnm, boxMin, rootBoxSize, maxLevel);
    
    // M2M: Multipole to Multipole (upward pass)
    for (int level = maxLevel - 1; level >= 2; level--) {
        int numBoxesLevel = 0;  // Number of boxes at this level
        int* d_boxParent;       // Parent indices for boxes at this level
        
        // Allocate and compute parent indices
        CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxParent, numBoxesLevel * sizeof(int)));
        
        // M2M kernel for this level
        m2mKernel(numBoxesLevel, d_boxParent, d_Mnm, d_Mnm, level);
        
        cudaFree(d_boxParent);
    }
}

// Compute forces using FMM
void FMMSystem::computeForces() {
    // M2L: Multipole to Local (across the tree)
    for (int level = 2; level <= maxLevel; level++) {
        m2lKernel(numBoxIndexLeaf, d_interactionList, d_numInteraction, 
                 d_Mnm, d_Lnm, boxMin, rootBoxSize, level);
    }
    
    // L2L: Local to Local (downward pass)
    for (int level = 3; level <= maxLevel; level++) {
        int numBoxesLevel = 0;  // Number of boxes at this level
        int* d_boxChildren;     // Child indices for boxes at this level
        
        // Allocate and compute child indices
        CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxChildren, numBoxesLevel * sizeof(int)));
        
        // L2L kernel for this level
        l2lKernel(numBoxesLevel, d_boxChildren, d_Lnm, d_Lnm, level);
        
        cudaFree(d_boxChildren);
    }
    
    // L2P: Local to Particle
    l2pKernel(numBoxIndexLeaf, d_particleOffset, d_particleOffset + numBoxIndexLeaf, 
             d_Lnm, d_pos, d_acc, boxMin, rootBoxSize, maxLevel);
    
    // M2P: Multipole to Particle (for well-separated boxes not covered by L2P)
    m2pKernel(numBoxIndexLeaf, d_particleOffset, d_particleOffset + numBoxIndexLeaf, 
             d_interactionList, d_numInteraction, d_Mnm, d_pos, d_acc, 
             boxMin, rootBoxSize, maxLevel);
    
    // P2P: Direct calculation for nearby particles
    p2pKernel(numBoxIndexLeaf, d_particleOffset, d_particleOffset + numBoxIndexLeaf, 
             d_interactionList, d_numInteraction, d_pos, d_acc);
}

// Update particle positions and velocities
void FMMSystem::updateParticles(float deltaTime) {
    updateParticlesKernel(numParticles, d_pos, d_vel, d_acc, deltaTime, damping);
}

// Create an FMM system
FMMSystem* createFMMSystem(int numParticles, Pos* h_pos, Vel* h_vel) {
    FMMSystem* system = new FMMSystem();
    system->initialize(numParticles, h_pos, h_vel);
    return system;
}

// Destroy an FMM system
void destroyFMMSystem(FMMSystem* system) {
    if (system) {
        delete system;
    }
} 