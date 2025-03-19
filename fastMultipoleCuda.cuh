#ifndef __FAST_MULTIPOLE_CUDA_H__
#define __FAST_MULTIPOLE_CUDA_H__

#include <cuda_runtime.h>
#include "constants.h"

// Vector types for particle data
typedef float4 Pos;  // x, y, z, mass
typedef float3 Vel;  // vx, vy, vz
typedef float3 Acc;  // ax, ay, az

// 3D vector template for box indices and other integer vectors
template<typename T>
class Vec3 {
public:
    T x, y, z;
};

// FMM system class for managing the multipole algorithm
class FMMSystem {
public:
    // Constructor and destructor
    FMMSystem();
    ~FMMSystem();

    // Initialization
    void initialize(int numParticles, Pos* h_pos, Vel* h_vel);
    void reset();
    
    // Main simulation step
    void step(float deltaTime);
    
    // Data transfer
    void getPositions(Pos* h_pos);
    void getVelocities(Vel* h_vel);
    
    // Tree construction and FMM algorithm
    void buildTree();
    void computeMultipoles();
    void computeForces();
    void updateParticles(float deltaTime);
    
    // Utility functions
    int getNumParticles() const { return numParticles; }
    void setDomainSize(float size);
    void setOptimumLevel(int numParticles);
    
private:
    // Particle data
    int numParticles;
    Pos* d_pos;
    Vel* d_vel;
    Acc* d_acc;
    
    // Tree data
    int maxLevel;
    int numBoxIndexFull;
    int numBoxIndexLeaf;
    int numBoxIndexTotal;
    int* d_boxIndexMask;
    int* d_boxIndexFull;
    int* d_levelOffset;
    int* d_mortonIndex;
    int* d_particleOffset;
    int* d_numInteraction;
    int* d_interactionList;
    
    // Multipole data
    float* d_Mnm;
    float* d_Lnm;
    
    // Domain data
    float rootBoxSize;
    float3 boxMin;
    float damping;
    
    // Helper functions
    void allocateMemory();
    void freeMemory();
    void computeMortonIndices();
    void sortParticles();
    void countNonEmptyBoxes();
    void computeInteractionLists();
};

// Function to initialize CUDA and create FMM system
FMMSystem* createFMMSystem(int numParticles, Pos* h_pos, Vel* h_vel);

// Function to destroy FMM system
void destroyFMMSystem(FMMSystem* system);

#endif // __FAST_MULTIPOLE_CUDA_H__ 