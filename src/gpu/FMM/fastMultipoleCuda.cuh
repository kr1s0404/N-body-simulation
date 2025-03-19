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

#ifndef FAST_MULTIPOLE_CUDA_H_
#define FAST_MULTIPOLE_CUDA_H_

// Structure for position and mass
typedef struct {
    float x;
    float y;
    float z;
    float w;  // mass
} Pos;

// Structure for velocity
typedef struct {
    float x;
    float y;
    float z;
} Vel;

// Structure for acceleration
typedef struct {
    float x;
    float y;
    float z;
} Acc;

// Structure for multipole expansion coefficients
typedef struct {
    float real;
    float imag;
} Complex;

// Structure for FMM cell
typedef struct {
    float x;       // center x
    float y;       // center y
    float z;       // center z
    float size;    // cell size
    int parent;    // parent index
    int children[8]; // indices of children cells
    int numParticles;
    int particleOffset;
} Cell;

// Forward declaration of the FMMSystem class
class FMMSystem;

// Create an FMM system
FMMSystem* createFMMSystem(int numParticles, Pos* positions, Vel* velocities);

// Destroy an FMM system
void destroyFMMSystem(FMMSystem* system);

// FMM System class
class FMMSystem {
private:
    int numParticles;
    int numCells;
    int maxLevel;
    float domainSize;
    
    // Host data
    Pos* h_pos;
    Vel* h_vel;
    Acc* h_acc;
    Cell* h_cells;
    int* h_particleIndices;
    Complex* h_multipoles;
    Complex* h_locals;
    
    // Device data
    Pos* d_pos;
    Vel* d_vel;
    Acc* d_acc;
    Cell* d_cells;
    int* d_particleIndices;
    Complex* d_multipoles;
    Complex* d_locals;
    
    // Parameters
    int p;  // Multipole expansion order
    float theta;  // Multipole acceptance criterion
    
    // Private methods
    void buildTree();
    void computeMultipoles();
    void translateMultipoles();
    void computeLocalExpansions();
    void evaluateLocalExpansions();
    void directInteractions();
    void updateParticles(float dt);

public:
    FMMSystem(int numParticles, Pos* positions, Vel* velocities);
    ~FMMSystem();
    
    void setDomainSize(float size);
    void setMultipoleOrder(int order);
    void setTheta(float theta);
    
    void step(float dt);
    void getPositions(Pos* positions);
    void getVelocities(Vel* velocities);
    
    friend FMMSystem* createFMMSystem(int numParticles, Pos* positions, Vel* velocities);
    friend void destroyFMMSystem(FMMSystem* system);
};

#endif // FAST_MULTIPOLE_CUDA_H_ 