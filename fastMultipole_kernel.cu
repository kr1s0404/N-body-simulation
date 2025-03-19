#include "fastMultipole_kernel.cuh"
#include "err.h"
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

// Helper functions for spherical harmonics and FMM operations
__device__ void cart2sph(float &r, float &theta, float &phi, float x, float y, float z) {
    r = sqrtf(x*x + y*y + z*z);
    if (r < eps) {
        theta = 0.0f;
        phi = 0.0f;
        return;
    }
    theta = acosf(z/r);
    phi = atan2f(y, x);
}

__device__ void sph2cart(float r, float theta, float phi, float &x, float &y, float &z) {
    x = r * sinf(theta) * cosf(phi);
    y = r * sinf(theta) * sinf(phi);
    z = r * cosf(theta);
}

// Helper function to convert 3D position to Morton code
__device__ unsigned int morton3D(float x, float y, float z, float boxMin_x, float boxMin_y, float boxMin_z, float boxSize, int maxLevel) {
    // Scale to [0,1]
    x = (x - boxMin_x) / boxSize;
    y = (y - boxMin_y) / boxSize;
    z = (z - boxMin_z) / boxSize;
    
    // Clamp to [0,1]
    x = fmaxf(0.0f, fminf(1.0f, x));
    y = fmaxf(0.0f, fminf(1.0f, y));
    z = fmaxf(0.0f, fminf(1.0f, z));
    
    // Scale to [0, 2^maxLevel - 1]
    int scale = (1 << maxLevel) - 1;
    unsigned int xi = (unsigned int)(x * scale);
    unsigned int yi = (unsigned int)(y * scale);
    unsigned int zi = (unsigned int)(z * scale);
    
    // Interleave bits using lookup table or bit manipulation
    // This is a simplified version - a full implementation would use a more efficient method
    unsigned int result = 0;
    for (int i = 0; i < maxLevel; i++) {
        unsigned int mask = 1 << i;
        result |= ((xi & mask) << (2*i)) | ((yi & mask) << (2*i + 1)) | ((zi & mask) << (2*i + 2));
    }
    
    return result;
}

// Kernel to compute Morton indices for all particles
__global__ void computeMortonIndicesKernel_impl(int numParticles, float4* pos, int* mortonIndex, 
                                              float3 boxMin, float boxSize, int maxLevel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        mortonIndex[i] = morton3D(pos[i].x, pos[i].y, pos[i].z, 
                                 boxMin.x, boxMin.y, boxMin.z, 
                                 boxSize, maxLevel);
    }
}

// Host function to launch Morton index computation kernel
void computeMortonIndicesKernel(int numParticles, float4* d_pos, int* d_mortonIndex, 
                               float3 boxMin, float boxSize, int maxLevel) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    
    computeMortonIndicesKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numParticles, d_pos, d_mortonIndex, boxMin, boxSize, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// Kernel to reorder particles based on sorted indices
__global__ void reorderParticlesKernel(int numParticles, float4* d_pos_in, float3* d_vel_in, 
                                      float4* d_pos_out, float3* d_vel_out, int* d_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numParticles) {
        int sortedIndex = d_indices[i];
        d_pos_out[i] = d_pos_in[sortedIndex];
        d_vel_out[i] = d_vel_in[sortedIndex];
    }
}

// Host function to sort particles by Morton index
void sortParticlesKernel(int numParticles, float4* d_pos, float3* d_vel, 
                        int* d_mortonIndex, int* d_sortedIndex) {
    // Use Thrust for sorting
    thrust::device_ptr<int> thrust_keys(d_mortonIndex);
    
    // Create temporary arrays for sorting
    float4* d_pos_temp;
    float3* d_vel_temp;
    int* d_indices;
    
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_pos_temp, numParticles * sizeof(float4)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_vel_temp, numParticles * sizeof(float3)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_indices, numParticles * sizeof(int)));
    
    // Initialize indices
    thrust::device_ptr<int> thrust_indices(d_indices);
    thrust::sequence(thrust_indices, thrust_indices + numParticles);
    
    // Sort indices by Morton code
    thrust::sort_by_key(thrust_keys, thrust_keys + numParticles, thrust_indices);
    
    // Reorder particle data based on sorted indices
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    
    reorderParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        numParticles, d_pos, d_vel, d_pos_temp, d_vel_temp, d_indices);
    
    // Copy back to original arrays
    CUDA_CHECK_ERROR(cudaMemcpy(d_pos, d_pos_temp, numParticles * sizeof(float4), cudaMemcpyDeviceToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_vel, d_vel_temp, numParticles * sizeof(float3), cudaMemcpyDeviceToDevice));
    
    // If we need to keep track of the original indices
    if (d_sortedIndex) {
        CUDA_CHECK_ERROR(cudaMemcpy(d_sortedIndex, d_indices, numParticles * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    // Clean up
    cudaFree(d_pos_temp);
    cudaFree(d_vel_temp);
    cudaFree(d_indices);
    
    CUDA_CHECK_LAST_ERROR();
}

// Kernel to count non-empty boxes and set up box data
__global__ void countBoxesKernel_impl(int numParticles, int* mortonIndex, int* boxCount, 
                                     int* boxStart, int* boxEnd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        int currentBox = mortonIndex[i];
        
        // If this is the first particle or it's in a different box than the previous particle
        if (i == 0 || currentBox != mortonIndex[i-1]) {
            boxStart[currentBox] = i;
            
            // Increment box count (atomic to avoid race conditions)
            atomicAdd(boxCount, 1);
            
            // If not the first particle, set end of previous box
            if (i > 0) {
                boxEnd[mortonIndex[i-1]] = i - 1;
            }
        }
        
        // If this is the last particle, set end of last box
        if (i == numParticles - 1) {
            boxEnd[currentBox] = i;
        }
    }
}

// Host function to launch box counting kernel
void countBoxesKernel(int numParticles, int* d_mortonIndex, int* d_boxCount, 
                     int* d_boxStart, int maxLevel) {
    // Initialize box count to 0
    CUDA_CHECK_ERROR(cudaMemset(d_boxCount, 0, sizeof(int)));
    
    // Initialize box start and end indices to -1 (indicating empty box)
    int numPossibleBoxes = 1 << (3 * maxLevel);  // 8^maxLevel
    CUDA_CHECK_ERROR(cudaMemset(d_boxStart, -1, numPossibleBoxes * sizeof(int)));
    
    int* d_boxEnd;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_boxEnd, numPossibleBoxes * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset(d_boxEnd, -1, numPossibleBoxes * sizeof(int)));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    
    countBoxesKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numParticles, d_mortonIndex, d_boxCount, d_boxStart, d_boxEnd);
    
    // Clean up
    cudaFree(d_boxEnd);
    
    CUDA_CHECK_LAST_ERROR();
}

// Kernel to build interaction lists for each box
__global__ void buildInteractionListsKernel_impl(int numBoxes, int* boxIndexFull, 
                                               int* interactionLists, int* numInteractions, 
                                               int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx < numBoxes) {
        int mortonCode = boxIndexFull[boxIdx];
        int x, y, z;
        
        // Extract x, y, z coordinates from Morton code
        // This is a simplified version - a full implementation would use a more efficient method
        x = y = z = 0;
        for (int i = 0; i < maxLevel; i++) {
            x |= ((mortonCode >> (3*i)) & 1) << i;
            y |= ((mortonCode >> (3*i+1)) & 1) << i;
            z |= ((mortonCode >> (3*i+2)) & 1) << i;
        }
        
        // Find interaction list (boxes that are well-separated)
        int interactionCount = 0;
        
        // For each potential neighbor box
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    // Skip the box itself and its immediate neighbors
                    if (abs(dx) + abs(dy) + abs(dz) <= 1) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;
                    
                    // Check if neighbor is within bounds
                    if (nx >= 0 && nx < (1 << maxLevel) &&
                        ny >= 0 && ny < (1 << maxLevel) &&
                        nz >= 0 && nz < (1 << maxLevel)) {
                        
                        // Compute Morton code of neighbor
                        int neighborCode = 0;
                        for (int i = 0; i < maxLevel; i++) {
                            neighborCode |= ((nx >> i) & 1) << (3*i);
                            neighborCode |= ((ny >> i) & 1) << (3*i+1);
                            neighborCode |= ((nz >> i) & 1) << (3*i+2);
                        }
                        
                        // Add to interaction list
                        if (interactionCount < maxM2LInteraction) {
                            interactionLists[boxIdx * maxM2LInteraction + interactionCount] = neighborCode;
                            interactionCount++;
                        }
                    }
                }
            }
        }
        
        numInteractions[boxIdx] = interactionCount;
    }
}

// Host function to launch interaction list building kernel
void buildInteractionListsKernel(int numBoxes, int* d_boxIndexFull, int* d_interactionLists, 
                               int* d_numInteractions, int maxLevel) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    buildInteractionListsKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxIndexFull, d_interactionLists, d_numInteractions, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// P2P kernel (particle to particle) for direct force calculation between nearby particles
__global__ void p2pKernel_impl(int numBoxes, int* boxStart, int* boxEnd, int* interactionLists, 
                             int* numInteractions, float4* pos, float3* acc) {
    int boxIdx = blockIdx.x;
    int particleIdx = threadIdx.x;
    
    // Shared memory for particle data in current box
    __shared__ float4 sharedPos[threadsPerBlockTypeA];
    
    if (boxIdx < numBoxes) {
        int start = boxStart[boxIdx];
        int end = boxEnd[boxIdx];
        int numParticles = end - start + 1;
        
        // Load particle data for current box into shared memory
        if (particleIdx < numParticles) {
            sharedPos[particleIdx] = pos[start + particleIdx];
        }
        __syncthreads();
        
        // Process each particle in the current box
        if (particleIdx < numParticles) {
            float4 myPos = sharedPos[particleIdx];
            float3 myAcc = make_float3(0.0f, 0.0f, 0.0f);
            
            // Process interactions with particles in the same box
            for (int j = 0; j < numParticles; j++) {
                if (j != particleIdx) {
                    float4 otherPos = sharedPos[j];
                    
                    float dx = otherPos.x - myPos.x;
                    float dy = otherPos.y - myPos.y;
                    float dz = otherPos.z - myPos.z;
                    
                    float distSqr = dx*dx + dy*dy + dz*dz + softening;
                    float invDist = rsqrtf(distSqr);
                    float invDistCube = invDist * invDist * invDist;
                    
                    float s = G * otherPos.w * invDistCube;
                    
                    myAcc.x += dx * s;
                    myAcc.y += dy * s;
                    myAcc.z += dz * s;
                }
            }
            
            // Process interactions with particles in neighboring boxes
            int numNeighbors = numInteractions[boxIdx];
            for (int n = 0; n < numNeighbors; n++) {
                int neighborCode = interactionLists[boxIdx * maxM2LInteraction + n];
                
                // Find the box index for this Morton code
                // This is a simplified approach - in a real implementation, you'd use a lookup table
                int neighborBoxIdx = -1;
                for (int b = 0; b < numBoxes; b++) {
                    if (interactionLists[b] == neighborCode) {
                        neighborBoxIdx = b;
                        break;
                    }
                }
                
                if (neighborBoxIdx >= 0) {
                    int nStart = boxStart[neighborBoxIdx];
                    int nEnd = boxEnd[neighborBoxIdx];
                    
                    // Process each particle in the neighboring box
                    for (int j = nStart; j <= nEnd; j++) {
                        float4 otherPos = pos[j];
                        
                        float dx = otherPos.x - myPos.x;
                        float dy = otherPos.y - myPos.y;
                        float dz = otherPos.z - myPos.z;
                        
                        float distSqr = dx*dx + dy*dy + dz*dz + softening;
                        float invDist = rsqrtf(distSqr);
                        float invDistCube = invDist * invDist * invDist;
                        
                        float s = G * otherPos.w * invDistCube;
                        
                        myAcc.x += dx * s;
                        myAcc.y += dy * s;
                        myAcc.z += dz * s;
                    }
                }
            }
            
            // Write accumulated acceleration back to global memory
            atomicAdd(&acc[start + particleIdx].x, myAcc.x);
            atomicAdd(&acc[start + particleIdx].y, myAcc.y);
            atomicAdd(&acc[start + particleIdx].z, myAcc.z);
        }
    }
}

// Host function to launch P2P kernel
void p2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, int* d_interactionLists, 
              int* d_numInteractions, float4* d_pos, float3* d_acc) {
    p2pKernel_impl<<<numBoxes, threadsPerBlockTypeA>>>(
        numBoxes, d_boxStart, d_boxEnd, d_interactionLists, d_numInteractions, d_pos, d_acc);
    
    CUDA_CHECK_LAST_ERROR();
}

// P2M kernel (particle to multipole)
__global__ void p2mKernel_impl(int numBoxes, int* boxStart, int* boxEnd, float4* pos, 
                             float* Mnm, float3 boxMin, float boxSize, int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    // Box center calculation
    unsigned int mortonCode = boxIdx; // Assuming boxIdx is the Morton code
    float3 boxCenter;
    
    // Decode Morton code to get 3D box index
    unsigned int x = 0, y = 0, z = 0;
    for (unsigned int i = 0; i < maxLevel; ++i) {
        x |= ((mortonCode & (1u << (3*i))) >> (2*i));
        y |= ((mortonCode & (1u << (3*i+1))) >> (2*i+1));
        z |= ((mortonCode & (1u << (3*i+2))) >> (2*i+2));
    }
    
    // Calculate box center
    float levelBoxSize = boxSize / (1 << maxLevel);
    boxCenter.x = boxMin.x + (x + 0.5f) * levelBoxSize;
    boxCenter.y = boxMin.y + (y + 0.5f) * levelBoxSize;
    boxCenter.z = boxMin.z + (z + 0.5f) * levelBoxSize;
    
    // Initialize multipole coefficients to zero
    for (int j = 0; j < numCoefficients; j++) {
        Mnm[boxIdx * numCoefficients + j] = 0.0f;
    }
    
    // Process all particles in this box
    for (int j = boxStart[boxIdx]; j <= boxEnd[boxIdx]; j++) {
        if (j < 0) continue; // Skip if no particles
        
        // Calculate relative position
        float dx = pos[j].x - boxCenter.x;
        float dy = pos[j].y - boxCenter.y;
        float dz = pos[j].z - boxCenter.z;
        
        // Convert to spherical coordinates
        float r, theta, phi;
        cart2sph(r, theta, phi, dx, dy, dz);
        
        // Calculate spherical harmonics
        float xx = cosf(theta);
        float s2 = sqrtf((1.0f - xx) * (1.0f + xx));
        float fact = 1.0f;
        float pn = 1.0f;
        float rhom = 1.0f;
        
        // Temporary storage for Ynm values
        float YnmReal[numExpansion2];
        
        // Calculate Ynm for all required n,m
        for (int m = 0; m < numExpansions; m++) {
            float p = pn;
            int nm = m*m + 2*m;
            YnmReal[nm] = rhom * factorial[nm] * p;
            
            float p1 = p;
            p = xx * (2*m + 1) * p;
            rhom *= r;
            float rhon = rhom;
            
            for (int n = m+1; n < numExpansions; n++) {
                nm = n*n + n + m;
                YnmReal[nm] = rhon * factorial[nm] * p;
                
                float p2 = p1;
                p1 = p;
                p = (xx * (2*n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                rhon *= r;
            }
            
            pn = -pn * fact * s2;
            fact += 2.0f;
        }
        
        // Calculate multipole coefficients
        for (int n = 0; n < numExpansions; n++) {
            for (int m = 0; m <= n; m++) {
                int nm = n*n + n + m;
                int nms = n*(n+1)/2 + m;
                
                // Complex exponential
                float sinmPhi, cosmPhi;
                sincosf(m * phi, &sinmPhi, &cosmPhi);
                
                // Update real and imaginary parts of multipole coefficient
                float mass = pos[j].w;
                atomicAdd(&Mnm[boxIdx * numCoefficients * 2 + nms * 2], 
                         mass * YnmReal[nm] * cosmPhi);
                atomicAdd(&Mnm[boxIdx * numCoefficients * 2 + nms * 2 + 1], 
                         -mass * YnmReal[nm] * sinmPhi);
            }
        }
    }
}

// Host function to launch P2M kernel
void p2mKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, float4* d_pos, 
              float* d_Mnm, float3 boxMin, float boxSize, int maxLevel) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    // Precompute factorial values on device
    static bool factorialsComputed = false;
    static float* d_factorial = nullptr;
    
    if (!factorialsComputed) {
        float h_factorial[numExpansion2];
        for (int i = 0; i < numExpansion2; i++) {
            h_factorial[i] = 1.0f;
            for (int j = 1; j <= i; j++) {
                h_factorial[i] *= j;
            }
            h_factorial[i] = 1.0f / h_factorial[i];
        }
        
        CUDA_CHECK_ERROR(cudaMalloc((void**)&d_factorial, numExpansion2 * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_factorial, h_factorial, numExpansion2 * sizeof(float), 
                                   cudaMemcpyHostToDevice));
        factorialsComputed = true;
    }
    
    // Define factorial as a device constant
    cudaMemcpyToSymbol(factorial, d_factorial, numExpansion2 * sizeof(float));
    
    p2mKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxStart, d_boxEnd, d_pos, d_Mnm, boxMin, boxSize, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// M2M kernel (multipole to multipole)
__global__ void m2mKernel_impl(int numBoxes, int* boxParent, float* Mnm, float* MnmParent, 
                             int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    int parentIdx = boxParent[boxIdx];
    if (parentIdx < 0) return; // Skip if no parent
    
    // Determine relative position of child box within parent
    int childPos = boxIdx % 8; // Assuming 8 children per parent in octree
    
    // Calculate translation vector based on child position
    float dx = ((childPos & 1) ? 0.25f : -0.25f);
    float dy = ((childPos & 2) ? 0.25f : -0.25f);
    float dz = ((childPos & 4) ? 0.25f : -0.25f);
    
    // Convert to spherical coordinates
    float r, theta, phi;
    cart2sph(r, theta, phi, dx, dy, dz);
    
    // Precomputed rotation and translation matrices would be used here
    // This is a simplified version - a full implementation would use more efficient methods
    
    for (int n = 0; n < numExpansions; n++) {
        for (int m = 0; m <= n; m++) {
            int nms = n*(n+1)/2 + m;
            float mnmReal = 0.0f;
            float mnmImag = 0.0f;
            
            for (int j = 0; j <= n; j++) {
                for (int k = -j; k <= j; k++) {
                    if (abs(m-k) <= n-j) {
                        // Calculate index in child's multipole expansion
                        int jks = j*(j+1)/2 + abs(k);
                        
                        // Get child's multipole coefficient
                        float childReal = Mnm[boxIdx * numCoefficients * 2 + jks * 2];
                        float childImag = Mnm[boxIdx * numCoefficients * 2 + jks * 2 + 1];
                        if (k < 0) childImag = -childImag;
                        
                        // Calculate translation coefficient (simplified)
                        float transCoeff = powf(r, j) * factorial[j+abs(k)] / factorial[j-abs(k)];
                        
                        // Calculate spherical harmonic for translation
                        float Ynm = powf(r, n-j) * factorial[n+m] / factorial[n-m];
                        
                        // Accumulate contribution
                        mnmReal += transCoeff * Ynm * childReal;
                        mnmImag += transCoeff * Ynm * childImag;
                    }
                }
            }
            
            // Update parent's multipole coefficient
            atomicAdd(&MnmParent[parentIdx * numCoefficients * 2 + nms * 2], mnmReal);
            atomicAdd(&MnmParent[parentIdx * numCoefficients * 2 + nms * 2 + 1], mnmImag);
        }
    }
}

// Host function to launch M2M kernel
void m2mKernel(int numBoxes, int* d_boxParent, float* d_Mnm, float* d_MnmParent, 
              int maxLevel) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    m2mKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxParent, d_Mnm, d_MnmParent, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// M2L kernel (multipole to local)
__global__ void m2lKernel_impl(int numBoxes, int* interactionLists, int* numInteractions, 
                             float* Mnm, float* Lnm, float3 boxMin, float boxSize, int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    // Get number of interactions for this box
    int numInter = numInteractions[boxIdx];
    
    // Box center calculation
    unsigned int mortonCode = boxIdx; // Assuming boxIdx is the Morton code
    float3 boxCenter;
    
    // Decode Morton code to get 3D box index
    unsigned int x = 0, y = 0, z = 0;
    for (unsigned int i = 0; i < maxLevel; ++i) {
        x |= ((mortonCode & (1u << (3*i))) >> (2*i));
        y |= ((mortonCode & (1u << (3*i+1))) >> (2*i+1));
        z |= ((mortonCode & (1u << (3*i+2))) >> (2*i+2));
    }
    
    // Calculate box center
    float levelBoxSize = boxSize / (1 << maxLevel);
    boxCenter.x = boxMin.x + (x + 0.5f) * levelBoxSize;
    boxCenter.y = boxMin.y + (y + 0.5f) * levelBoxSize;
    boxCenter.z = boxMin.z + (z + 0.5f) * levelBoxSize;
    
    // Process all interaction boxes
    for (int i = 0; i < numInter; i++) {
        int sourceIdx = interactionLists[boxIdx * maxM2LInteraction + i];
        if (sourceIdx < 0) continue;
        
        // Source box center calculation
        unsigned int sourceMortonCode = sourceIdx;
        unsigned int sx = 0, sy = 0, sz = 0;
        for (unsigned int j = 0; j < maxLevel; ++j) {
            sx |= ((sourceMortonCode & (1u << (3*j))) >> (2*j));
            sy |= ((sourceMortonCode & (1u << (3*j+1))) >> (2*j+1));
            sz |= ((sourceMortonCode & (1u << (3*j+2))) >> (2*j+2));
        }
        
        float3 sourceCenter;
        sourceCenter.x = boxMin.x + (sx + 0.5f) * levelBoxSize;
        sourceCenter.y = boxMin.y + (sy + 0.5f) * levelBoxSize;
        sourceCenter.z = boxMin.z + (sz + 0.5f) * levelBoxSize;
        
        // Calculate relative position
        float dx = sourceCenter.x - boxCenter.x;
        float dy = sourceCenter.y - boxCenter.y;
        float dz = sourceCenter.z - boxCenter.z;
        
        // Convert to spherical coordinates
        float r, theta, phi;
        cart2sph(r, theta, phi, dx, dy, dz);
        
        // M2L translation
        for (int n = 0; n < numExpansions; n++) {
            for (int m = 0; m <= n; m++) {
                int nms = n*(n+1)/2 + m;
                float lnmReal = 0.0f;
                float lnmImag = 0.0f;
                
                for (int j = 0; j < numExpansions; j++) {
                    for (int k = 0; k <= j; k++) {
                        int jks = j*(j+1)/2 + k;
                        
                        // Get source multipole coefficient
                        float sourceReal = Mnm[sourceIdx * numCoefficients * 2 + jks * 2];
                        float sourceImag = Mnm[sourceIdx * numCoefficients * 2 + jks * 2 + 1];
                        
                        // Calculate M2L translation coefficient (simplified)
                        float transCoeff = powf(-1.0f, j) / powf(r, j+n+1);
                        
                        // Calculate spherical harmonic for translation
                        float Ynm = factorial[j+k] * factorial[n+m] / (factorial[j-k] * factorial[n-m]);
                        
                        // Calculate rotation based on theta and phi
                        float rotReal, rotImag;
                        sincosf((k-m) * phi, &rotImag, &rotReal);
                        
                        // Accumulate contribution
                        lnmReal += transCoeff * Ynm * (sourceReal * rotReal - sourceImag * rotImag);
                        lnmImag += transCoeff * Ynm * (sourceReal * rotImag + sourceImag * rotReal);
                    }
                }
                
                // Update local expansion coefficient
                atomicAdd(&Lnm[boxIdx * numCoefficients * 2 + nms * 2], lnmReal);
                atomicAdd(&Lnm[boxIdx * numCoefficients * 2 + nms * 2 + 1], lnmImag);
            }
        }
    }
}

// Host function to launch M2L kernel
void m2lKernel(int numBoxes, int* d_interactionLists, int* d_numInteractions, 
              float* d_Mnm, float* d_Lnm, float3 boxMin, float boxSize, int maxLevel) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    m2lKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_interactionLists, d_numInteractions, d_Mnm, d_Lnm, 
        boxMin, boxSize, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// L2L kernel (local to local)
__global__ void l2lKernel_impl(int numBoxes, int* boxChildren, float* Lnm, float* LnmChildren, 
                             int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    int childIdx = boxChildren[boxIdx];
    if (childIdx < 0) return; // Skip if no children
    
    // Determine relative position of child box within parent
    int childPos = childIdx % 8; // Assuming 8 children per parent in octree
    
    // Calculate translation vector based on child position
    float dx = ((childPos & 1) ? 0.25f : -0.25f);
    float dy = ((childPos & 2) ? 0.25f : -0.25f);
    float dz = ((childPos & 4) ? 0.25f : -0.25f);
    
    // Convert to spherical coordinates
    float r, theta, phi;
    cart2sph(r, theta, phi, dx, dy, dz);
    
    // L2L translation
    for (int n = 0; n < numExpansions; n++) {
        for (int m = 0; m <= n; m++) {
            int nms = n*(n+1)/2 + m;
            float lnmReal = 0.0f;
            float lnmImag = 0.0f;
            
            for (int j = n; j < numExpansions; j++) {
                for (int k = -j; k <= j; k++) {
                    if (abs(k-m) <= j-n) {
                        // Calculate index in parent's local expansion
                        int jks = j*(j+1)/2 + abs(k);
                        
                        // Get parent's local coefficient
                        float parentReal = Lnm[boxIdx * numCoefficients * 2 + jks * 2];
                        float parentImag = Lnm[boxIdx * numCoefficients * 2 + jks * 2 + 1];
                        if (k < 0) parentImag = -parentImag;
                        
                        // Calculate translation coefficient (simplified)
                        float transCoeff = powf(r, j-n) * factorial[j+abs(k)] / factorial[j-abs(k)];
                        
                        // Calculate spherical harmonic for translation
                        float Ynm = factorial[n+m] / factorial[n-m];
                        
                        // Calculate rotation based on theta and phi
                        float rotReal, rotImag;
                        sincosf((k-m) * phi, &rotImag, &rotReal);
                        
                        // Accumulate contribution
                        lnmReal += transCoeff * Ynm * (parentReal * rotReal - parentImag * rotImag);
                        lnmImag += transCoeff * Ynm * (parentReal * rotImag + parentImag * rotReal);
                    }
                }
            }
            
            // Update child's local expansion coefficient
            atomicAdd(&LnmChildren[childIdx * numCoefficients * 2 + nms * 2], lnmReal);
            atomicAdd(&LnmChildren[childIdx * numCoefficients * 2 + nms * 2 + 1], lnmImag);
        }
    }
}

// Host function to launch L2L kernel
void l2lKernel(int numBoxes, int* d_boxChildren, float* d_Lnm, float* d_LnmChildren, 
              int maxLevel) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    l2lKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxChildren, d_Lnm, d_LnmChildren, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// L2P kernel (local to particle)
__global__ void l2pKernel_impl(int numBoxes, int* boxStart, int* boxEnd, float* Lnm, 
                             float4* pos, float3* acc, float3 boxMin, float boxSize, int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    // Box center calculation
    unsigned int mortonCode = boxIdx; // Assuming boxIdx is the Morton code
    float3 boxCenter;
    
    // Decode Morton code to get 3D box index
    unsigned int x = 0, y = 0, z = 0;
    for (unsigned int i = 0; i < maxLevel; ++i) {
        x |= ((mortonCode & (1u << (3*i))) >> (2*i));
        y |= ((mortonCode & (1u << (3*i+1))) >> (2*i+1));
        z |= ((mortonCode & (1u << (3*i+2))) >> (2*i+2));
    }
    
    // Calculate box center
    float levelBoxSize = boxSize / (1 << maxLevel);
    boxCenter.x = boxMin.x + (x + 0.5f) * levelBoxSize;
    boxCenter.y = boxMin.y + (y + 0.5f) * levelBoxSize;
    boxCenter.z = boxMin.z + (z + 0.5f) * levelBoxSize;
    
    // Process all particles in this box
    for (int j = boxStart[boxIdx]; j <= boxEnd[boxIdx]; j++) {
        if (j < 0) continue; // Skip if no particles
        
        // Calculate relative position
        float dx = pos[j].x - boxCenter.x;
        float dy = pos[j].y - boxCenter.y;
        float dz = pos[j].z - boxCenter.z;
        
        // Convert to spherical coordinates
        float r, theta, phi;
        cart2sph(r, theta, phi, dx, dy, dz);
        
        float xx = cosf(theta);
        float yy = sinf(theta);
        float s2 = sqrtf((1.0f - xx) * (1.0f + xx));
        float fact = 1.0f;
        float pn = 1.0f;
        
        // Arrays to store Ynm and its derivative
        float YnmReal[numExpansion2];
        float YnmRealTheta[numExpansion2];
        
        // Calculate Ynm and its derivative for all required n,m
        for (int m = 0; m < numExpansions; m++) {
            float p = pn;
            int nm = m*m + 2*m;
            YnmReal[nm] = factorial[nm] * p;
            
            float p1 = p;
            p = xx * (2*m + 1) * p;
            YnmRealTheta[nm] = factorial[nm] * (p - (m+1) * xx * p1) / yy;
            
            for (int n = m+1; n < numExpansions; n++) {
                nm = n*n + n + m;
                YnmReal[nm] = factorial[nm] * p;
                
                float p2 = p1;
                p1 = p;
                p = (xx * (2*n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                YnmRealTheta[nm] = factorial[nm] * ((n-m+1) * p - (n+1) * xx * p1) / yy;
            }
            
            pn = -pn * fact * s2;
            fact += 2.0f;
        }
        
        // Calculate acceleration components
        float accelR = 0.0f;
        float accelTheta = 0.0f;
        float accelPhi = 0.0f;
        float rn = 1.0f;
        
        for (int n = 0; n < numExpansions; n++) {
            for (int m = 0; m <= n; m++) {
                int nms = n*(n+1)/2 + m;
                
                // Get local expansion coefficient
                float lnmReal = Lnm[boxIdx * numCoefficients * 2 + nms * 2];
                float lnmImag = Lnm[boxIdx * numCoefficients * 2 + nms * 2 + 1];
                
                // Calculate complex exponential
                float sinmPhi, cosmPhi;
                sincosf(m * phi, &sinmPhi, &cosmPhi);
                
                // Calculate contribution to acceleration
                float rr = rn * YnmReal[n*n + n + m];
                float rtheta = rn * YnmRealTheta[n*n + n + m];
                
                // Real part contribution
                accelR += n * rr * (lnmReal * cosmPhi - lnmImag * sinmPhi);
                accelTheta += rtheta * (lnmReal * cosmPhi - lnmImag * sinmPhi);
                
                if (m > 0) {
                    // Imaginary part contribution (only for m > 0)
                    accelPhi += m * rr * (lnmReal * sinmPhi + lnmImag * cosmPhi);
                }
            }
            rn *= r;
        }
        
        // Convert from spherical to Cartesian coordinates
        float ax = sinf(theta) * cosf(phi) * accelR + 
                  cosf(theta) * cosf(phi) / r * accelTheta - 
                  sinf(phi) / (r * sinf(theta)) * accelPhi;
                  
        float ay = sinf(theta) * sinf(phi) * accelR + 
                  cosf(theta) * sinf(phi) / r * accelTheta + 
                  cosf(phi) / (r * sinf(theta)) * accelPhi;
                  
        float az = cosf(theta) * accelR - 
                  sinf(theta) / r * accelTheta;
        
        // Update acceleration
        acc[j].x += inv4PI * ax;
        acc[j].y += inv4PI * ay;
        acc[j].z += inv4PI * az;
    }
}

// Host function to launch L2P kernel
void l2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, float* d_Lnm, 
              float4* d_pos, float3* d_acc, float3 boxMin, float boxSize, int maxLevel) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    l2pKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxStart, d_boxEnd, d_Lnm, d_pos, d_acc, 
        boxMin, boxSize, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// M2P kernel (multipole to particle)
__global__ void m2pKernel_impl(int numBoxes, int* boxStart, int* boxEnd, int* interactionLists, 
                             int* numInteractions, float* Mnm, float4* pos, float3* acc, 
                             float3 boxMin, float boxSize, int maxLevel) {
    int boxIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (boxIdx >= numBoxes) return;
    
    // Get number of interactions for this box
    int numInter = numInteractions[boxIdx];
    
    // Box center calculation
    unsigned int mortonCode = boxIdx; // Assuming boxIdx is the Morton code
    float3 boxCenter;
    
    // Decode Morton code to get 3D box index
    unsigned int x = 0, y = 0, z = 0;
    for (unsigned int i = 0; i < maxLevel; ++i) {
        x |= ((mortonCode & (1u << (3*i))) >> (2*i));
        y |= ((mortonCode & (1u << (3*i+1))) >> (2*i+1));
        z |= ((mortonCode & (1u << (3*i+2))) >> (2*i+2));
    }
    
    // Calculate box center
    float levelBoxSize = boxSize / (1 << maxLevel);
    boxCenter.x = boxMin.x + (x + 0.5f) * levelBoxSize;
    boxCenter.y = boxMin.y + (y + 0.5f) * levelBoxSize;
    boxCenter.z = boxMin.z + (z + 0.5f) * levelBoxSize;
    
    // Process all particles in this box
    for (int i = boxStart[boxIdx]; i <= boxEnd[boxIdx]; i++) {
        if (i < 0) continue; // Skip if no particles
        
        // Process all interaction boxes
        for (int j = 0; j < numInter; j++) {
            int sourceIdx = interactionLists[boxIdx * maxM2LInteraction + j];
            if (sourceIdx < 0) continue;
            
            // Source box center calculation
            unsigned int sourceMortonCode = sourceIdx;
            unsigned int sx = 0, sy = 0, sz = 0;
            for (unsigned int k = 0; k < maxLevel; ++k) {
                sx |= ((sourceMortonCode & (1u << (3*k))) >> (2*k));
                sy |= ((sourceMortonCode & (1u << (3*k+1))) >> (2*k+1));
                sz |= ((sourceMortonCode & (1u << (3*k+2))) >> (2*k+2));
            }
            
            float3 sourceCenter;
            sourceCenter.x = boxMin.x + (sx + 0.5f) * levelBoxSize;
            sourceCenter.y = boxMin.y + (sy + 0.5f) * levelBoxSize;
            sourceCenter.z = boxMin.z + (sz + 0.5f) * levelBoxSize;
            
            // Calculate relative position
            float dx = pos[i].x - sourceCenter.x;
            float dy = pos[i].y - sourceCenter.y;
            float dz = pos[i].z - sourceCenter.z;
            
            // Convert to spherical coordinates
            float r, theta, phi;
            cart2sph(r, theta, phi, dx, dy, dz);
            
            // Calculate spherical harmonics
            float xx = cosf(theta);
            float yy = sinf(theta);
            float s2 = sqrtf((1.0f - xx) * (1.0f + xx));
            float fact = 1.0f;
            float pn = 1.0f;
            
            // Arrays to store Ynm and its derivative
            float YnmReal[numExpansion2];
            float YnmRealTheta[numExpansion2];
            
            // Calculate Ynm and its derivative for all required n,m
            for (int m = 0; m < numExpansions; m++) {
                float p = pn;
                int nm = m*m + 2*m;
                YnmReal[nm] = factorial[nm] * p;
                
                float p1 = p;
                p = xx * (2*m + 1) * p;
                YnmRealTheta[nm] = factorial[nm] * (p - (m+1) * xx * p1) / yy;
                
                for (int n = m+1; n < numExpansions; n++) {
                    nm = n*n + n + m;
                    YnmReal[nm] = factorial[nm] * p;
                    
                    float p2 = p1;
                    p1 = p;
                    p = (xx * (2*n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                    YnmRealTheta[nm] = factorial[nm] * ((n-m+1) * p - (n+1) * xx * p1) / yy;
                }
                
                pn = -pn * fact * s2;
                fact += 2.0f;
            }
            
            // Calculate acceleration components
            float accelR = 0.0f;
            float accelTheta = 0.0f;
            float accelPhi = 0.0f;
            float rn = 1.0f / r;  // 1/r for M2P (inverse of L2P)
            
            for (int n = 0; n < numExpansions; n++) {
                rn *= r;  // r^n for M2P (inverse of L2P which uses 1/r^(n+1))
                
                for (int m = 0; m <= n; m++) {
                    int nms = n*(n+1)/2 + m;
                    
                    // Get multipole expansion coefficient
                    float mnmReal = Mnm[sourceIdx * numCoefficients * 2 + nms * 2];
                    float mnmImag = Mnm[sourceIdx * numCoefficients * 2 + nms * 2 + 1];
                    
                    // Calculate complex exponential
                    float sinmPhi, cosmPhi;
                    sincosf(m * phi, &sinmPhi, &cosmPhi);
                    
                    // Calculate contribution to acceleration
                    float rr = YnmReal[n*n + n + m] / rn;
                    float rtheta = YnmRealTheta[n*n + n + m] / rn;
                    
                    // Real part contribution
                    accelR += (n+1) * rr * (mnmReal * cosmPhi + mnmImag * sinmPhi);
                    accelTheta += rtheta * (mnmReal * cosmPhi + mnmImag * sinmPhi);
                    
                    if (m > 0) {
                        // Imaginary part contribution (only for m > 0)
                        accelPhi -= m * rr * (mnmReal * sinmPhi - mnmImag * cosmPhi);
                    }
                }
            }
            
            // Convert from spherical to Cartesian coordinates
            float ax = sinf(theta) * cosf(phi) * accelR + 
                      cosf(theta) * cosf(phi) / r * accelTheta - 
                      sinf(phi) / (r * sinf(theta)) * accelPhi;
                      
            float ay = sinf(theta) * sinf(phi) * accelR + 
                      cosf(theta) * sinf(phi) / r * accelTheta + 
                      cosf(phi) / (r * sinf(theta)) * accelPhi;
                      
            float az = cosf(theta) * accelR - 
                      sinf(theta) / r * accelTheta;
            
            // Update acceleration
            acc[i].x += inv4PI * ax;
            acc[i].y += inv4PI * ay;
            acc[i].z += inv4PI * az;
        }
    }
}

// Host function to launch M2P kernel
void m2pKernel(int numBoxes, int* d_boxStart, int* d_boxEnd, int* d_interactionLists, 
              int* d_numInteractions, float* d_Mnm, float4* d_pos, float3* d_acc, 
              float3 boxMin, float boxSize, int maxLevel) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (numBoxes + threadsPerBlock - 1) / threadsPerBlock;
    
    m2pKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numBoxes, d_boxStart, d_boxEnd, d_interactionLists, d_numInteractions, 
        d_Mnm, d_pos, d_acc, boxMin, boxSize, maxLevel);
    
    CUDA_CHECK_LAST_ERROR();
}

// Update particles kernel
__global__ void updateParticlesKernel_impl(int numParticles, float4* pos, float3* vel, 
                                         float3* acc, float deltaTime, float damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numParticles) {
        // Update velocity
        vel[i].x += acc[i].x * deltaTime;
        vel[i].y += acc[i].y * deltaTime;
        vel[i].z += acc[i].z * deltaTime;
        
        // Apply damping
        vel[i].x *= damping;
        vel[i].y *= damping;
        vel[i].z *= damping;
        
        // Update position
        pos[i].x += vel[i].x * deltaTime;
        pos[i].y += vel[i].y * deltaTime;
        pos[i].z += vel[i].z * deltaTime;
    }
}

// Host function to launch update particles kernel
void updateParticlesKernel(int numParticles, float4* d_pos, float3* d_vel, 
                          float3* d_acc, float deltaTime, float damping) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    
    updateParticlesKernel_impl<<<blocksPerGrid, threadsPerBlock>>>(
        numParticles, d_pos, d_vel, d_acc, deltaTime, damping);
    
    CUDA_CHECK_LAST_ERROR();
} 