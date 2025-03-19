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
#include <opencv2/opencv.hpp>
#include "fastMultipoleCuda.cuh"
#include "constants.h"
#include "err.h"

// Constants for simulation
#define NUM_PARTICLES 10000
#define NUM_STEPS 1000
#define TIME_STEP 0.01f
#define RENDER_INTERVAL 10
#define DOMAIN_SIZE 10.0f
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024

// Function to initialize particles in a spiral pattern
void initSpiralParticles(Pos* positions, Vel* velocities, int numParticles) {
    float centerX = DOMAIN_SIZE / 2.0f;
    float centerY = DOMAIN_SIZE / 2.0f;
    float centerZ = DOMAIN_SIZE / 2.0f;
    
    // Central massive body
    positions[0].x = centerX;
    positions[0].y = centerY;
    positions[0].z = centerZ;
    positions[0].w = 1000.0f;  // Mass
    
    velocities[0].x = 0.0f;
    velocities[0].y = 0.0f;
    velocities[0].z = 0.0f;
    
    // Spiral particles
    for (int i = 1; i < numParticles; i++) {
        float angle = 0.1f * i;
        float radius = 0.1f + 0.01f * i;
        float height = 0.1f * sinf(angle * 0.1f);
        
        positions[i].x = centerX + radius * cosf(angle);
        positions[i].y = centerY + radius * sinf(angle);
        positions[i].z = centerZ + height;
        positions[i].w = 0.1f;  // Mass
        
        // Orbital velocity
        float speed = sqrtf(positions[0].w / radius) * 0.1f;
        velocities[i].x = -speed * sinf(angle);
        velocities[i].y = speed * cosf(angle);
        velocities[i].z = 0.0f;
    }
}

// Function to render particles to an OpenCV frame
void renderParticles(cv::Mat& frame, Pos* positions, int numParticles) {
    // Clear frame
    frame = cv::Scalar(0, 0, 0);
    
    // Draw particles
    for (int i = 0; i < numParticles; i++) {
        // Map particle position to screen coordinates
        int x = (int)((positions[i].x / DOMAIN_SIZE) * WINDOW_WIDTH);
        int y = (int)((positions[i].y / DOMAIN_SIZE) * WINDOW_HEIGHT);
        
        // Ensure coordinates are within frame bounds
        if (x >= 0 && x < WINDOW_WIDTH && y >= 0 && y < WINDOW_HEIGHT) {
            // Draw particle (size and color based on mass)
            float mass = positions[i].w;
            int radius = std::max(1, (int)(mass * 0.1f));
            cv::Scalar color;
            
            if (i == 0) {
                // Central body is yellow
                color = cv::Scalar(0, 255, 255);
            } else {
                // Other particles are white with blue tint based on z-coordinate
                int blue = 128 + (int)(127.0f * (positions[i].z / DOMAIN_SIZE));
                color = cv::Scalar(blue, blue, 255);
            }
            
            cv::circle(frame, cv::Point(x, y), radius, color, -1);
        }
    }
}

// Function to check command line arguments
bool checkArgs(int numParticles, int numSteps) {
    if (numParticles < 1) {
        std::cout << "ERROR: need to have at least 1 particle" << std::endl;
        return false;
    }
    
    if (numSteps < 1) {
        std::cout << "ERROR: need to have at least 1 step" << std::endl;
        return false;
    }
    
    return true;
}

int main(int argc, char** argv) {
    // Initialize CUDA
    cudaFree(0);
    
    // Parse command line arguments
    int numParticles = NUM_PARTICLES;
    int numSteps = NUM_STEPS;
    
    if (argc == 3) {
        numParticles = atoi(argv[1]);
        numSteps = atoi(argv[2]);
    }
    
    if (!checkArgs(numParticles, numSteps)) {
        return -1;
    }
    
    // Allocate host memory for particles
    Pos* h_pos = new Pos[numParticles];
    Vel* h_vel = new Vel[numParticles];
    
    // Initialize particles
    initSpiralParticles(h_pos, h_vel, numParticles);
    
    // Create FMM system
    FMMSystem* fmmSystem = createFMMSystem(numParticles, h_pos, h_vel);
    
    // Set domain size
    fmmSystem->setDomainSize(DOMAIN_SIZE);
    
    // Create video writer
    cv::VideoWriter video("nbody_fmm.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, 
                         cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));
    
    // Create window for display
    cv::namedWindow("N-body Simulation (FMM)", cv::WINDOW_NORMAL);
    cv::resizeWindow("N-body Simulation (FMM)", WINDOW_WIDTH, WINDOW_HEIGHT);
    
    // Frame for rendering
    cv::Mat frame(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    
    // Main simulation loop
    for (int step = 0; step < numSteps; step++) {
        // Perform simulation step
        fmmSystem->step(TIME_STEP);
        
        // Render every RENDER_INTERVAL steps
        if (step % RENDER_INTERVAL == 0) {
            // Get updated particle positions
            fmmSystem->getPositions(h_pos);
            
            // Render particles
            renderParticles(frame, h_pos, numParticles);
            
            // Display frame
            cv::imshow("N-body Simulation (FMM)", frame);
            
            // Write frame to video
            video.write(frame);
            
            // Process keyboard input (ESC to exit)
            int key = cv::waitKey(1);
            if (key == 27) break;
            
            // Print progress
            printf("Step %d/%d\n", step, numSteps);
        }
    }
    
    // Clean up
    video.release();
    cv::destroyAllWindows();
    
    destroyFMMSystem(fmmSystem);
    delete[] h_pos;
    delete[] h_vel;
    
    return 0;
} 