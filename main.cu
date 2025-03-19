#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "fastMultipoleCuda.cuh"
#include "err.h"
#include "constants.h"

// Constants for simulation
const int NUM_PARTICLES = 10000;
const float DOMAIN_SIZE = 10.0f;
const float TIME_STEP = 0.01f;
const int NUM_STEPS = 1000;
const int RENDER_INTERVAL = 10;

// Constants for rendering
const int WINDOW_WIDTH = 1024;
const int WINDOW_HEIGHT = 768;
const float PARTICLE_RADIUS = 2.0f;
const cv::Scalar PARTICLE_COLOR(0, 255, 0);
const cv::Scalar SUN_COLOR(255, 255, 0);

// Function to initialize particles in a spiral pattern
void initSpiralParticles(Pos* h_pos, Vel* h_vel, int numParticles) {
    // Create a central "sun" particle
    h_pos[0].x = 0.0f;
    h_pos[0].y = 0.0f;
    h_pos[0].z = 0.0f;
    h_pos[0].w = 1000.0f;  // Mass
    
    h_vel[0].x = 0.0f;
    h_vel[0].y = 0.0f;
    h_vel[0].z = 0.0f;
    
    // Create spiral of particles around the sun
    float radius = 0.5f;
    float angle = 0.0f;
    float angleIncrement = 0.1f;
    float radiusIncrement = 0.01f;
    
    for (int i = 1; i < numParticles; i++) {
        // Position in spiral
        h_pos[i].x = radius * cosf(angle);
        h_pos[i].y = radius * sinf(angle);
        h_pos[i].z = 0.0f;
        h_pos[i].w = 1.0f;  // Mass
        
        // Initial velocity perpendicular to radius
        float speed = sqrtf(G * h_pos[0].w / radius);
        h_vel[i].x = -speed * sinf(angle);
        h_vel[i].y = speed * cosf(angle);
        h_vel[i].z = 0.0f;
        
        // Increment for next particle
        angle += angleIncrement;
        radius += radiusIncrement;
    }
}

// Function to initialize particles in a random distribution
void initRandomParticles(Pos* h_pos, Vel* h_vel, int numParticles) {
    srand(time(NULL));
    
    // Create a central "sun" particle
    h_pos[0].x = 0.0f;
    h_pos[0].y = 0.0f;
    h_pos[0].z = 0.0f;
    h_pos[0].w = 1000.0f;  // Mass
    
    h_vel[0].x = 0.0f;
    h_vel[0].y = 0.0f;
    h_vel[0].z = 0.0f;
    
    // Create random particles
    for (int i = 1; i < numParticles; i++) {
        // Random position within domain
        float r = (DOMAIN_SIZE / 4.0f) * sqrtf((float)rand() / RAND_MAX);
        float angle = 2.0f * M_PI * ((float)rand() / RAND_MAX);
        
        h_pos[i].x = r * cosf(angle);
        h_pos[i].y = r * sinf(angle);
        h_pos[i].z = 0.0f;
        h_pos[i].w = 1.0f;  // Mass
        
        // Initial velocity perpendicular to radius
        float speed = sqrtf(G * h_pos[0].w / r) * (0.8f + 0.4f * ((float)rand() / RAND_MAX));
        h_vel[i].x = -speed * sinf(angle);
        h_vel[i].y = speed * cosf(angle);
        h_vel[i].z = 0.0f;
    }
}

// Function to render particles
void renderParticles(cv::Mat& frame, Pos* h_pos, int numParticles) {
    // Clear frame
    frame = cv::Scalar(0, 0, 0);
    
    // Scale factor to map from simulation space to pixel space
    float scale = fminf(WINDOW_WIDTH, WINDOW_HEIGHT) / (DOMAIN_SIZE * 1.5f);
    float offsetX = WINDOW_WIDTH / 2.0f;
    float offsetY = WINDOW_HEIGHT / 2.0f;
    
    // Draw each particle
    for (int i = 0; i < numParticles; i++) {
        int x = (int)(h_pos[i].x * scale + offsetX);
        int y = (int)(h_pos[i].y * scale + offsetY);
        
        // Skip particles outside the visible area
        if (x < 0 || x >= WINDOW_WIDTH || y < 0 || y >= WINDOW_HEIGHT)
            continue;
        
        // Draw particle (size based on mass)
        float radius = i == 0 ? PARTICLE_RADIUS * 5.0f : PARTICLE_RADIUS;
        cv::Scalar color = i == 0 ? SUN_COLOR : PARTICLE_COLOR;
        
        cv::circle(frame, cv::Point(x, y), radius, color, -1);
    }
}

int main() {
    // Initialize CUDA
    cudaFree(0);
    
    // Allocate host memory for particles
    Pos* h_pos = new Pos[NUM_PARTICLES];
    Vel* h_vel = new Vel[NUM_PARTICLES];
    
    // Initialize particles
    initSpiralParticles(h_pos, h_vel, NUM_PARTICLES);
    
    // Create FMM system
    FMMSystem* fmmSystem = createFMMSystem(NUM_PARTICLES, h_pos, h_vel);
    
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
    for (int step = 0; step < NUM_STEPS; step++) {
        // Perform simulation step
        fmmSystem->step(TIME_STEP);
        
        // Render every RENDER_INTERVAL steps
        if (step % RENDER_INTERVAL == 0) {
            // Get updated particle positions
            fmmSystem->getPositions(h_pos);
            
            // Render particles
            renderParticles(frame, h_pos, NUM_PARTICLES);
            
            // Display frame
            cv::imshow("N-body Simulation (FMM)", frame);
            
            // Write frame to video
            video.write(frame);
            
            // Process keyboard input (ESC to exit)
            int key = cv::waitKey(1);
            if (key == 27) break;
            
            // Print progress
            printf("Step %d/%d\n", step, NUM_STEPS);
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