#ifndef DIRECT_SUM_KERNEL_H_
#define DIRECT_SUM_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "constants.h"
#include "err.h"

#define BLOCK_SIZE 256

cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

typedef struct
{
    double x;
    double y;
} Vector;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

} Body;

__device__ double getDistance(Vector pos1, Vector pos2)
{

    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

__device__ bool isCollide(Body &b1, Body &b2)
{
    return b1.radius + b2.radius + COLLISION_TH > getDistance(b1.position, b2.position);
}

__device__ void calculateAcceleration(Body &bi, int i, Body *bodies, int n)
{

    bi.acceleration = {0.0, 0.0};
    for (int b = 0; b < n; ++b)
    {
        if (b != i)
        {
            Body &bj = bodies[b];
            if (!isCollide(bi, bj))
            {
                Vector rij = {bj.position.x - bi.position.x, bj.position.y - bi.position.y};
                double inv_r3 = pow(rij.x * rij.x + rij.y * rij.y + E * E, -1.5);
                double f = (GRAVITY * bj.mass) / inv_r3;
                Vector force = {rij.x * f, rij.y * f};
                bi.acceleration.x += (force.x / bi.mass);
                bi.acceleration.y += (force.y / bi.mass);
            }
        }
    }
}

__device__ void calculateVelocity(Body &bi)
{

    bi.velocity.x += bi.acceleration.x * DT / 2.0;
    bi.velocity.y += bi.acceleration.y * DT / 2.0;
}

__device__ void calculatePosition(Body &bi)
{
    int boundaryWidth = WINDOW_WIDTH, boundaryHeight = WINDOW_HEIGHT;
    bi.position.x += bi.velocity.x * DT;
    bi.position.y += bi.velocity.y * DT;
    if (bi.position.x < 0)
    {
        bi.position.x = 0;
        bi.velocity.x *= -1.0f;
    }
    if (bi.position.x > boundaryWidth)
    {
        bi.position.x = boundaryWidth;
        bi.velocity.x *= -1.0f;
    }
    if (bi.position.y < 0)
    {
        bi.position.y = 0;
        bi.velocity.y *= -1.0f;
    }
    if (bi.position.y > boundaryHeight)
    {
        bi.position.y = boundaryHeight;
        bi.velocity.y *= -1.0f;
    }
}

__global__ void DirectSumKernel(Body *bodies, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        Body &bi = bodies[i];
        double fx = 0.0, fy = 0.0;
        if (bi.isDynamic)
        {

            bi.acceleration = {0.0, 0.0};
            for (int b = 0; b < n; ++b)
            {
                if (b != i)
                {
                    Body &bj = bodies[b];
                    if (!isCollide(bi, bj) && bi.isDynamic)
                    {
                        Vector rij = {bj.position.x - bi.position.x, bj.position.y - bi.position.y};
                        double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
                        double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                        Vector force = {rij.x * f, rij.y * f};
                        fx += (force.x / bi.mass);
                        fy += (force.y / bi.mass);
                    }
                }
            }
            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
        }
    }
}

__global__ void DirectSumTiledKernel(Body *bodies, int n)
{
    __shared__ Body Bds[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;

    if (i < n)
    {
        Body &bi = bodies[i];
        double fx = 0.0, fy = 0.0;
        bi.acceleration = {0.0, 0.0};
        for (int tile = 0; tile < gridDim.x; ++tile)
        {

            Bds[tx] = bodies[tile * blockDim.x + tx];
            __syncthreads();

            for (int b = 0; b < BLOCK_SIZE; ++b)
            {
                int j = tile * blockDim.x + b;
                if (j < n)
                {
                    Body bj = Bds[b];
                    if (!isCollide(bi, bj) && bi.isDynamic)
                    {
                        Vector rij = {bj.position.x - bi.position.x, bj.position.y - bi.position.y};
                        double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
                        double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (E * E));
                        Vector force = {rij.x * f, rij.y * f};
                        fx += (force.x / bi.mass);
                        fy += (force.y / bi.mass);
                    }
                }
            }
            __syncthreads();
        }
        bi.acceleration.x += fx;
        bi.acceleration.y += fy;
        bi.velocity.x += bi.acceleration.x * DT;
        bi.velocity.y += bi.acceleration.y * DT;
        bi.position.x += bi.velocity.x * DT;
        bi.position.y += bi.velocity.y * DT;
    }
}

void display(Body *bodies)
{

    std::cout << bodies[0].position.x << " " << bodies[0].position.y << std::endl;
}

Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + 800, (pos.y - 0) * scaleY + 800};
}

void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color = cv::Scalar(255, 255, 255); // White color
    for (int i = 0; i < n; i++)
    {
        Vector pos = scaleToWindow(bodies[i].position);
        cv::Point center(pos.x, pos.y);
        cv::circle(image, center, 1, color, -1);
    }
    video.write(image);
    // cv::imwrite("frame" + std::to_string(id) + ".jpg", image);
}

Body *initRandomBodies(int n)
{

    Body *bodies = new Body[n];
    srand(time(NULL));
    double maxDistance = 2.2790e11;
    double minDistance = 1.4960e11;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < n - 1; ++i)
    {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = CENTERX + radius * std::cos(angle);
        double y = CENTERY + radius * std::sin(angle);
        Vector position = {x, y};
        bodies[i].isDynamic = true;
        bodies[i].mass = 5.974e24;
        bodies[i].radius = 1.3927e6;
        bodies[i].position = position;
        bodies[i].velocity = {0.0, 0.0};
        bodies[i].acceleration = {0.0, 0.0};
    }

    bodies[n - 1].isDynamic = false;
    bodies[n - 1].mass = 1.9890e30;
    bodies[n - 1].radius = 1.3927e6;
    bodies[n - 1].position = centerPos;
    bodies[n - 1].velocity = {0.0, 0.0};
    bodies[n - 1].acceleration = {0.0, 0.0};

    return bodies;
}

Body *initSpiralBodies(int n)
{

    Body *bodies = new Body[n];
    srand(time(NULL));
    double maxDistance = 2.2790e11;
    double minDistance = 1.4960e11;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < n - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = CENTERX + radius * std::cos(angle);
        double y = CENTERY + radius * std::sin(angle);

        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * 1.9891e30) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        bodies[i].isDynamic = true;
        bodies[i].mass = 5.974e24;
        bodies[i].radius = 1.3927e6;
        bodies[i].position = position;
        bodies[i].velocity = velocity;
        bodies[i].acceleration = {0.0, 0.0};
    }

    bodies[n - 1].isDynamic = false;
    bodies[n - 1].mass = 1.9890e30;
    bodies[n - 1].radius = 1.3927e6;
    bodies[n - 1].position = centerPos;
    bodies[n - 1].velocity = {0.0, 0.0};
    bodies[n - 1].acceleration = {0.0, 0.0};
    return bodies;
}

void setBody(Body *bodies, int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
{
    bodies[i].isDynamic = isDynamic;
    bodies[i].mass = mass;
    bodies[i].radius = radius;
    bodies[i].position = position;
    bodies[i].velocity = velocity;
    bodies[i].acceleration = acceleration;
}

Body *initSolarSystem()
{

    Body *bodies = new Body[5];
    setBody(bodies, 0, true, 5.9740e24, 1.3927e6, {1.4960e11, 0}, {0, 2.9800e4}, {0, 0});
    setBody(bodies, 1, true, 6.4190e23, 1.3927e6, {2.2790e11, 0}, {0, 2.4100e4}, {0, 0});
    setBody(bodies, 2, true, 3.3020e23, 1.3927e6, {5.7900e10, 0}, {0, 4.7900e4}, {0, 0});
    setBody(bodies, 3, true, 4.8690e24, 1.3927e6, {1.0820e11, 0}, {0, 3.5000e4}, {0, 0});
    setBody(bodies, 4, false, 1.9890e30, 1.3927e6, {CENTERX, CENTERY}, {0, 0}, {0, 0});
    return bodies;
}

int main(int argc, char **argv)
{
    int nBodies = NUM_BODIES;
    int iters = 300;
    if (argc == 3)
    {
        nBodies = atoi(argv[1]);
        iters = atoi(argv[2]);
    }

    Body *h_bodies = initRandomBodies(nBodies);

    int bytes = nBodies * sizeof(Body);

    Body *d_bodies;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bodies, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = ceil((double)nBodies / blockSize);
    int it = 0;
    while (it < iters) // main loop
    {

        DirectSumTiledKernel<<<gridSize, blockSize>>>(d_bodies, nBodies);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost));
        storeFrame(h_bodies, nBodies, ++it);
        // display(bodies);
    }
    video.release();

    // free memories
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    free(h_bodies);

    CHECK_LAST_CUDA_ERROR();
    return 0;
}

#endif