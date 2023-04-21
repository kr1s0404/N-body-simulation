#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <GLFW/glfw3.h>
#include "nBody.h"
#include "directSum.h"
#include "barnesHut.h"
#include "quadtree.h"
#include "constants.h"

using namespace std;
using namespace std::chrono;

Vector scaleToWindow(Vector pos)
{

     double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
     double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
     return Vector((pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2);
}

void drawDots(NBody &nb)
{

     glColor3f(1.0, 1.0, 1.0); // set drawing color to white

     for (auto &body : nb.bodies)
     {
          glPointSize(5);     // set point size to 5 pixels
          glBegin(GL_POINTS); // start drawing points
          Vector pos = scaleToWindow(body->position);
          glVertex2f(pos.x, pos.y);
          glEnd(); // end drawing points
     }
}

bool checkArgs(int nBodies, int alg, int sim)
{

     if (nBodies < 1)
     {
          std::cout << "ERROR: need to have at least 1 body" << std::endl;
          return false;
     }

     if (alg < 0 && alg > 1)
     {
          std::cout << "ERROR: algorithm doesn't exist" << std::endl;
          return false;
     }

     if (sim < 0 || sim > 2)
     {
          std::cout << "ERROR: simulation doesn't exist" << std::endl;
          return false;
     }

     return true;
}

int main(int argc, char **argv)
{
     int nBodies = NUM_BODIES;
     int alg = 0;
     int sim = 0;
     if (argc == 4)
     {
          nBodies = atoi(argv[1]);
          alg = atoi(argv[2]);
          sim = atoi(argv[3]);
     }

     if (!checkArgs(nBodies, alg, sim))
          return -1;

     NBody nb(nBodies, alg, sim);
     // initialize GLFW
     if (!glfwInit())
          return -1;
     GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body Simulation CPU", NULL, NULL); // create window
     if (!window)
     {
          glfwTerminate();
          return -1;
     }
     glfwMakeContextCurrent(window); // set context to current window

     glClearColor(0.0, 0.0, 0.0, 1.0); // set background color to black
     glMatrixMode(GL_PROJECTION);      // set up projection matrix
     glLoadIdentity();
     glOrtho(0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, -1.0f, 1.0f);
     long long execution_time = 0;
     long long iter = 0;
     while (!glfwWindowShouldClose(window)) // main loop
     {
          glClear(GL_COLOR_BUFFER_BIT); // clear the screen

          auto start = high_resolution_clock::now();
          nb.update();
          auto stop = high_resolution_clock::now();
          execution_time += duration_cast<milliseconds>(stop - start).count();
          drawDots(nb);
          glfwSwapBuffers(window); // swap front and back buffers
          glfwPollEvents();        // poll for events

          ++iter;
     }

     glfwTerminate(); // terminate GLFW

     std::cout << "average execution time per frame: " << execution_time / iter << " milliseconds" << std::endl;

     return 0;
}