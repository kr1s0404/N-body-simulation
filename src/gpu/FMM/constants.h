/*
   Copyright 2023 Hsin-Hung Wu

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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NUM_BODIES 300
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11
#define GRAVITY 6.67E-11
#define E 0.5
#define DT 25000
#define CENTERX 0
#define CENTERY 0
#define BLOCK_SIZE 1024
#define GRID_SIZE 512
#define MAX_N 4194304
#define COLLISION_TH 1.0e10
#define MIN_DIST 2.0e10
#define MAX_DIST 5.0e11
#define SUN_MASS 1.9890e30
#define SUN_DIA 1.3927e6
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756
#define HBL 1.6e29

// FMM specific constants
#define MAX_DEPTH 9           // Maximum depth of the octree (match BH implementation)
#define P 6                   // Number of terms in multipole expansion
#define THETA 0.5             // Multipole acceptance criterion
#define MAX_PARTICLES_PER_LEAF 64  // Maximum particles per leaf node
#define MAX_CELLS 1000000     // Maximum number of cells in the quadtree

#endif 