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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NUM_BODIES 10000
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024
#define NBODY_WIDTH 10.0f
#define NBODY_HEIGHT 10.0f
#define GRAVITY 6.67430e-11f
#define E 0.5f
#define DT 0.01f
#define THETA 0.5f
#define CENTERX 5.0f
#define CENTERY 5.0f
#define CENTERZ 5.0f
#define BLOCK_SIZE 256
#define GRID_SIZE 512
#define MAX_LEVEL 5
#define MAX_CELLS (1 << (3 * MAX_LEVEL + 3))
#define COLLISION_TH 0.01f
#define MIN_DIST 0.1f
#define MAX_DIST 5.0f
#define SUN_MASS 1000.0f
#define SUN_RADIUS 0.5f
#define PLANET_MASS 0.1f
#define PLANET_RADIUS 0.05f

#endif 