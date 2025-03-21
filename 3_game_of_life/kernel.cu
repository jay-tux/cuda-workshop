//
// Created by jay on 3/15/25.
//

#include "cuda_wrapper.hpp"
#include "kernel.hpp"


using namespace c2_game_of_life;

__global__ void do_step(const buffer in, buffer out) {
  TODO("Update state in out based on state in in");
}

__host__ void c2_game_of_life::step(const buffer& in, const buffer& out) {
  TODO("Call do_step with the correct block and grid sizes");
}
