// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include "matrix_generators.hpp"
#include "timer.hpp"
#include "directcl.hpp"
#include "hpxcl_single.hpp"
#include "hpx_helpers.hpp"

#include <string>

int main(int argc, char* argv[]) {
  // Print help message on wrong argument count
  if (argc < 2) {
    hpx::cerr << "Usage: " << argv[0] << " matrixsize" << hpx::endl;
    return hpx::finalize();
  }

  {
    ////////////////////////////////////////////
    // Initializes all matrices
    //
    size_t vector_size = std::stoul(argv[1]);
    hpx::cout << "Vector size: " << vector_size << std::endl;

    hpx::cout << "Generating matrix A ..." << hpx::endl;
    auto a = generate_input_matrix(vector_size);
    hpx::cout << "Generating matrix B ..." << hpx::endl;
    auto b = generate_input_matrix(vector_size);
    hpx::cout << "Generating matrix C ..." << hpx::endl;
    auto c = generate_input_matrix(vector_size);

    hpx::cout << "Calculating reference result on CPU ..." << hpx::endl;
    double time_cpu;
    auto z = calculate_result(a, b, c, &time_cpu);
    hpx::cout << "        ... " << time_cpu << " ms" << hpx::endl;

    ////////////////////////////////////////////
    // Direct OpenCL calculation
    //
    hpx::cout << hpx::endl;
    hpx::cout << "///////////////////////////////////////" << hpx::endl;
    hpx::cout << "// Direct OpenCL" << hpx::endl;
    hpx::cout << "//" << hpx::endl;

    // initializes
    hpx::cout << "Initializing ..." << hpx::endl;
    directcl_initialize(vector_size);

    // main calculation with benchmark
    double time_directcl_nonblock;
    double time_directcl_total;
    hpx::cout << "Running calculation ..." << hpx::endl;
    std::shared_ptr<std::vector<float>> z_directcl = directcl_calculate(
        a, b, c, &time_directcl_nonblock, &time_directcl_total);

    // shuts down
    hpx::cout << "Shutting down ..." << hpx::endl;
    directcl_shutdown();

    // checks for correct result
    check_for_correct_result(z_directcl->data(), (*z_directcl).size(), z.data(),
                             z.size());

    // Prints the benchmark statistics
    hpx::cout << hpx::endl;
    hpx::cout << "    Nonblocking calls:       " << time_directcl_nonblock
              << " ms" << hpx::endl;
    hpx::cout << "    Total Calculation Time:  " << time_directcl_total << " ms"
              << hpx::endl;
    hpx::cout << hpx::endl;

    ////////////////////////////////////////////
    // HPXCL local calculation
    //
    hpx::cout << hpx::endl;
    hpx::cout << "///////////////////////////////////////" << hpx::endl;
    hpx::cout << "// HPXCL local" << hpx::endl;
    hpx::cout << "//" << hpx::endl;

    // initializes
    hpx::cout << "Initializing ..." << hpx::endl;
    hpxcl_single_initialize(hpx::find_here(), vector_size);

    // main calculation with benchmark
    hpx::cout << "Running calculation ..." << hpx::endl;
    double time_hpxcl_local_nonblock;
    double time_hpxcl_local_total;
    auto z_hpxcl_local = hpxcl_single_calculate(
        a, b, c, &time_hpxcl_local_nonblock, &time_hpxcl_local_total);

    // shuts down
    hpx::cout << "Shutting down ..." << hpx::endl;
    hpxcl_single_shutdown();

    // checks for correct result
    check_for_correct_result(z_hpxcl_local.data(), z_hpxcl_local.size(),
                             z.data(), z.size());

    // Prints the benchmark statistics
    hpx::cout << hpx::endl;
    hpx::cout << "    Nonblocking calls:       " << time_hpxcl_local_nonblock
              << " ms" << hpx::endl;
    hpx::cout << "    Total Calculation Time:  " << time_hpxcl_local_total
              << " ms" << hpx::endl;
    hpx::cout << hpx::endl;

    ////////////////////////////////////////////
    // HPXCL remote calculation
    //
    hpx::cout << hpx::endl;
    hpx::cout << "///////////////////////////////////////" << hpx::endl;
    hpx::cout << "// HPXCL remote" << hpx::endl;
    hpx::cout << "//" << hpx::endl;

    // initializes
    hpx::cout << "Initializing ..." << hpx::endl;
    hpx::id_type remote_node = hpx_get_remote_node();
    if (remote_node) {
      hpxcl_single_initialize(remote_node, vector_size);

      // main calculation with benchmark
      hpx::cout << "Running calculation ..." << hpx::endl;
      double time_hpxcl_remote_nonblock;
      double time_hpxcl_remote_total;
      auto z_hpxcl_remote = hpxcl_single_calculate(
          a, b, c, &time_hpxcl_remote_nonblock, &time_hpxcl_remote_total);

      // shuts down
      hpx::cout << "Shutting down ..." << hpx::endl;
      hpxcl_single_shutdown();

      // checks for correct result
      check_for_correct_result(z_hpxcl_remote.data(), z_hpxcl_remote.size(),
                               z.data(), z.size());

      // Prints the benchmark statistics
      hpx::cout << hpx::endl;
      hpx::cout << "    Nonblocking calls:       " << time_hpxcl_remote_nonblock
                << " ms" << hpx::endl;
      hpx::cout << "    Total Calculation Time:  " << time_hpxcl_remote_total
                << " ms" << hpx::endl;
      hpx::cout << hpx::endl;
    }

    ////////////////////////////////////////////
    // HPXCL distributed calculation
    //
    hpx::cout << hpx::endl;
    // hpx::cout << "///////////////////////////////////////" << hpx::endl;
    // hpx::cout << "// HPXCL distributed" << hpx::endl;
    // hpx::cout << "//" << hpx::endl;

    ///////////////////////////////////////////
    // Shutdown
    //
    hpx::cout << hpx::endl;
    hpx::cout << "Shutting down hpx ... " << hpx::endl;
  }

  hpx::cout << "Program finished." << hpx::endl;

  // End the program
  return hpx::finalize();
}
