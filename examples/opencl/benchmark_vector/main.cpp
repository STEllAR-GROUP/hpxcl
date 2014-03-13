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

#include <string>

int hpx_main(int argc, char* argv[])
{

    // Print help message on wrong argument count
    if(argc < 2)
    {
        hpx::cerr << "Usage: " << argv[0] << " matrixsize" << hpx::endl;
        return hpx::finalize();
    }


    {
    
        ////////////////////////////////////////////
        // Initializes all matrices
        //
        size_t vector_size = std::stoul(argv[1]);
        hpx::cout << "Vector size: " << vector_size << hpx::endl;
    
        hpx::cout << "Generating matrix A ..." << hpx::endl;
        std::vector<float> a = generate_input_matrix(vector_size);
        hpx::cout << "Generating matrix B ..." << hpx::endl;
        std::vector<float> b = generate_input_matrix(vector_size);
        hpx::cout << "Generating matrix C ..." << hpx::endl;
        std::vector<float> c = generate_input_matrix(vector_size);
        
        hpx::cout << "Calculating reference result on CPU ..." << hpx::endl;
        timer_start();
        std::vector<float> z = calculate_result(a,b,c);
        double time_cpu = timer_stop();
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
        hpx::cout << "Running calculation ..." << hpx::endl;
        timer_start();
        std::vector<float> z_directcl = directcl_calculate(a,b,c);
        double time_directcl = timer_stop();

        // shuts down
        hpx::cout << "Shutting down ..." << hpx::endl;
        directcl_shutdown();

        // checks for correct result
        check_for_correct_result(&z_directcl[0], z_directcl.size(),
                                 &z[0], z.size());
        
        // Prints the benchmark statistics
        hpx::cout << hpx::endl;
        hpx::cout << "    Calculation Time: " << time_directcl << " ms";
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
        timer_start();
        std::vector<char> z_hpxcl_local = hpxcl_single_calculate(a,b,c);
        double time_hpxcl_local = timer_stop();

        // shuts down
        hpx::cout << "Shutting down ..." << hpx::endl;
        hpxcl_single_shutdown();

        // checks for correct result
        check_for_correct_result((float*)(&z_hpxcl_local[0]),
                                 z_hpxcl_local.size()/sizeof(float),
                                 &z[0], z.size());
        
        // Prints the benchmark statistics
        hpx::cout << hpx::endl;
        hpx::cout << "    Calculation Time: " << time_hpxcl_local << " ms";
        hpx::cout << hpx::endl;



        ////////////////////////////////////////////
        // HPXCL remote calculation
        //
        hpx::cout << hpx::endl;
        hpx::cout << "///////////////////////////////////////" << hpx::endl;
        hpx::cout << "// HPXCL remote" << hpx::endl;
        hpx::cout << "//" << hpx::endl;


        ////////////////////////////////////////////
        // HPXCL distributed calculation
        //
        hpx::cout << hpx::endl;
        hpx::cout << "///////////////////////////////////////" << hpx::endl;
        hpx::cout << "// HPXCL distributed" << hpx::endl;
        hpx::cout << "//" << hpx::endl;



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

