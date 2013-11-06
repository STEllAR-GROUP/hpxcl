
// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_PROGRAM_HPP__
#define HPX_OPENCL_PROGRAM_HPP__

#include <hpx/include/components.hpp>

#include "device.hpp"

namespace hpx {
namespace opencl {


    class program
    { 
        //////////////////////////////////////////////////////
        /// This class represents an OpenCL program object
        ///
        
        public:
            ///////////////////////////////////////
            /// Exposed functionality
            /// 

            // Constructors
            program(const char* source);
            program(std::string source);

            ~program();

            // File reader, to be called as argument of constructor
            static std::string read_from_file(const char* filename);

            // sets device as a target for the program.
            // needs to be set before compilation.
            void connect_to_device(device device_);
            void connect_devices(device* devices, size_t num_devices);

            // compiles + links the program
            //  (mutually exclusive to compile() and link())
            void build(const char* options);

            // compiles and links the program seperately
            //  (mutually exclusive to build())
            /*
            void compile(const char* options, const header* headers,
                         size_t num_headers);
            void link(const char* options, const program *dependencies,
                      size_t num_input_programs);
            */

        private:
            ////////////////////////////////////
            /// Private member functions
            ///
            void create_programs_on_devices();

        private:
            ////////////////////////////////////
            /// Private member variables
            ///
            
            // List of target devices
            std::vector<device> connected_devices;
            
            // Current build status
            enum build_status_enum{raw, created, compiled, ready}
            build_status = raw;
            
            // The program code
            std::string program_code;
    };

}}







#endif

