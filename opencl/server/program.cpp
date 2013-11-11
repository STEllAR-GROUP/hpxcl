// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "program.hpp"
#include "../tools.hpp"
#include "device.hpp"

#include <string>
#include <sstream>

#include <CL/cl.h>


using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(program);


program::program(hpx::naming::id_type device_id, std::string _code)
{
    this->program_id = NULL;
    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                         <hpx::opencl::server::device>(parent_device_id).get();
    this->code = _code;


    // create variables for clCreateProgram call
    size_t code_size = code.length();
    const char* code_ptr = code.c_str();

    // initialize the cl_program object
    cl_int err;
    program_id = clCreateProgramWithSource(parent_device->get_context(), 1,
                                            &code_ptr, &code_size, &err);
    clEnsure(err, "clCreateProgramWithSource()");
                              
}

program::~program()
{
    cl_int err;

    // release the cl_program object
    if(program_id)
    {
        err = clReleaseProgram(program_id);
        clEnsure_nothrow(err, "clReleaseProgram()");
        program_id = NULL;
    }

}

std::string
program::acquire_build_log()
{
    cl_int err;

    size_t build_log_size;

    // Query size
    err = clGetProgramBuildInfo(program_id, parent_device->get_device_id(),
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
    
    // Create buffer
    std::vector<char> buf(build_log_size);

    // Get log
    err = clGetProgramBuildInfo(program_id, parent_device->get_device_id(),
                                CL_PROGRAM_BUILD_LOG, build_log_size,
                                &buf[0], NULL);

    // make build log look nice in exception
    std::stringstream sstream;
    sstream << std::endl << std::endl;
    sstream << "//////////////////////////////////////" << std::endl;
    sstream << "/// OPENCL BUILD LOG" << std::endl;
    sstream << "///" << std::endl;
    sstream << std::endl << &buf[0] << std::endl;
    sstream << "///" << std::endl;
    sstream << "/// OPENCL BUILD LOG END" << std::endl;
    sstream << "//////////////////////////////////////" << std::endl;
    sstream << std::endl;
    
    // return the nice looking error string.
    return sstream.str();

}

void
program::build(std::string options)
{
    
    cl_int err;

    // fetch device id from parent device
    cl_device_id device_id = parent_device->get_device_id();

    // build the program
    err = clBuildProgram(program_id, 1, &device_id, options.c_str(), NULL, NULL);
    
    // in case of build error print build log
    if(err == CL_BUILD_PROGRAM_FAILURE)
    {
        // throw beautiful build log exception.
        HPX_THROW_EXCEPTION(hpx::no_success, "clBuildProgram()",
                            (std::string(hpx::opencl::clErrToStr(err))
                            + acquire_build_log()).c_str());
    }
        
    // check for other errors
    clEnsure(err, "clBuildProgram()");

}

cl_program
program::get_cl_program()
{
    return program_id;
}

hpx::naming::id_type
program::get_device_id()
{
    return parent_device_id;
}
