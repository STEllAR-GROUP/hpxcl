// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <ifstream>

#include "program.hpp"

using namespace hpx::opencl;


program::program(const char* source)
{
    program_code = std::string(source);
    build_status = raw;
}
   
program::program(std::string source)
{
    program_code = source;
    build_status = raw;
}

static std::string read_from_file(const char* filename)
{
    try
    { 

        // Open file
        std::ifstream in(filename, std::ios_base:in | std::ios_base::binary);

        // Read file length
        in.seekg(0, std::ios_base::end);
        size_t filelength = in.tellg();
        in.seekg(0, std::ios_base::beg);

        // Create return string
        std::string str;
        str.resize(filelength);
        
        // Read file to string
        in.read(&str[0], str.size());

        // Close file
        in.close();

        // Return string
        return str;

    }
    catch(exception e)
    {
        std::string errormessage("Unable to read file '")
        errormessage += filename;
        errormessage += "'";
        HPX_THROW_EXCEPTION(hpx::no_success,
                            "hpx::opencl::program::read_from_file()",
                            errormessage.c_str()); 
    }

}

void
program::connect_to_device(device device_)
{

    if(build_status != raw)
        HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::opencl::program::connect_to_device()",
                        "This function needs to be called before compilation.");

    connected_devices.push_back(device_);

}

void 
program::connect_to_devices(devices* devices, size_t num_devices)
{

    if(build_status != raw)
        HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::opencl::program::connect_to_devices()",
                        "This function needs to be called before compilation.");
    
    for(int i = 0; i < num_devices; i++)
    {
        connected_devices.push_back(devices[i]);
    }

}

void
program::create_programs_on_devices()
{


    // TODO : create program objects on devices

   build_status = created; 

}


void
program::build(const char* options)
{

    // create program objects on device
    if(build_status == raw)
        create_programs_on_devices();

    if(build_status != created)
        HPX_THROW_EXCEPTION(hpx::invalid_status,
                            "hpx::opencl::program::build()",
                            "Invalid program status.");

   
    if(build_status != raw)
// TODO : run build on every device

}














