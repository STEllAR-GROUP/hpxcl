// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


static void cl_test(hpx::opencl::device cldevice)
{

    ////////////////////////////////////////////////////////////////////////////
    // Test if cast to string and cast to vector<char> returns
    // identical results
    //

    hpx::opencl::info name_info = cldevice.get_device_info(CL_DEVICE_NAME);

    std::string name_str = static_cast<std::string>(name_info);
    std::vector<char> name_arr = static_cast<std::vector<char>>(name_info);
    
    // count valid characters in array
    std::size_t name_arr_size = 0;
    for(char &c : name_arr){
        if(c == '\0') break;
        name_arr_size++;
    }
    
    HPX_TEST_EQ(name_str.size(), name_arr_size);


    ////////////////////////////////////////////////////////////////////////////
    // Test if CL_DEVICE_MAX_WORK_ITEM_SIZES-Array returns as many items as
    // specified by OpenCL
    //

    cl_uint work_dims = static_cast<cl_uint>(cldevice.get_device_info(
                                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));

    std::vector<std::size_t> work_items = static_cast<std::vector<std::size_t> >(
        cldevice.get_device_info(CL_DEVICE_MAX_WORK_ITEM_SIZES));

    HPX_TEST_EQ(work_items.size(), work_dims);


}


