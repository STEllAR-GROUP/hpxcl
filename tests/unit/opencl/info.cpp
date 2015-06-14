// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device cldevice )
{

    ////////////////////////////////////////////////////////////////////////////
    // Test if cast to string and cast to vector<char> returns
    // identical results
    //

    std::string device_version =
        cldevice.get_device_info<CL_DEVICE_VERSION>().get();
    
    HPX_TEST(device_version.find("OpenCL ") == 0);


    ////////////////////////////////////////////////////////////////////////////
    // Test if CL_DEVICE_MAX_WORK_ITEM_SIZES-Array returns as many items as
    // specified by OpenCL
    //

    cl_uint work_dims =
        cldevice.get_device_info<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>().get();

    std::vector<std::size_t> work_items = 
        cldevice.get_device_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>().get();

    HPX_TEST_EQ(work_items.size(), work_dims);


}


