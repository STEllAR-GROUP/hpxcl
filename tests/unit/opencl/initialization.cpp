// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


int hpx_main(variables_map & vm)
{
    {
        init(vm);   
//        buffer_init();
 //       buffer_write();
  //      events();
   //     events_futures();
    }
    
    hpx::finalize();
    return hpx::util::report_errors();
}



///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ( "deviceid"
        , value<std::size_t>()->default_value(0)
        , "the ID of the device we will run our tests on") ;

    return hpx::init(cmdline, argc, argv);
}
