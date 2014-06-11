// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>

#include "cuda_components/device.hpp"



int main(int argc,char* argv[])
{   boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING "[options]");
    desc_commandline.add_options()
        ("info",
            boost::program_options::value<bool>()->default_value(false)
        );
	return hpx::init(desc_commandline,argc,argv);
}

int hpx_main(boost::program_options::variables_map& vm)
{   
    bool info = vm["info"].as<bool>();
    typedef hpx::cuda::device cuda_device;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    cuda_device client;
    client.create(hpx::find_here());
    if(info)
        client.get_cuda_info();
    hpx::util::high_resolution_timer t;
    hpx::lcos::future<double> result = client.wait();
    float cpu_pi = calculate_pi(10000000,100000);
    std::cout<<cpu_pi<<" and ";
    std::cout<<std::endl<<gpu_pi<<" were calculated at the same time"<<std::endl;
    std::cout<<"It took "<<t.elapsed()<<" seconds"<<std::endl;
    return hpx::finalize();
}