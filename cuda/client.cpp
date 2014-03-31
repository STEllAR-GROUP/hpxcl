// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>

#include "cuda_components/device.hpp"

int main()
{
    return hpx::init();
}

int hpx_main()
{
    return hpx::finalize();
}
/*std::default_random_engine gen(std::time(0));
std::uniform_real_distribution<double> dist(0.0,1.0);

double calculate_pi(boost::uint64_t num_of_iterations,boost::uint64_t num_of_sets);
double check_if_hit(boost::uint64_t num_of_sets);

HPX_PLAIN_ACTION(check_if_hit,check_if_hit_action);

double calculate_pi(boost::uint64_t num_of_iterations,boost::uint64_t num_of_sets)
{
    boost::atomic<uint64_t> hits(0);
    double pi;

    num_of_iterations = num_of_iterations / num_of_sets;
    std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();
    std::vector<hpx::lcos::future<double> > futures;
    futures.reserve(num_of_iterations);
    for(boost::uint64_t i = 0;i<num_of_iterations;i++)
    {
        BOOST_FOREACH(hpx::naming::id_type const& node, localities)
        {
            futures.push_back(hpx::async<check_if_hit_action>(node,num_of_sets));
        }
    }
    hpx::wait(futures,
        [&](std::size_t,double d)
        {
            hits+=d;
        });

    pi = (double)hits/num_of_iterations*4.0;
    return (pi/num_of_sets)/localities.size();
}

double check_if_hit(boost::uint64_t num_of_sets)
{
    boost::atomic<uint64_t> hits_per_set(0);
    double x,y,z;
    for(boost::uint64_t i=0;i<num_of_sets;i++)
    {
        x = dist(gen);
        y = dist(gen);
        z = x*x+y*y;
        if(z<=1)
            hits_per_set++;
    }
    return hits_per_set;
}

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
    hpx::lcos::unique_future<int> result = client.wait();
    float cpu_pi = calculate_pi(10000000,100000);
    std::cout<<cpu_pi<<" and ";
    float gpu_pi = pi.get();
    std::cout<<std::endl<<gpu_pi<<" were calculated at the same time"<<std::endl;
    std::cout<<"It took "<<t.elapsed()<<" seconds"<<std::endl;
    return hpx::finalize();












    g++ -o ~/my_hpx_libs/libcuda_device.so ~/hpxcl/cuda/cuda_components/device.cpp \
`pkg-config --cflags --libs hpx_component` -DHPX_COMPONENT_NAME=cuda_device 2> errors/device.cpp

}*/