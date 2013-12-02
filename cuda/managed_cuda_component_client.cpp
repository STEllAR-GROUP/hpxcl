#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>

#include <vector>
#include <iostream>

#include <boost/foreach.hpp>

#include "cuda_components/managed_cuda_component.hpp"

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

int main(int argc,char** argv)
{
    boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING "[options]");
    desc_commandline.add_options()
        ("info",
            boost::program_options::value<bool>()->default_value(false)
        );

	return hpx::init(desc_commandline,argc,argv);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        bool info = vm["info"].as<bool>();
        typedef cuda_hpx::server::managed_cuda_component cuda_component_type;
        typedef cuda_component_type::argument_type argument_type;

        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        cuda_hpx::managed_cuda_component client(
        hpx::components::new_<cuda_component_type>(localities.back()));
        if(info)
        {
            client.get_cuda_info();
        }
        cuda_hpx::managed_cuda_component client2;
        client2.create(hpx::find_here());

        hpx::util::high_resolution_timer t;
        hpx::lcos::future<float> pi = client2.calculate_pi_async(200,200);
        float cpu_pi = calculate_pi(10000000,100000);
        std::cout<<cpu_pi<<" and ";
        float gpu_pi = pi.get();
        std::cout<<gpu_pi<<" where calculated at the same time"<<std::endl;
        std::cout<<"It took "<<t.elapsed()<<" seconds"<<std::endl;
    }
    return hpx::finalize();
}
