#if !defined(MANAGED_CUDA_COMPONENT_2_HPP)
#define MANAGED_CUDA_COMPONENT_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/foreach.hpp>

#include "../cuda/kernel.cuh"

#include <iostream>
#include <random>
#include <ctime>

std::default_random_engine gen(std::time(0));
std::uniform_real_distribution<double> dist(0,1);

namespace cuda_hpx
{
	namespace server
	{
		class managed_cuda_component
			: public hpx::components::locking_hook<
				hpx::components::managed_component_base<managed_cuda_component>
			>
		{
			public:
				typedef boost::int64_t argument_type;
                //argument_type num_of_iterations, num_of_sets;

				managed_cuda_component() {}
                //cuda wrapper functions

                static void get_cuda_info()
                {
                    get_device_info();
                }

				void test1()
				{
                    std::cout<<"There is/are "<<get_devices()<<" cuda enabled device(s) available"<<std::endl;
				}
				//cuda kernel functions
				argument_type test2()
				{
                    argument_type a[] = {1};
                    cuda_test(a);
                    return a[0];
				}

                /*double calculate_pi(boost::uint64_t num_of_iterations,boost::uint64_t num_of_sets,int cuda_blocks,int cuda_threads)
                {
                    uint64_t hits = 0;
                    num_of_iterations  /= num_of_sets;

                    for(uint64_t i = 0;i<num_of_iterations;i++)
                    {
                        //parallel implementation
                    }
                }*/
				argument_type check_if_hit(boost::uint64_t num_of_sets, int cuda_blocks, int cuda_threads)
				{
                    boost::atomic<uint64_t> hits_per_set(0);
                    //split work between cuda and hpx
                    num_of_sets = num_of_sets / 2;
                    std::cout<<"num of sets is "<<num_of_sets<<std::endl;
                    int sets_per_thread = (num_of_sets / cuda_blocks) / cuda_threads;
                    std::cout<<"sets per thread are "<<sets_per_thread<<std::endl;
                    uint64_t cuda_hits = gpu_num_of_hits(cuda_blocks,cuda_threads, sets_per_thread);
                    double x,y,z;
                    for(boost::uint64_t i=0;i<num_of_sets;i++)
                    {
                        x = dist(gen);
                        y = dist(gen);
                        z = x*x + y*y;
                        if(z<=1)
                        hits_per_set++;
                    }
                    return hits_per_set + cuda_hits;
				}

		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,test1);
		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,test2);
		 //HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,calculate_pi);
		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,check_if_hit);
		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,get_cuda_info);
		};
	}
}

//HPX action declarations

HPX_REGISTER_ACTION_DECLARATION(
	cuda_hpx::server::managed_cuda_component::test1_action,
	managed_cuda_component_test1_action);
HPX_REGISTER_ACTION_DECLARATION(
	cuda_hpx::server::managed_cuda_component::test2_action,
	managed_cuda_component_test2_action);
/*HPX_REGISTER_ACTION_DECLARATION(
	cuda_hpx::server::managed_cuda_component::calculate_pi_action,
	managed_cuda_component_calculate_pi_action);*/
HPX_REGISTER_ACTION_DECLARATION(
    cuda_hpx::server::managed_cuda_component::check_if_hit_action,
    managed_cuda_component_check_if_hit_action);
HPX_REGISTER_ACTION_DECLARATION(
    cuda_hpx::server::managed_cuda_component::get_cuda_info_action,
    managed_cuda_component_get_cuda_info_action);

#endif //MANAGED_CUDA_COMPONENT_2_HPP
