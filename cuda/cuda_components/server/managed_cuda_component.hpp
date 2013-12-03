//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

				float calculate_pi(int nthreads,int nblocks)
				{
                    return pi(nthreads,nblocks);
				}

		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,test1);
		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,test2);
		 HPX_DEFINE_COMPONENT_ACTION(managed_cuda_component,calculate_pi);
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
HPX_REGISTER_ACTION_DECLARATION(
	cuda_hpx::server::managed_cuda_component::calculate_pi_action,
	managed_cuda_component_calculate_pi_action);
HPX_REGISTER_ACTION_DECLARATION(
    cuda_hpx::server::managed_cuda_component::get_cuda_info_action,
    managed_cuda_component_get_cuda_info_action);

#endif //MANAGED_CUDA_COMPONENT_2_HPP
