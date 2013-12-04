//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(MANAGED_CUDA_COMPONENT_3_HPP)
#define MANAGED_CUDA_COMPONENT_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/managed_cuda_component.hpp"

namespace cuda_hpx
{
	namespace stubs
	{
		struct managed_cuda_component
			: hpx::components::stub_base<server::managed_cuda_component>
		{
			typedef server::managed_cuda_component::argument_type argument_type;

			static void test1_non_blocking(hpx::naming::id_type const& gid)
			{
				typedef server::managed_cuda_component::test1_action action_type;
				hpx::apply<action_type>(gid);
			}

			static void test1_sync(hpx::naming::id_type const& gid)
			{
				typedef server::managed_cuda_component::test1_action action_type;
				hpx::async<action_type>(gid).get();
			}

			static hpx::lcos::future<argument_type>
			test2_async(hpx::naming::id_type const& gid)
			{
				typedef server::managed_cuda_component::test2_action action_type;
				return hpx::async<action_type>(gid);
			}

			static argument_type test2_sync(hpx::naming::id_type const& gid)
			{
				return test2_async(gid).get();
			}

            static void get_cuda_info()
            {
                server::managed_cuda_component::get_cuda_info();
            }

			static hpx::lcos::future<float>
			calculate_pi_async(hpx::naming::id_type const& gid,int nthreads, int nblocks)
			{
                typedef server::managed_cuda_component::calculate_pi_action action_type;
                return hpx::async<action_type>(gid,nthreads,nblocks);
			}

			static float calculate_pi_sync(hpx::naming::id_type const& gid,int nthreads, int nblocks)
			{
                return calculate_pi_async(gid,nthreads,nblocks).get();
			}
		};

	}
}
#endif
