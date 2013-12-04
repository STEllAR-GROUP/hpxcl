//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(MANAGED_CUDA_COMPONENT_1_HPP)
#define MANAGED_CUDA_COMPONENT_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/managed_cuda_component.hpp"

namespace cuda_hpx
{
	class managed_cuda_component
		: public hpx::components::client_base<
			managed_cuda_component, stubs::managed_cuda_component
		  >
	{
		typedef hpx::components::client_base<
			managed_cuda_component, stubs::managed_cuda_component
			> base_type;

		typedef base_type::argument_type argument_type;

	public:
		managed_cuda_component()
		{}

		managed_cuda_component(hpx::future<hpx::naming::id_type> const& gid)
			: base_type(gid)
		{}

		void test1_non_blocking()
		{
			BOOST_ASSERT(this->get_gid());
			this->base_type::test1_non_blocking(this->get_gid());
		}

		void test1_sync()
		{
			BOOST_ASSERT(this->get_gid());
			this->base_type::test1_sync(this->get_gid());
		}

		hpx::lcos::future<argument_type> test2_async()
		{
			BOOST_ASSERT(this->git_gid());
			return this->base_type::test2_async(this->get_gid());
		}

		argument_type test2_sync()
		{
			BOOST_ASSERT(this->get_gid());
			return this->base_type::test2_sync(this->get_gid());
		}

        void get_cuda_info()
        {
            base_type::get_cuda_info();
        }

        hpx::lcos::future<float> calculate_pi_async(int nthreads,int nblocks)
        {
            BOOST_ASSERT(this->get_gid());
            return this->base_type::calculate_pi_async(this->get_gid(),nthreads,nblocks);
        }
        float calculate_pi_sync(int nthreads,int nblocks)
        {
            BOOST_ASSERT(this->get_gid());
            return this->base_type::calculate_pi_sync(this->get_gid(),nthreads,nblocks);
        }
	};
}
#endif //MANAGED_CUDA_COMPONENT_1_HPP
