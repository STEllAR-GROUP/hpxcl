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

            static hpx::lcos::future<argument_type>
            check_if_hit_async(hpx::naming::id_type const& gid,argument_type num_of_sets,int cuda_blocks, int cuda_threads)
            {
                typedef server::managed_cuda_component::check_if_hit_action action_type;
                return hpx::async<action_type>(gid,num_of_sets,cuda_blocks,cuda_threads);
            }

            static argument_type check_if_hit_sync(hpx::naming::id_type const& gid,argument_type num_of_sets,int cuda_blocks, int cuda_threads)
            {
                return check_if_hit_async(gid,num_of_sets,cuda_blocks,cuda_threads).get();
            }

            static void get_cuda_info()
            {
                server::managed_cuda_component::get_cuda_info();
            }

			/*static hpx::lcos::future<double>
			calculate_pi_async(hpx::naming::id_type const& gid,uint64_t num_of_iterations,uint64_t num_of_sets,int cuda_blocks,int cuda_threads)
			{
                typedef server::managed_cuda_component::calculate_pi_action action_type;
                return hpx::async<action_type>(gid,num_of_iterations,num_of_sets,cuda_blocks,cuda_threads);
			}
            */
			/*static double calculate_pi_sync(hpx::naming::id_type const& gid)
			{
                return calculate_pi_async(gid).get();
			}*/
		};

	}
}
#endif
