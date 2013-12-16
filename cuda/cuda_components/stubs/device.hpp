#if !defined(DEVICE_3_HPP)
#define DEVICE_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/device.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct device
                : hpx::components::stub_base<server::device>
            {
                static void test1_non_blocking(hpx::naming::id_type const& gid)
                {
                    typedef server::device::test1_action action_type;
                    hpx::apply<action_type>(gid);
                }

                static void test1_sync(hpx::naming::id_type const& gid)
                {
                    typedef server::device::test1_action action_type;
                    hpx::async<action_type>(gid).get();
                }

                static hpx::lcos::future<long>
                test2_async(hpx::naming::id_type const& gid)
                {
                    typedef server::device::test2_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static long test2_sync(hpx::naming::id_type const& gid)
                {
                    return test2_async(gid).get();
                }

                static void get_cuda_info(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_cuda_info_action action_type;
                    hpx::apply<action_type>(gid);
                }

                static hpx::lcos::future<float>
                calculate_pi_async(hpx::naming::id_type const& gid,int nthreads, int nblocks)
                {
                    typedef server::device::calculate_pi_action action_type;
                    return hpx::async<action_type>(gid,nthreads,nblocks);
                }

                static float calculate_pi_sync(hpx::naming::id_type const& gid,int nthreads, int nblocks)
                {
                    return calculate_pi_async(gid,nthreads,nblocks).get();
                }
            };

        }
    }
}
#endif
