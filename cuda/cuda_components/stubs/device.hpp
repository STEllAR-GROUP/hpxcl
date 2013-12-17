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
                static void get_cuda_info(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_cuda_info_action action_type;
                    hpx::apply<action_type>(gid);
                }

                static void set_device(hpx::naming::id_type const& gid,int dev)
                {
                    typedef server::device::set_device_action action_type;
                    hpx::async<action_type>(gid,dev).get();
                }

                static std::vector<int> get_all_devices(std::vector<hpx::naming::id_type> localities)
                {
                    std::vector<int> vec;
                    typedef server::device::get_all_devices_action action_type;
                    for(uint64_t i = 0;i<localities.size();i++)
                    {
                        vec.push_back(hpx::async<action_type>(localities[i]).get());
                    }
                    return vec;
                }

                template <typename T>
                static T* device_malloc(hpx::naming::id_type const& gid, size_t mem_size)
                {
                    typedef typename server::device::device_malloc_action<T*> action_type;
                    return hpx::async<action_type>(gid,mem_size).get();
                }

                template <typename T>
                static hpx::lcos::future<T*> device_malloc_async(hpx::naming::id_type const& gid,size_t mem_size)
                {
                    typedef typename server::device::device_malloc_action<T*> action_type;
                    return hpx::async<action_type>(gid,mem_size);
                }

                //functions to run CUDA kernels
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
