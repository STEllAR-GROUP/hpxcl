#if !defined(DEVICE_2_HPP)
#define DEVICE_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/foreach.hpp>

#include "../cuda/kernel.cuh"

#include <iostream>
#include <random>
#include <ctime>

namespace hpx
{
    namespace cuda
    {
        namespace server
        {
         //////////////////////////////////////
         ///This class represents a cuda device

         class device
             : public hpx::components::locking_hook<
                 hpx::components::managed_component_base<device>
                 >
             {
                 public:
				 device() {}
                 //cuda device managedment functions

                 typedef uint64_t argument_type;

                 int get_device_count()
                 {
                    return get_devices();
                 }

                 void set_device(int dev)
                 {
                    set_device(dev);
                 }

                 void get_cuda_info()
                 {
                    get_device_info();
                 }

                 void test1()
                 {
                    std::cout<<"There is/are "<<get_devices()<<" cuda enabled device(s) available"<<std::endl;
                 }

                 //cuda kernel functions
                 long test2()
                 {
                    long a[] = {1};
                    cuda_test(a);
                    return a[0];
                 }

                 float calculate_pi(int nthreads,int nblocks)
                 {
                    return pi(nthreads,nblocks);
                 }

                 HPX_DEFINE_COMPONENT_ACTION(device,test1);
                 HPX_DEFINE_COMPONENT_ACTION(device,test2);
                 HPX_DEFINE_COMPONENT_ACTION(device,calculate_pi);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_cuda_info);
            };
	    }
    }
}

//HPX action declarations

HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::device::test1_action,
	device_test1_action);
HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::device::test2_action,
	device_test2_action);
HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::device::calculate_pi_action,
	device_calculate_pi_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_cuda_info_action,
    device_get_cuda_info_action);

#endif //cuda_device_2_HPP
