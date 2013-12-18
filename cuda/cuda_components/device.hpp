#if !defined(DEVICE_1_HPP)
#define DEVICE_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/device.hpp"

namespace hpx
{
    namespace cuda
    {
        class device
            : public hpx::components::client_base<
                device, stubs::device >
        {
            typedef hpx::components::client_base<
                device, stubs::device
                > base_type;

        public:
            device()
            {}

            device(hpx::future<hpx::naming::id_type> const& gid)
			: base_type(gid)
            {}

            void get_cuda_info()
            {
                BOOST_ASSERT(this->get_gid());
                this->base_type::get_cuda_info(this->get_gid());
            }

            static std::vector<int> get_all_devices(std::vector<hpx::naming::id_type> localities)
            {
                return base_type::get_all_devices(localities);
            }

            void set_device(int dev)
            {
                BOOST_ASSERT(this->get_gid());
                this->base_type::set_device(this->get_gid(),dev);
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
}
#endif //MANAGED_CUDA_COMPONENT_1_HPP
