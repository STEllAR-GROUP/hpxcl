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

            hpx::lcos::future<long> test2_async()
            {
                BOOST_ASSERT(this->git_gid());
                return this->base_type::test2_async(this->get_gid());
            }

            long test2_sync()
            {
                BOOST_ASSERT(this->get_gid());
                return this->base_type::test2_sync(this->get_gid());
            }

            void get_cuda_info()
            {
                BOOST_ASSERT(this->get_gid());
                base_type::get_cuda_info(this->get_gid());
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
