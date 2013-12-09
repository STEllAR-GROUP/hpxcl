#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/device.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
	hpx::cuda::server::device
	> cuda_device_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_device_type,device);

HPX_REGISTER_ACTION(
	cuda_device_type::wrapped_type::test1_action,
	cuda_device_test1_action);
HPX_REGISTER_ACTION(
	cuda_device_type::wrapped_type::test2_action,
	cuda_device_test2_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::calculate_pi_action,
    cuda_device_calculate_pi_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_cuda_info_action,
    cuda_device_get_cuda_info_action);

