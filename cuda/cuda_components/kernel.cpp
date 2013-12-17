#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::cuda::server::kernel>
    kernel_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(kernel_type,kernel);
