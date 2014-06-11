#include <iostream>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/util.hpp>
#include "cuda_components/device.hpp"

using namespace hpx::cuda;

int main(int argc,char** argv)
{
  return hpx::init();
}

int hpx_main()
{
 vector<hpx::naming::id_type> localities = hpx::find_all_localities();
 vector<int> devices = device.get_all_devices(localities);
 typedef hpx::cuda::server::device device_type;
 hpx::cuda::device device(
    hpx::components::new_<device_type>(hpx::find_here()));
 hpx::util::high_resolution_timer t;
 std::cout << "Before calling wait " << std::endl;
 hpx::lcos::future<int> f = device.wait();
 std::cout << "After calling wait" << std::endl;
 int x = f.get();
 std::cout << "Time" << t.elapsed() << std::endl;
 std::cout << "Value of x " <<  x << std::endl;
 return hpx::finalize();
}
