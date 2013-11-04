
#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
//#include <hpx/include/components.hpp>

#include <vector>

#include <boost/foreach.hpp>

#include "hpxcl/opencl/device.hpp"
#include "hpxcl/opencl/buffer.hpp"

void sayhellofunc()
{

	hpx::cout << hpx::get_locality_id() << " - " << hpx::get_worker_thread_num() << " : " << hpx::find_here() << hpx::endl;

}

HPX_PLAIN_ACTION(sayhellofunc, sayhello);


int hpx_main(int argc, char* argv[])
{


	std::vector<hpx::opencl::clx_device_id> devices
            = hpx::opencl::clGetDeviceIDs(hpx::find_here(),
        //                CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR);
                        CL_DEVICE_TYPE_ALL);

    hpx::cout << "#Devices: " << devices.size() << hpx::endl;
   
    BOOST_FOREACH(
        const std::vector<hpx::opencl::clx_device_id>::value_type& device,
        devices)
    {
        char name[100];
        hpx::opencl::clGetDeviceInfo(hpx::find_here(), device, CL_DEVICE_NAME,
                                     100, (void*)name, NULL);
        hpx::cout << "\t- " << name << hpx::endl;
    }
   
   
    {
    	typedef hpx::opencl::server::device device_type;
    	hpx::opencl::device cldevice(
              hpx::components::new_<device_type>(hpx::find_here(), devices[0]));
        cldevice.get_gid();

//        hpx::opencl::buffer buffer
//        cldevice.test();
        hpx::opencl::buffer buffer = cldevice.clCreateBuffer(CL_MEM_READ_WRITE, 10);
        hpx::cout <<  buffer.get_gid() << hpx::endl;
        hpx::opencl::event event = buffer.clEnqueueReadBuffer(0, 10, std::vector<hpx::opencl::event>(0)).get() ;
        event.get_gid();
        //hpx::cout << "clx_event_id:" << event.get_cl_event_id() << hpx::endl; 
    }
    return hpx::finalize();

/*

	std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

    std::vector<hpx::lcos::future<void>> futures;
    futures.reserve(localities.size());

    BOOST_FOREACH(hpx::naming::id_type const& node, localities)
    {
        sayhello hello;
        futures.push_back(hpx::async(hello, node));
    }

    hpx::lcos::wait(futures);
*/

//	hpx::cout << hpx::find_here() << hpx::endl;
	return hpx::finalize();
}

int main(int argc, char* argv[])
{
	// initialize HPX, run hpx_main.
	hpx::start(argc, argv);

	// wait for hpx::finalize being called.
	return hpx::stop();
}
