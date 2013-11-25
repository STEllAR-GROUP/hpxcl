// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
//#include <hpx/include/components.hpp>

#include <vector>

#include <boost/foreach.hpp>

#include "../../opencl/device.hpp"
#include "../../opencl/buffer.hpp"
#include "../../opencl/program.hpp"
#include "../../opencl/kernel.hpp"

void sayhellofunc()
{

	hpx::cout << hpx::get_locality_id() << " - " << hpx::get_worker_thread_num() << " : " << hpx::find_here() << hpx::endl;

}

HPX_PLAIN_ACTION(sayhellofunc, sayhello);


int hpx_main(int argc, char* argv[])
{


	std::vector<hpx::opencl::clx_device_id> devices
            = hpx::opencl::get_device_ids(hpx::find_here(),
             //           CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR);
                        CL_DEVICE_TYPE_ALL, 1.1f).get();

    if(devices.size() < 1)
    {
        devices = hpx::opencl::get_device_ids(hpx::find_here(),
                        CL_DEVICE_TYPE_CPU, 1.1f).get();
    }


    hpx::cout << "#Devices: " << devices.size() << hpx::endl;
   
    for(size_t i = 0; i < devices.size(); i++)
    {
        std::vector<char> name;
        name = hpx::opencl::get_device_info(hpx::find_here(), devices[i], CL_DEVICE_NAME
                                     ).get();
        hpx::cout << "\t" << i << ": " << &name[0] << " ~ ";
        name = hpx::opencl::get_device_info(hpx::find_here(), devices[i], CL_DEVICE_VERSION
                                     ).get();
        hpx::cout << &name[0] << hpx::endl;
    }
   
    size_t gpuid;
    std::cin >> gpuid;

    {
    	typedef hpx::opencl::server::device device_type;
    	hpx::opencl::device cldevice(
              hpx::components::new_<device_type>(hpx::find_here(), devices[gpuid]));

        #define datasize 10000
        std::vector<char> databuf(datasize);
        char *datain = &databuf[0];
        for(int i = 0; i < datasize; i++)
        {
            datain[i] = i;
        }

        hpx::opencl::buffer buffer = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                                      datasize, datain);
        hpx::opencl::buffer buffer2 = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                                      datasize);
        hpx::opencl::event event = buffer.enqueue_read(datasize-10, 10,
                                    std::vector<hpx::opencl::event>(0)).get() ;
        char datain2[] = {3,3};

        hpx::opencl::event event2 = buffer.enqueue_write(datasize - 3, 2, datain2, event).get();
        hpx::lcos::future<hpx::opencl::event> event3 = buffer.enqueue_read(datasize - 10, 10, event2);

        boost::shared_ptr<std::vector<char>> data
                                        = event.get_data().get();
        boost::shared_ptr<std::vector<char>> data2
                                        = event3.get().get_data().get();
        for(size_t i = 0; i < data->size(); i++)
        {
            hpx::cout << "~" << (int)data->at(i);
        }
        hpx::cout << hpx::endl;
        for(size_t i = 0; i < data2->size(); i++)
        {
            hpx::cout << "~" << (int)data2->at(i);
        }
        hpx::cout << hpx::endl;

        std::string program_src = 
        "__kernel void test(__global char * in, __global char * out)  \n"
        "{                                                            \n"
        "    size_t tid = get_global_id(0);                           \n"
        "    char var = in[tid];                                      \n"
        "    for(int i = 0; i < 10000; i++){                          \n"
        "         var+=(i%7);                                         \n"
        "         var%=13;                                            \n"
        "    }                                                        \n"
        "    out[tid] = var;                                          \n" 
        "}                                                            \n";

        hpx::opencl::program prog = cldevice.create_program_with_source(program_src);
        prog.build();

        hpx::opencl::kernel kernel = prog.create_kernel("test");

        kernel.set_arg(0, buffer);
        kernel.set_arg(1, buffer2);

       
        size_t offset = 0;
        size_t size = datasize;
        hpx::cout << "enqueue" << hpx::endl;
        hpx::opencl::event event4 = kernel.enqueue((cl_uint)1, &offset, &size, (size_t*)NULL, event3.get()).get();
        hpx::cout << "\tdone" << hpx::endl;
        hpx::cout << "read" << hpx::endl;
        hpx::opencl::event event5 = buffer2.enqueue_read(datasize-10, 10, event4).get();
//        hpx::opencl::event event5 = buffer2.enqueue_read(datasize-10, 10).get();
        hpx::cout << "\tdone" << hpx::endl;
        boost::shared_ptr<std::vector<char>> data3
                                        = event5.get_data().get();
        hpx::cout << "\tfinished" << hpx::endl;
        for(size_t i = 0; i < data3->size(); i++)
        {
            hpx::cout << "~" << (int)data3->at(i);
        }
        hpx::cout << hpx::endl;

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
