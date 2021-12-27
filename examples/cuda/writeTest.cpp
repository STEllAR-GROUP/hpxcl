#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;


int main (int argc, char* argv[]) {
	std::cout << "Start" << std::endl;
	auto start = std::chrono::steady_clock::now();

	std::vector<hpx::lcos::future<void>> data_futures;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	device cudaDevice = devices[0];

	int* array;
	cudaMallocHost((void**)&array, sizeof(int) * 8);
	checkCudaError("Malloc array");

	for(int i = 0; i < 8; i++){
		array[i] = 1;
	}


	buffer array_buffer = cudaDevice.create_buffer(sizeof(int) * 8).get();
	data_futures.push_back(array_buffer.enqueue_write(0, sizeof(int) * 8, array));



	
	program prog = cudaDevice.create_program_with_file("writeTest_kernel.cu").get();

	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	prog.build_sync(flags, "writeTest");


	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	block.x = 1;
	block.y = 1;
	block.z = 1;


	std::vector<hpx::cuda::buffer> args;
	args.push_back(array_buffer);



	int* small_array;
	cudaMallocHost((void**)&small_array, sizeof(int) * 4);
	checkCudaError("Malloc small_array");


	small_array[0] = 8;
	small_array[1] = 8;
	small_array[2] = 16;
	small_array[3] = 16;



	data_futures.push_back(array_buffer.enqueue_write_parcel(sizeof(int) * 4, sizeof(int) * 2, small_array, sizeof(int) * 1));

//(size_t dst_offset, size_t size, hpx::serialization::serialize_buffer<char> data, size_t src_offset) {


	hpx::wait_all(data_futures);


	std::cout << "Before kernel launch" << std::endl;
	auto kernel_future = prog.run(args, "writeTest", grid, block, 0);
	kernel_future.get();
	std::cout << "After kernel launch" << std::endl;

	int* res;
	cudaMallocHost((void**)&res, sizeof(int) * 8);
	res = array_buffer.enqueue_read_sync<int>(0, sizeof(int) * 8);

	for(int i = 0; i < 8; i++){
		std::cout << res[i] << std::endl;
	}


	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}
