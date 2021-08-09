#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;

#define SIZE 8

int main(int argc, char* argv[]) {
  auto start = std::chrono::steady_clock::now();

  std::vector<hpx::lcos::future<void>> data_futures;

  std::vector<device> devices = get_all_devices(2, 0).get();

  if (devices.size() < 1) {
    hpx::cerr << "No CUDA devices found!" << hpx::endl;
    return hpx::finalize();
  }

  int* input_1;
  cudaMallocHost((void**)&input_1, sizeof(int) * SIZE);
  checkCudaError("Malloc inputData");

  for (int i = 0; i < SIZE; i++) {
    input_1[i] = 1;
  }

  for (int i = 0; i < SIZE - 1; i++) {
    std::cout << input_1[i] << ", ";
  }
  std::cout << input_1[SIZE - 1] << std::endl;

  device cudaDevice_1 = devices[0];

  buffer inbuffer_1 = cudaDevice_1.create_buffer(sizeof(int) * SIZE).get();

  data_futures.push_back(
      inbuffer_1.enqueue_write(0, sizeof(int) * SIZE, input_1));

  program prog_1 =
      cudaDevice_1.create_program_with_file("p2p_test_kernel.cu").get();

  device cudaDevice_2 = devices[1];

  buffer inbuffer_2 = cudaDevice_2.create_buffer(sizeof(int) * SIZE).get();

  program prog_2 =
      cudaDevice_2.create_program_with_file("p2p_test_kernel.cu").get();

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Add compiler flags for compiling the kernel
  std::vector<std::string> flags_1;
  std::string mode_1 = "--gpu-architecture=compute_";
  mode_1.append(
      std::to_string(cudaDevice_1.get_device_architecture_major().get()));
  mode_1.append(
      std::to_string(cudaDevice_1.get_device_architecture_minor().get()));
  flags_1.push_back(mode_1);

  std::vector<std::string> flags_2;
  std::string mode_2 = "--gpu-architecture=compute_";
  mode_2.append(
      std::to_string(cudaDevice_2.get_device_architecture_major().get()));
  mode_2.append(
      std::to_string(cudaDevice_2.get_device_architecture_minor().get()));
  flags_2.push_back(mode_2);

  // Compile the program
  prog_1.build_sync(flags_1, "multiply_1");
  prog_2.build_sync(flags_2, "multiply_2");
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int* output_1;
  cudaMallocHost((void**)&output_1, sizeof(int) * SIZE);
  checkCudaError("Malloc result");

  buffer outbuffer_1 = cudaDevice_1.create_buffer(sizeof(int) * SIZE).get();
  data_futures.push_back(
      outbuffer_1.enqueue_write(0, sizeof(int) * SIZE, output_1));

  // Generate the grid and block dim
  hpx::cuda::server::program::Dim3 grid;
  hpx::cuda::server::program::Dim3 block;

  // Set the values for the grid dimension
  grid.x = 1;
  grid.y = 1;
  grid.z = 1;

  // Set the values for the block dimension
  block.x = 1;
  block.y = 1;
  block.z = 1;

  int* n;
  cudaMallocHost((void**)&n, sizeof(int));
  n[0] = SIZE;

  buffer sizebuffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();
  data_futures.push_back(sizebuffer_1.enqueue_write(0, sizeof(int), n));

  // Set the parameter for the kernel, have to be the same order as in the
  // definition
  std::vector<hpx::cuda::buffer> args_1;
  args_1.push_back(inbuffer_1);
  args_1.push_back(outbuffer_1);
  args_1.push_back(sizebuffer_1);

  hpx::wait_all(data_futures);

  // Run the kernel at the default stream
  auto kernel_future_1 = prog_1.run(args_1, "multiply_1", grid, block, 0);

  hpx::wait_all(kernel_future_1);

  // Copy the result back
  int* res_1 = outbuffer_1.enqueue_read_sync<int>(0, sizeof(int) * SIZE);

  for (int i = 0; i < SIZE - 1; i++) {
    std::cout << res_1[i] << ", ";
  }
  std::cout << res_1[SIZE - 1] << std::endl;

  auto f = outbuffer_1.p2p_copy(inbuffer_2.get_device_pointer().get(),
                                inbuffer_2.get_device_id().get(),
                                sizeof(int) * SIZE);

  f.get();

  int* output_2;
  cudaMallocHost((void**)&output_2, sizeof(int) * SIZE);
  checkCudaError("Malloc result");

  buffer outbuffer_2 = cudaDevice_2.create_buffer(sizeof(int) * SIZE).get();
  data_futures.push_back(
      outbuffer_2.enqueue_write(0, sizeof(int) * SIZE, output_2));

  buffer sizebuffer_2 = cudaDevice_2.create_buffer(sizeof(int)).get();
  data_futures.push_back(sizebuffer_2.enqueue_write(0, sizeof(int), n));

  // Set the parameter for the kernel, have to be the same order as in the
  // definition
  std::vector<hpx::cuda::buffer> args_2;
  args_2.push_back(inbuffer_2);
  args_2.push_back(outbuffer_2);
  args_2.push_back(sizebuffer_2);

  hpx::wait_all(data_futures);

  // Run the kernel at the default stream
  auto kernel_future_2 = prog_2.run(args_2, "multiply_2", grid, block, 0);

  hpx::wait_all(kernel_future_2);

  // Copy the result back
  int* res_2 = outbuffer_2.enqueue_read_sync<int>(0, sizeof(int) * SIZE);

  for (int i = 0; i < SIZE - 1; i++) {
    std::cout << res_2[i] << ", ";
  }
  std::cout << res_2[SIZE - 1] << std::endl;

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

  return EXIT_SUCCESS;
}
