// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/version.hpp>

#include <hpxcl/cuda.hpp>

using namespace hpx::cuda;

///////////////////////////////////////////////////////////////////////////////
double mysecond() {
	return hpx::util::high_resolution_clock::now() * 1e-9;
}

///////////////////////////////////////////////////////////////////////////////
int checktick() {
	static const std::size_t M = 20;
	int minDelta, Delta;
	double t1, t2, timesfound[M];

	// Collect a sequence of M unique time values from the system.
	for (std::size_t i = 0; i < M; i++) {
		t1 = mysecond();
		while (((t2 = mysecond()) - t1) < 1.0E-6)
			;
		timesfound[i] = t1 = t2;
	}

	// Determine the minimum difference between these M values.
	// This result will be our estimate (in microseconds) for the
	// clock granularity.
	minDelta = 1000000;
	for (std::size_t i = 1; i < M; i++) {
		Delta = (int) (1.0E6 * (timesfound[i] - timesfound[i - 1]));
		minDelta = (std::min)(minDelta, (std::max)(Delta, 0));
	}

	return (minDelta);
}

///////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<double> > run_benchmark(size_t iterations,
		size_t size) {

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(1, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
	}

	//Get the cuda device
	device cudaDevice = devices[0];

	//Compile the kernels

	// Create the hello_world device program
	program prog = cudaDevice.create_program_with_file("kernels.cu");

	//Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));

	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);
	flags.push_back("-use_fast_math");

	std::vector<std::string> kernels;
	kernels.push_back("multiply_step");
	kernels.push_back("add_step");
	kernels.push_back("triad_step");

	// Compile the program
	hpx::wait_all(prog);
	auto fProg = prog.build(flags, kernels);

	//Fill the vector
	double* a;
	double* b;
	double* c;
	size_t* s;
	double* factor;

	cudaMallocHost((void**) &a, sizeof(double) * size);
	memset((void*) a, 1.0, sizeof(double) * size);
	cudaMallocHost((void**) &b, sizeof(double) * size);
	memset((void*) b, 2.0, sizeof(double) * size);
	cudaMallocHost((void**) &c, sizeof(double) * size);
	memset((void*) c, 3.0, sizeof(double) * size);
	cudaMallocHost((void**) &s, sizeof(size_t));
	cudaMallocHost((void**) &factor, sizeof(double));
	s[0] = size;
	factor[0] = 2.;

	//Allocate device buffer
	buffer aBuffer = cudaDevice.create_buffer(size * sizeof(double));
	buffer bBuffer = cudaDevice.create_buffer(size * sizeof(double));
	buffer cBuffer = cudaDevice.create_buffer(size * sizeof(double));
	buffer sizeBuffer = cudaDevice.create_buffer(sizeof(size_t));
	buffer fBuffer = cudaDevice.create_buffer(sizeof(double));

	//Fill device buffer
	hpx::wait_all(aBuffer);
	auto fa = aBuffer.enqueue_write(0, size * sizeof(double), a);
	hpx::wait_all(bBuffer);
	auto fb = bBuffer.enqueue_write(0, size * sizeof(double), b);
	hpx::wait_all(cBuffer);
	auto fc = cBuffer.enqueue_write(0, size * sizeof(double), c);
	hpx::wait_all(sizeBuffer);
	auto fsize = sizeBuffer.enqueue_write(0, sizeof(size_t), s);
	hpx::wait_all(fBuffer);
	auto ffactor = fBuffer.enqueue_write(0, sizeof(double), factor);

	futures.push_back(std::move(fa));
	futures.push_back(std::move(fsize));
	futures.push_back(std::move(fProg));
	futures.push_back(std::move(ffactor));

	//Prepare kernel launch
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	//Set the values for the grid dimension
	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	//Set the values for the block dimension
	block.x = 32;
	block.y = 1;
	block.z = 1;

	//Prepare buffer arguments
	std::vector<hpx::cuda::buffer> args;
	args.push_back(sizeBuffer);
	args.push_back(aBuffer);
	args.push_back(aBuffer);
	args.push_back(fBuffer);

	hpx::wait_all(futures);

	// Check clock ticks ...
	double t = mysecond();
	auto fk = prog.run(args, "multiply_step", grid, block);
	hpx::wait_all(fk);
	t = 1.0E6 * (mysecond() - t);

	// Get initial value for system clock.
	int quantum = checktick();
	if (quantum >= 1) {
		std::cout << "Your clock granularity/precision appears to be "
				<< quantum << " microseconds.\n";
	} else {
		std::cout
				<< "Your clock granularity appears to be less than one microsecond.\n";
		quantum = 1;
	}

	std::cout << "Each test below will take on the order" << " of " << (int) t
			<< " microseconds.\n" << "   (= " << (int) (t / quantum)
			<< " clock ticks)\n"
			<< "Increase the size of the arrays if this shows that\n"
			<< "you are not getting at least 20 clock ticks per test.\n"
			<< "-------------------------------------------------------------\n";

	std::cout << "WARNING -- The above is only a rough guideline.\n"
			<< "For best results, please be sure you know the\n"
			<< "precision of your system timer.\n"
			<< "-------------------------------------------------------------\n";

	std::vector<std::vector<double> > timing(4,
			std::vector<double>(iterations));

	factor[0] = 3.;
	ffactor = fBuffer.enqueue_write(0, sizeof(double), factor);
	hpx::wait_all(std::move(fk));

	for (std::size_t iteration = 0; iteration != iterations; ++iteration) {

		// Copy
		timing[0][iteration] = mysecond();
		auto fcopy = cBuffer.enqueue_write(0, size * sizeof(double), a);
		hpx::wait_all(std::move(fcopy));
		timing[0][iteration] = mysecond() - timing[0][iteration];

		// Scale
		timing[1][iteration] = mysecond();
		args.clear();
		args.push_back(sizeBuffer);
		args.push_back(cBuffer);
		args.push_back(bBuffer);
		args.push_back(fBuffer);
		fk = prog.run(args, "multiply_step", grid, block);
		hpx::wait_all(fk);
		timing[1][iteration] = mysecond() - timing[1][iteration];

		// Add
		timing[2][iteration] = mysecond();
		args.clear();
		args.push_back(sizeBuffer);
		args.push_back(aBuffer);
		args.push_back(bBuffer);
		args.push_back(cBuffer);
		fk = prog.run(args, "add_step", grid, block);
		hpx::wait_all(fk);
		timing[2][iteration] = mysecond() - timing[2][iteration];

		// Triad
		timing[3][iteration] = mysecond();
		args.clear();
		args.push_back(sizeBuffer);
		args.push_back(bBuffer);
		args.push_back(cBuffer);
		args.push_back(aBuffer);
		args.push_back(fBuffer);
		fk = prog.run(args, "add_step", grid, block);
		timing[3][iteration] = mysecond() - timing[3][iteration];

	}

	return timing;
}

///////////////////////////////////////////////////////////////////////////////
//Main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char*argv[]) {

	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	size_t size = atoi(argv[1]);
	size_t iterations = atoi(argv[2]);

	std::cout
	<< "-------------------------------------------------------------\n"
	<< "Modified STREAM bechmark based on\nHPX version: "
	<< hpx::build_string() << "\n"
	<< "-------------------------------------------------------------\n"
	<< "This system uses " << sizeof(double)
	<< " bytes per array element.\n"
	 << "Memory per array = "
	            << sizeof(double) * (size / 1024. / 1024.) << " MiB "
	        << "(= "
	            <<  sizeof(double) * (size / 1024. / 1024. / 1024.)
	            << " GiB).\n"
	<< "-------------------------------------------------------------\n"
	<< "Each kernel will be executed " << iterations << " times.\n"
	<< " The *best* time for each kernel (excluding the first iteration)\n"
	<< " will be used to compute the reported bandwidth.\n"
	<< "-------------------------------------------------------------\n"
	<< "Number of Threads requested = "
	<< hpx::get_os_thread_count() << "\n"
	<< "-------------------------------------------------------------\n";

	double time_total = mysecond();
	std::vector<std::vector<double> > timing;
	timing = run_benchmark(10,size);
	time_total = mysecond() - time_total;

	/* --- SUMMARY --- */
	const char *label[4] = {
		"Copy:      ",
		"Scale:     ",
		"Add:       ",
		"Triad:     "
	};

	const double bytes[4] = {
		2 * sizeof(double) * size,
		2 * sizeof(double) * size,
		3 * sizeof(double) * size,
		3 * sizeof(double) * size
	};

	// Note: skip first iteration
	std::vector<double> avgtime(4, 0.0);
	std::vector<double> mintime(4, (std::numeric_limits<double>::max)());
	std::vector<double> maxtime(4, 0.0);
	for(std::size_t iteration = 1; iteration != iterations; ++iteration)
	{
		for (std::size_t j=0; j<4; j++)
		{
			avgtime[j] = avgtime[j] + timing[j][iteration];
			mintime[j] = (std::min)(mintime[j], timing[j][iteration]);
			maxtime[j] = (std::max)(maxtime[j], timing[j][iteration]);
		}
	}

	printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
	for (std::size_t j=0; j<4; j++) {
		avgtime[j] = avgtime[j]/(double)(iterations-1);

		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
				1.0E-06 * bytes[j]/mintime[j],
				avgtime[j],
				mintime[j],
				maxtime[j]);
	}

	std::cout
	<< "\nTotal time: " << time_total
	<< " (per iteration: " << time_total/iterations << ")\n";

	std::cout
	<< "-------------------------------------------------------------\n"
	;

	return EXIT_SUCCESS;
}
