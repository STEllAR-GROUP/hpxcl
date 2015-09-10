// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/server/program.hpp"
#include "cuda/kernel.hpp"

#include <sstream>
#include <string>

namespace hpx {
namespace cuda {
namespace server {

program::program() {
}

program::program(hpx::naming::id_type device_id, std::string code) {
}

program::program(hpx::naming::id_type device_id,
		hpx::serialization::serialize_buffer<char> binary) {
}

program::~program() {
}

void program::set_source(std::string source) {
	this->kernel_source = source;
}
hpx::cuda::kernel program::create_kernel(std::string module_name,
		std::string kernel_name) {
	typedef hpx::cuda::server::kernel kernel_type;

	hpx::cuda::kernel cu_kernel(
			hpx::components::new_ < kernel_type > (hpx::find_here()));
	cu_kernel.load_module_sync(module_name);
	cu_kernel.load_kernel_sync(kernel_name);
	return cu_kernel;
}


void program::build(std::vector<std::string> compilerFlags) {

	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	std::string filename = to_string(uuid);
	filename.append(".cu");

	nvrtcCreateProgram(&(this->prog), this->kernel_source.c_str(),
			filename.c_str(), 0, NULL, NULL);



	//nvrtcResult compileResult = nvrtcCompileProgram(prog,1,opts);
}

}
}
}

