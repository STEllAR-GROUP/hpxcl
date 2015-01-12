#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include <string>

#include  "../fwd_declarations.hpp"

#include "kernel.hpp"
#include "../buffer.hpp"

using namespace hpx::cuda::server;

kernel::kernel(){}

kernel::kernel(std::string kernel_name)
{
    this->kernel_name = kernel_name;
}

kernel::~kernel(){}
            	
void kernel::set_stream()
{
    cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT);
}
                
void kernel::set_grid_dim(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
{
    this->grid.x = grid_x;
    this->grid.y = grid_y;
    this->grid.z = grid_z;
}

void kernel::set_block_dim(unsigned int block_x, unsigned int block_y, unsigned int block_z)
{
    this->block.x = block_x;
    this->block.y = block_y;
    this->block.z = block_z;
}

void kernel::load_module(std::string file_name)
{
    this->module_name = file_name;
}

void kernel::load_kernel(std::string kernel_name)
{
    this->kernel_name = kernel_name;
}

std::string kernel::get_function()
{
    return this->kernel_name;
}

std::string kernel::get_module()
{
    return this->module_name;
}

kernel::Dim3 kernel::get_grid()
{
    return this->grid;
}

kernel::Dim3 kernel::get_block()
{
    return this->block;
}

void kernel::set_arg(size_t size, hpx::cuda::buffer cu_buffer)
{
    ///sets a buffer component as the argument to a kernel
}