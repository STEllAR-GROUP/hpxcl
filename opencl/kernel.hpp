// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_KERNEL_HPP__
#define HPX_OPENCL_KERNEL_HPP__


#include <hpx/include/components.hpp>
#include <boost/serialization/vector.hpp>

#include "server/kernel.hpp"


namespace hpx {
namespace opencl {

    

    class kernel
      : public hpx::components::client_base<
          kernel, hpx::components::stub_base<server::kernel>
        >
    
    {
    
        typedef hpx::components::client_base<
            kernel, hpx::components::stub_base<server::kernel>
            > base_type;

        public:
            kernel(){}

            kernel(hpx::future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            //////////////////////////////////////////
            /// Exposed Component functionality
            /// 

            // Sets buffer as argument for kernel
            void
            set_arg(cl_uint arg_index, hpx::opencl::buffer arg);

            hpx::lcos::future<void>
            set_arg_async(cl_uint arg_index, hpx::opencl::buffer arg);
            
            // Runs the kernel
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size);

            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    hpx::opencl::event event);
            
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    std::vector<hpx::opencl::event> events);



    };

}}



#endif
