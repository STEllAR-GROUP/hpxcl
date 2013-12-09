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

    template <size_t DIM>
    struct work_size
    {
        private:
        struct dimension
        {
            size_t offset;
            size_t size;
            size_t local_size;
            dimension(){
                offset = 0;
                size = 0;
                local_size = 0;
            }
        };
        private:
            // local_size be treated as NULL if all dimensions have local_size == 0
            dimension dims[DIM];
        public:
            dimension& operator[](size_t idx){ return dims[idx]; }
    };
    

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
            set_arg(cl_uint arg_index, hpx::opencl::buffer arg) const;

            hpx::lcos::future<void>
            set_arg_async(cl_uint arg_index, hpx::opencl::buffer arg) const;
            
            // Runs the kernel
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size) const;

            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    hpx::opencl::event event) const;
            
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    std::vector<hpx::opencl::event> events) const;

            // Runs the kernel with hpx::opencl::work_size
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size) const;
            
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
                    hpx::opencl::event event) const;

            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
                    std::vector<hpx::opencl::event> events) const;


    };

    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue(hpx::opencl::work_size<DIM> dim,
                    std::vector<hpx::opencl::event> events) const
    {

        // Casts everything to pointers
        size_t global_work_offset[DIM];
        size_t global_work_size[DIM];
        size_t local_work_size_[DIM];
        size_t *local_work_size = NULL;

        // Write work_size to size_t arrays
        for(size_t i = 0; i < DIM; i++)
        {
            global_work_offset[i] = dim[i].offset;
            global_work_size[i] = dim[i].size;
            local_work_size_[i] = dim[i].local_size;
        }

        // Checks for local_work_size == NULL
        for(size_t i = 0; i < DIM; i++)
        {
            if(local_work_size_[i] != 0)
            {
                local_work_size = local_work_size_;
                break;
            }
        }

        // run with casted parameters
        return enqueue(DIM, global_work_offset, global_work_size,
                       local_work_size, events);

    }

    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue(hpx::opencl::work_size<DIM> size,
                    hpx::opencl::event event) const
    {
        // Create vector with events
        std::vector<hpx::opencl::event> events(1);
        events[0] = event;

        // Run
        return enqueue(size, events);
    }

    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue(hpx::opencl::work_size<DIM> size) const
    {
        // Create vector with events
        std::vector<hpx::opencl::event> events(0);

        // Run
        return enqueue(size, events);
    }





}}



#endif
