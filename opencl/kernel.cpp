// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "kernel.hpp"

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "buffer.hpp"
#include "event.hpp"


using namespace hpx::opencl;

void
kernel::set_arg(cl_uint arg_index, buffer arg) const
{

    set_arg_async(arg_index, arg).get();

}

hpx::lcos::unique_future<void>
kernel::set_arg_async(cl_uint arg_index, buffer arg) const
{
    
    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::kernel::set_arg_action func;

    return hpx::async<func>(this->get_gid(), arg_index, arg);

}

hpx::lcos::unique_future<hpx::opencl::event>
kernel::enqueue(cl_uint work_dim,
                const size_t *global_work_offset_ptr,
                const size_t *global_work_size_ptr,
                const size_t *local_work_size_ptr) const
{
 
    std::vector<hpx::opencl::event> events(0);
    return enqueue(work_dim, global_work_offset_ptr, global_work_size_ptr,
                   local_work_size_ptr, events);

}


hpx::lcos::unique_future<hpx::opencl::event>
kernel::enqueue(cl_uint work_dim,
                const size_t *global_work_offset_ptr,
                const size_t *global_work_size_ptr,
                const size_t *local_work_size_ptr,
                hpx::opencl::event event) const
{
 
    std::vector<hpx::opencl::event> events;
    events.push_back(event);
    return enqueue(work_dim, global_work_offset_ptr, global_work_size_ptr,
                   local_work_size_ptr, events);

}

hpx::lcos::unique_future<hpx::opencl::event>
kernel::enqueue(cl_uint work_dim,
                const size_t *global_work_offset_ptr,
                const size_t *global_work_size_ptr,
                const size_t *local_work_size_ptr,
                hpx::lcos::shared_future<hpx::opencl::event> event) const
{
    // Create vector with event
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events;
    events.push_back(std::move(event));

    // Forward call
    return enqueue(work_dim, global_work_offset_ptr,
                   global_work_size_ptr,
                   local_work_size_ptr, std::move(events));
}

hpx::lcos::unique_future<hpx::opencl::event>
kernel::enqueue(cl_uint work_dim,
                const size_t *global_work_offset_ptr,
                const size_t *global_work_size_ptr,
                const size_t *local_work_size_ptr,
           std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
{
/*    
    // define the async call
    future_call_def_4(kernel, cl_uint, const size_t *, const size_t *,
                      const size_t *, enqueue);

    // run the async call
    return future_call::run(*this, work_dim, global_work_offset_ptr,
                            global_work_size_ptr, local_work_size_ptr, events);

*/
    return unique_future<hpx::opencl::event>();
}


hpx::lcos::unique_future<hpx::opencl::event>
kernel::enqueue(cl_uint work_dim,
                const size_t *global_work_offset_ptr,
                const size_t *global_work_size_ptr,
                const size_t *local_work_size_ptr,
                std::vector<hpx::opencl::event> events) const
{
    
    BOOST_ASSERT(this->get_gid());



    // Serialize global_work_offset
    std::vector<size_t> global_work_offset(0);
    if(global_work_offset_ptr != NULL)
    {
        global_work_offset.reserve(work_dim);
        for(cl_uint i = 0; i < work_dim; i++)
        {
            global_work_offset.push_back(global_work_offset_ptr[i]);
        }
    }
    
    // Serialize global_work_size
    std::vector<size_t> global_work_size(0);
    if(global_work_size_ptr != NULL)
    {
        global_work_size.reserve(work_dim);
        for(cl_uint i = 0; i < work_dim; i++)
        {
            global_work_size.push_back(global_work_size_ptr[i]);
        }
    }
    
    // Serialize local_work_size
    std::vector<size_t> local_work_size(0);
    if(local_work_size_ptr != NULL)
    {
        local_work_size.reserve(work_dim);
        for(cl_uint i = 0; i < work_dim; i++)
        {
            local_work_size.push_back(local_work_size_ptr[i]);
        }
    }
    
    // Create arguments vector
    std::vector<std::vector<size_t>> args;
    args.reserve(3);
    args.push_back(global_work_offset);
    args.push_back(global_work_size);
    args.push_back(local_work_size);

    // Invoke server call
    typedef hpx::opencl::server::kernel::enqueue_action func;
    return hpx::async<func>(this->get_gid(), work_dim,
                                             args,
                                             events);

}





