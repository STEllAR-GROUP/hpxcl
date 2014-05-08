// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_KERNEL_HPP__
#define HPX_OPENCL_KERNEL_HPP__

#include "server/kernel.hpp"

#include <hpx/include/components.hpp>
#include <hpx/lcos/when_all.hpp>
#include <boost/serialization/vector.hpp>

#include "event.hpp"
#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl {

    ////////////////////////
    /// @brief Kernel execution dimensions.
    ///
    /// This structure offers an alternative way to set and reuse kernel 
    /// execution dimensions.
    /// 
    /// Example:
    /// \code{.cpp}
    ///     // Create work_size object
    ///     hpx::opencl::work_size<1> dim;                                           
    ///
    ///     // Set dimensions. 
    ///     dim[0].offset = 0;                                                       
    ///     dim[0].size = 2048; 
    ///     
    ///     // Set local work size.
    ///     // This can be left out.
    ///     // OpenCL will then automatically determine the best local work size.
    ///     dim[0].local_size = 64;
    ///
    ///     // Enqueue a kernel using the work_size object
    ///     event kernel_event = kernel.enqueue(dim).get();
    ///
    /// \endcode
    ///
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
    
    /////////////////////////
    /// @brief An OpenCL kernel.
    ///
    /// This represents one specific OpenCL task that can be directly executed
    /// on an OpenCL device.
    ///
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

            kernel(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            // ///////////////////////////////////////
            //  Exposed Component functionality
            //  


            /**
             *  @brief Sets a kernel argument
             *
             *  @param arg_index    The argument index to which the buffer will
             *                      be connected.
             *  @param arg          The \ref buffer that will be connected.
             */
            void
            set_arg(cl_uint arg_index, hpx::opencl::buffer arg) const;

            /**
             *  @brief Sets a kernel argument
             *
             *  This is the non-blocking version of \ref set_arg.
             *
             *  @param arg_index    The argument index to which the buffer will
             *                      be connected.
             *  @param arg          The \ref buffer that will be connected.
             *  @return             A future that will trigger upon completion.
             */
            hpx::lcos::future<void>
            set_arg_async(cl_uint arg_index, hpx::opencl::buffer arg) const;
            
            // Runs the kernel
            /**
             *  @name Starts execution of a kernel.
             *
             *  @param work_dim     The number of dimensions the kernel should
             *                      get executed in.
             *  @param global_work_offset   The offset id with which to start
             *                              the execution.<BR>
             *                              This needs to be a pointer to a
             *                              work_dim-dimensional array.
             *  @param global_work_size     The total number of work-items per
             *                              dimensions on which the kernel
             *                              will be executed.<BR>
             *                              This needs to be a pointer to a
             *                              work_dim-dimensional array.
             *  @param local_work_size      The size of one OpenCL work-group.
             *                              <BR> This needs to be a pointer to a
             *                              work_dim-dimensional array, or NULL
             *                              for being set automatically by the
             *                              OpenCL runtime.
             *  @return             An \ref event that triggers upon completion.
             */
            //@{
            /**
             *  @brief Starts kernel immediately
             */
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size) const;

            /**
             *  @brief Depends on an event
             *
             *  This is an overloaded version of \ref enqueue with the
             *  possibility to add an event as dependency.
             *
             *  The kernel will not execute before the event triggered.
             *  
             *  @param event    The \ref event to wait for.
             */
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    hpx::opencl::event event) const;
            
            /**
             *  @brief Depends on multiple events
             *
             *  This is an overloaded version of \ref enqueue with the
             *  possibility to add multiple events as dependency.
             *
             *  The kernel will not execute before the events triggered.
             *  
             *  @param events   The \ref event "events" to wait for.
             */
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                    std::vector<hpx::opencl::event> events) const;

            /**
             *  @brief Depends on one future event
             *
             *  The kernel will not execute before the future event tirggered.
             *
             *  @param event    The future \ref event to wait for.
             */
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
                             hpx::lcos::shared_future<hpx::opencl::event> event) const;

            /**
             *  @brief Depends on multiple future events
             *
             *  The kernel will not execute before the future events triggered.
             *
             *  @param events   The future \ref event "events" to wait for.
             */
            hpx::lcos::future<hpx::opencl::event>
            enqueue(cl_uint work_dim,
                    const size_t *global_work_offset,
                    const size_t *global_work_size,
                    const size_t *local_work_size,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const;
             //@}

            // Runs the kernel with hpx::opencl::work_size
            /**
             *  @name Starts execution of a kernel, using work_size.
             *
             *  This is an overloaded version of \ref enqueue that takes a 
             *  \ref hpx::opencl::work_size for convenience purposes.
             *
             *  @param size     The work dimensions on which the kernel should
             *                  get executed on.
             *  @return         An \ref event that triggers upon completion.
             */
            //@{
            /**
             *  @brief Starts kernel immediately
             */
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size) const;
            
            /**
             *  @brief Depends on an event
             *
             *  The kernel will not execute before the event triggered.
             *
             *  @param event    The \ref event to wait for.
             */
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
                    hpx::opencl::event event) const;

            /**
             *  @brief Depends on multiple events
             *
             *  The kernel will not execute before the events triggered.
             *
             *  @param events   The \ref event "events" to wait for.
             */
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
                    std::vector<hpx::opencl::event> events) const;

            /**
             *  @brief Depends on one future event
             *
             *  The kernel will not execute before the future event tirggered.
             *
             *  @param event    The future \ref event to wait for.
             */
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
                             hpx::lcos::shared_future<hpx::opencl::event> event) const;

            /**
             *  @brief Depends on multiple future events
             *
             *  The kernel will not execute before the future events triggered.
             *
             *  @param events   The future \ref event "events" to wait for.
             */
            template<size_t DIM>
            hpx::lcos::future<hpx::opencl::event>
            enqueue(hpx::opencl::work_size<DIM> size,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const;
             //@}

        private:
            // LOCAL HELPER CALLBACK FUNCTIONS
            template<size_t DIM>
            static
            hpx::lcos::future<hpx::opencl::event>
            enqueue_future_single_callback_tpl(kernel cl,
                                              hpx::opencl::work_size<DIM> size,
                            hpx::lcos::shared_future<hpx::opencl::event> event);
 
            template<size_t DIM>
            static
            hpx::lcos::future<hpx::opencl::event>                                
            enqueue_future_multi_callback_tpl(kernel cl,
                            hpx::opencl::work_size<DIM> size,
                            hpx::lcos::future<std::vector<
                                hpx::lcos::shared_future<hpx::opencl::event>
                                                          >> futures);

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
    
    // Callback for template function for a single future event
    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue_future_single_callback_tpl(kernel cl,
                                              hpx::opencl::work_size<DIM> size,
                            hpx::lcos::shared_future<hpx::opencl::event> event)
    {
        return cl.enqueue(size, event.get());
    }


    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue(hpx::opencl::work_size<DIM> size,
                    hpx::lcos::shared_future<hpx::opencl::event> event) const
    {
        return event.then(                                                      
            hpx::util::bind(                                                
                    &(enqueue_future_single_callback_tpl<DIM>),
                    *this,                                                  
                    size,
                    util::placeholders::_1
            )                                                               
        );                                                                      
    
    }

    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>                                
    kernel::enqueue_future_multi_callback_tpl(kernel cl,
                hpx::opencl::work_size<DIM> size,    
               hpx::lcos::future<std::vector<                            
                            hpx::lcos::shared_future<hpx::opencl::event>        
                                                                >> futures)     
    {                                                                           
                                                                                
        /* Get list of futures */                                               
        std::vector<hpx::lcos::shared_future<hpx::opencl::event>>               
        futures_list = futures.get();                                           
                                                                                
        /* Create list of events */                                             
        std::vector<hpx::opencl::event> events;
        events.reserve(futures_list.size());            
                                                                                
        /* Put events into list */                                              
        BOOST_FOREACH(hpx::lcos::shared_future<hpx::opencl::event> & future,    
                        futures_list)                                           
        {                                                                       
            events.push_back(future.get());                                     
        }                                                                       
                                                                                
        /* Call actual function */                                              
        return cl.enqueue(size, events);                         
                                                                                
    }                                                                           
  

    template<size_t DIM>
    hpx::lcos::future<hpx::opencl::event>
    kernel::enqueue(hpx::opencl::work_size<DIM> size,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
    {
        return hpx::when_all(events).then(                                      
            hpx::util::bind(                                                    
                &(enqueue_future_multi_callback_tpl<DIM>),
                *this,
                size,
                util::placeholders::_1
            )
        );
    }

}}



#endif
