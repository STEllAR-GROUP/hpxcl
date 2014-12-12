// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(EVENT_2_HPP)
#define EVENT_2_HPP 

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/io_service_pool.hpp>

#include <cuda.h>

#include  "../fwd_declarations.hpp"

#include "../device.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace server
        {
            class event
                : public hpx::components::locking_hook<
                    hpx::components::managed_component_base<event>
                >
            {  
            	private:
                boost::shared_ptr<device> parent_device;
                unsigned int parent_device_id;
                unsigned int stream_id;
                unsigned int event_id;
                CUevent cu_event;
                CUstream cu_stream;
            	public:
       
                event(); 

            	event(int event_id, unsigned int stream_id,unsigned int event_flag, unsigned int stream_flag);
                //Create a CUDA Event object with flags
                event(unsigned int event_id,unsigned int stream_id);
                //Destroy the CUDA Event
                ~event();

                CUevent cuda_event();

                //Wrapper for the cudaEventSynchronize
                //this function will be called by an io-threadpool thread to prevent the 
                //blocking of an hpx thread 
                static void 
                hpx_cudaEventSynchronize(CUevent cu_event,
                    boost::shared_ptr<hpx::lcos::local::promise<int> > p);

                //waits for an event to happen
                void await() const;

                //Retruns true if the event is already finished
                bool finished() const;

                //Triggers the event
                void trigger();

                //define event class actions
                HPX_DEFINE_COMPONENT_ACTION(event,await);
                HPX_DEFINE_COMPONENT_ACTION(event,finished);
                HPX_DEFINE_COMPONENT_ACTION(event,trigger);
                HPX_DEFINE_COMPONENT_ACTION(event,cuda_event);

            };
        }
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::await_action,
    cuda_event_await_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::finished_action,
    cuda_event_finished_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::trigger_action,
    cuda_event_trigger_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::cuda_event_action,
    cuda_event_cuda_event_action);

#endif
