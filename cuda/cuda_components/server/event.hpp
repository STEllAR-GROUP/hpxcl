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

#include <cuda_runtime.h>
//#include <cuda.h>
#include <curand_kernel.h>

#include "device.hpp"

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
            {   //event class data members
            	private:
                //event class member functions
                boost::shared_ptr<device> parent_device;
                unsigned int parent_device_id;
                unsigned int stream_id;
                unsigned int event_id;
                cudaEvent_t cu_event;
                cudaStream_t stream;
            	public:
       
                //Event constructors
                event()
                {
                   cudaEventCreate(&cu_event);
                   cudaStreamCreate(&stream); 
                } 

            	event(int event_id, unsigned int stream_id,unsigned int flag)
            	{
                    this->event_id = event_id;
                    this->stream_id = stream_id;
                    cudaEventCreateWithFlags(&cu_event,flag);
                    //Valid flags are 
                    //cudaEventDefault, cudaEventBlockingSync, 
                    //cudaEventDisableTiming, cudaEventInterprocess
                    cudaStreamCreate(&stream);
                }
                //Create a CUDA Event object with flags
                event(unsigned int event_id,unsigned int stream_id)
                {      
                    this->stream_id = stream_id;
                    this->event_id = event_id;
                    cudaEventCreate(&cu_event);
                    cudaStreamCreate(&stream);
                }
                //Destroy the CUDA Event
                ~event()
                {
                    cudaEventDestroy(cu_event);
                }

                //Wrapper for the cudaEventSynchronize
                //this function will be called by an io-threadpool thread to prevent the 
                //blocking of an hpx thread 
                static void 
                hpx_cudaEventSynchronize(cudaEvent_t cu_event,
                    boost::shared_ptr<hpx::lcos::local::promise<int> > p)
                {
                    //wait for the given event to complete
                    cudaEventSynchronize(cu_event);

                    //return the error code via future
                    p->set_value(0);
                }

                //waits for an event to happen
                void await() const
                {
                    //create a promise
                    boost::shared_ptr<hpx::lcos::local::promise<int> > p =
                        boost::make_shared<hpx::lcos::local::promise<int> >();

                    //get a reference to one of the IO specific HPX io_service objects ...
                    hpx::util::io_service_pool* pool =
                        hpx::get_runtime().get_thread_pool("io_pool");

                    //...and schedule the handler to run the cudaEventSynchronize on one 
                    //of the OS-threads.
                    pool->get_io_service().post(
                        hpx::util::bind(&hpx_cudaEventSynchronize,this->cu_event,p));

                    //wait for event to finish
                    p->get_future();
                }

                //Retruns true if the event is already finished
                bool finished() const
                {
                    cudaError_t error = cudaEventQuery(cu_event);
                    if(error == cudaErrorNotReady)
                        return false;
                    else
                        return true;
                }

                //Triggers the event
                void trigger()
                {
                    //trigger the event on the parent device
                    cudaEventRecord(cu_event,stream);
                } 

                /*static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void *data)
                {
                }*/

                /*void callback_func()
                {
                }*/    

                //define event class actions
                HPX_DEFINE_COMPONENT_ACTION(event,await);
                HPX_DEFINE_COMPONENT_ACTION(event,finished);
                HPX_DEFINE_COMPONENT_ACTION(event,trigger);

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

//event registration declarations
#endif
