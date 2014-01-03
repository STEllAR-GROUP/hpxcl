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

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

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
                hpx::naming::id_type parent_device_id;
                unsigned int event_id;
                cudaEvent_t* event;
                cudaStream_t stream;
                unsigned int stream_id;

            	public:
                void check_error(cudaError_t error)
                {
                    switch(error)
                    {
                        case  cudaErrorInitializationError:
                            std::cout<<"Initialization error"<<std::endl;
                        break;
                        case cudaErrorInvalidValue:
                            std::cout<<"Invalid value for device"<<std::endl;
                        break; 
                        case cudaErrorLaunchFailure:
                            std::cout<<"Event Launch failure"<<std::endl;
                        break; 
                        case cudaErrorMemoryAllocation:
                            std::cout<<"Event memory allocation error"<<std::endl;
                        break;
                    }

                }

                //Event constructors 
            	event(int event_id,unsigned int flags,stream_id)
            	{
                    this->event_id = event_id;
                    this->stream_id = stream_id;
                    cudaError_t error = cudaEventCreateWithFlags(&this->event,flags);
                    //Valid flags are 
                    //cudaEventDefault, cudaEventBlockingSync, 
                    //cudaEventDisableTiming, cudaEventInterprocess
                    check_error(error);
                }
                //Create a CUDA Event object with flags
                event(int event_id,stream_id)
                {      
                    this->stream_id = stream_id;
                    this->event_id = event_id;
                    cudaError_t error = cudaEventCreate(&this->event);
                    check_error(error);

                }
                //Destroy the CUDA Event
                ~event()
                {
                    cudaError_t error = cudaEventDestroy(this->event);
                    check_error(error);
                }

                //Wrapper for the cuda wait for events
                //this function will be called by an io-threadpool thread to prevent the 
                //blocking of an hpx thread 
                static void hpx_cudaEventSynchronize(unsigned int event_id,boost::shared_ptr<hpx::lcos::local::promise<int>> p)
                {
                    //wait for the given event to complete
                    cudaError_t error = cudaEventSynchronize(this->event);

                    //return the error code via future
                    p->set_value(error);
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
                    pool->get_io_services().post(
                        hpx::util::bind(&hpx_cudaEventSynchronize,p));

                    //wait for event to finish
                    cudaError_t error = p->get_future().get();
                    check_error(error);
                }

                //Retruns true if the event is already finished
                bool finished() const
                {
                    cudaError_t error = cudaEventQuery(this->event);
                    check_error(error);
                    if(error == cudaErrorNotReady)
                        return false;
                    else
                        return true;
                }

                //Triggers the event
                void trigger()
                {
                    //trigger the event on the parent device
                    cudaError_t error = cudaEventRecord(&this->event,this->stream_id)
                    check_error(error);
                } 

                //define event class actions
                HPX_DEFINE_COMPONENT_ACTION(event,await);
                HPX_DEFINE_COMPONENT_ACTION(event,finished):
                HPX_DEFINE_COMPONENT_ACTION(event,trigger);

            };
        }
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::await,
    event_await_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::finished,
    event_finished_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::event::trigger,
    event_trigger_action);

//event registration declarations
#endif
