// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/include/util.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/version.h>
#include <boost/make_shared.hpp>
#include <string>
#include <iostream>

#include "event.hpp"


using namespace hpx::cuda::server;

event::event()
{
    cuEventCreate(&cu_event, CU_EVENT_DEFAULT);
    cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT);  
} 

event::event(int event_id, unsigned int stream_id,unsigned int event_flag, unsigned int stream_flag)
{
    this->event_id = event_id;
    this->stream_id = stream_id;
    cuEventCreate(&cu_event,event_flag);
    //Valid flags are 
    //CU_EVENT_DEFAULT
    //CU_EVENT_BLOCKING_SYNC
    //CU_EVENT_DISABLE_TIMING
    cuStreamCreate(&cu_stream, stream_flag);
    //valid flags are 
    //CU_STREAM_DEFAULT
    //CU_STREAM_NON_BLOCKING
}

//Create a CUDA Event object with flags
event::event(unsigned int event_id,unsigned int stream_id)
{      
  	this->stream_id = stream_id;
    this->event_id = event_id;
   	cuEventCreate(&cu_event, CU_EVENT_DEFAULT);
    cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT);
}
//Destroy the CUDA Event
event::~event()
{
    cuEventDestroy(cu_event);
}

CUevent event::cuda_event()
{
	return this->cu_event;
}

//Wrapper for the cudaEventSynchronize
//this function will be called by an io-threadpool thread to prevent the 
//blocking of an hpx thread 
static void 
event::hpx_cudaEventSynchronize(CUevent cu_event,
    boost::shared_ptr<hpx::lcos::local::promise<int> > p)
     {
        //wait for the given event to complete
        cuEventSynchronize(cu_event);

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
bool event::finished() const
{
    CUresult error = cuEventQuery(cu_event);
    if(error == CUDA_ERROR_NOT_READY)
        return false;
    else
        return true;
}

//Triggers the event
void event::trigger()
{
    //trigger the event on the parent device
    cuEventRecord(cu_event,cu_stream);
} 
