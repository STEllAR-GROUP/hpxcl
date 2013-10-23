// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#pragma once
#ifndef HPX_OPENCL_CLX_EVENT_HPP
#define HPX_OPENCL_CLX_EVENT_HPP

#include <cstdint>

#include <hpx/lcos/future.hpp>

#include <CL/cl.h>



namespace hpx {
namespace opencl {

    ///////////////////////////////////////////////////////////////////////
    /// This Class is meant to be a replacement for the OpenCL internal
    /// cl_event. It provides similar functionality with the addition of 
    /// improved interaction with hpx::lcos::future.
    //
// TODO read boost::serialize documentation

    typedef intptr_t clx_event_id;

    class clx_event
    {
        
        public:
           clx_event(){}
           // Constructor
           clx_event(hpx::naming::id_type, cl_event event);
           
           /// Converts the clx_event to hpx::lcos::future object
           hpx::lcos::future<void> get_future() const;

           /// Blocks until event has occured
           void await() const;

           /// Returns the GID of the hosting device
           hpx::naming::id_type get_device_gid() const;

           /// Returns the internal cl_event (!Only valid on hosting device!)
           clx_event_id get_cl_event_id() const;
           
        private:
            hpx::naming::id_type hosting_device;
            clx_event_id cl_event_ptr;


        ///////////////////////////////
        /// Serialization
        /// 
        public:
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version)
            {
                ar & hosting_device;
                ar & cl_event_ptr;
            }
    };













}}














#endif 
