
// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_MEMORY_HPP__
#define HPX_OPENCL_MEMORY_HPP__


#include <hpx/include/components.hpp>

#include "server/memory.hpp"

namespace hpx {
namespace opencl {


    class memory
      : public hpx::components::client_base<
          memory, hpx::components::stub_base<server::memory>
        >
    { 

        typedef hpx::components::client_base<
            memory, hpx::components::stub_base<server::memory>
            > base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            memory(){}

            // Constructor
            memory(hpx::future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}


    };

}}







#endif

