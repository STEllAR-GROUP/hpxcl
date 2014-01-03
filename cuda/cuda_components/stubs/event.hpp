// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(EVEMT_3_HPP)
#define EVENT_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../sever/event.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct event
                : hpx::components::stub_base<server::event>
            {
            	private:
                //event member functions
            	public:
                    //Empty constructor
                    event(){}
                    //Constructor
                    event(hpx::future<hpx::naming::id_type> const& gid)
                        : base_type(gid)
                        {}

            };
        }
    }
}
#endif
