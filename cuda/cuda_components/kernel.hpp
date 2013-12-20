// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(KERNEL_1_HPP)
#define KERNEL_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        class kernel
            : public hpx::components::client_base<
                kernel, stubs::kernel >
        {
            typedef hpx::components::client_base<
                kernel, stubs::kernel>
                base_type;

            public:
                kernel()
                {}

                kernel(hpx::future<hpx::naming::id_type> const& gid)
                : base_type(gid)
                {}
        }
    }
}
#endif //KERNEL_1_HPP
