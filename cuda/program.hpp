// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(PROGRAM_1_HPP)
#define PROGRAM_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/program.hpp"

namespace hpx
{
    namespace cuda
    {
        class program
            : public hpx::components::client_base<
                program, stubs::program >
        {
            typedef hpx::components::client_base<
                program, stubs::program>
                base_type;

            public:
                program()
                {}

                program(hpx::future<hpx::naming::id_type> && gid)
                : base_type(std::move(gid))
                {}

                hpx::lcos::future<void> build()
                {
                    HPX_ASSERT(this->get_gid());
                    return base_type::build(this->get_gid());
                }

                void build_sync()
                {
                    HPX_ASSERT(this->get_gid());
                    base_type::build_sync(this->get_gid());
                }
        };
    }
}
#endif //PROGRAM_1_HPP
