// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include "cuda/program.hpp"

typedef hpx::components::managed_component<
    hpx::cuda::server::program>
    cuda_program_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_program_type, cuda_program);

HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::build_action,
    cuda_program_build_action);
HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::set_source_action,
    cuda_program_set_source_action);
HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::run_action,
    cuda_program_run_action);
