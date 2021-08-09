// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_UTIL_SERVER_DEFINITIONS_HPP_
#define HPX_OPENCL_SERVER_UTIL_SERVER_DEFINITIONS_HPP_

#define HPX_OPENCL_REGISTER_ACTION_DECLARATION(component_name, action_name) \
  HPX_ACTION_USES_MEDIUM_STACK(                                             \
      hpx::opencl::server::component_name::action_name##_action);           \
  HPX_REGISTER_ACTION_DECLARATION(                                          \
      hpx::opencl::server::component_name::action_name##_action,            \
      hpx_opencl_##component_name##_##action_name##_action)

#define HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(component_name,          \
                                                     action_name)             \
  namespace hpx {                                                             \
  namespace traits {                                                          \
  template <typename T>                                                       \
  struct action_stacksize<                                                    \
      hpx::opencl::server::component_name::action_name##_action<T>,           \
      typename util::always_void<                                             \
          typename hpx::opencl::server::component_name::action_name##_action< \
              T>::type>::type> {                                              \
    enum { value = hpx::threads::thread_stacksize_medium };                   \
  };                                                                          \
  }                                                                           \
  }

#endif
