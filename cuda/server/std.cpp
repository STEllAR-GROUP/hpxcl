// Copyright (c)    2014 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "std.hpp"
#include "../tools.hpp"
#include "../device.hpp"

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/static.hpp>
#include <hpx/runtime.hpp>

#include <vector>
#include <string>

////STATIC FUNCTIONS

using hpx::lcos::spinlock;
struct global_device_list_tag {};
static bool device_list_initialized = false;
static bool device_shutdown_hook_initialized = false;

typedef
hpx::util::static_<std::vector<hpx::cuda::device>,
					global_device_list_tag> static_device_list_type;

typedef
hpx::util::static_<spinlock,
					global_device_list_tag> static_device_list_lock_type;

static void clear_device_list()
{
	//lock the list
	static_device_list_type device_lock;
	boost::locK_guard<spinlock> lock(device_lock.get());

	static_device_list_lock_type device_lock;
	boost::lock_guard<spinlock> lock(device_lock.get());

	static_device_list_type devices;
	devices.get().clear();
}

HPX_REGISTER_PLAIN_ACTION(hpx::cuda::server::get_devices_action, 
							cuda_get_devices_action);

std::vector<hpx::cuda::device>
hpx::cuda::server::get_devices()
{
	static_device_list_lock_type device_lock;
	boost::locK_guard<spinlock> lock(device_lock.get());
	static_device_list_type devices;

	std::vector<hpx::cuda::device> suitable_devices;
	BOOST_FOREACH(const std::vector<hpx::cuda::device>::value_type& device,
		devices.get())
	{
		suitable_devices.push_back(device);
	}
	//return the devices found
	return suitable_devices;
}