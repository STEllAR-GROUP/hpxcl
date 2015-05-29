// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"

#include "../../../opencl/server/util/event_map.hpp"

#include <atomic>

using hpx::naming::id_type;


static hpx::opencl::server::util::event_map *map;

static std::atomic_ulong id_counter(1);
static id_type next_id(){
    return id_type(0, id_counter++, id_type::management_type::unmanaged); 
}

static std::atomic_size_t num_deleted;
static void deletion_callback(cl_event e){
    num_deleted++;
    hpx::cout << "deletion_callback: " << (long) e << hpx::endl;
}

static std::size_t count_deleted(){
    return num_deleted.exchange(0);
}

hpx::future<cl_event> get_async(id_type id){
    return hpx::async(
            [id](){
                return map->get(id);
            });
}

static void cl_test(hpx::opencl::device cldevice)
{

    // Usually: do not use new, use make_shared<>. But in this case,
    // we also want to test the shutdown routine and therefore need
    // explicit deletion
    map = new hpx::opencl::server::util::event_map();

    // Register the desctruction callback
    map->register_deletion_callback(deletion_callback);

    // Test default functionality
    {
        id_type id = next_id();
        cl_event event = (cl_event)id.get_lsb();

        map->add(id, event);
        
        HPX_TEST_EQ(map->get(id), event);

        HPX_TEST_EQ(count_deleted(), 0);
        map->remove(id.get_gid());
        HPX_TEST_EQ(count_deleted(), 1);
    }
        
    // Test reverse get functionality
    {
        id_type id = next_id();
        cl_event event = (cl_event)id.get_lsb();
        id_type id2 = next_id();
        cl_event event2 = (cl_event)id2.get_lsb();

        // Run asynchronous thread
        hpx::future<cl_event> thread1_1 = get_async(id);
        hpx::future<cl_event> thread1_2 = get_async(id);
        hpx::future<cl_event> thread2_1 = get_async(id2);

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
        HPX_TEST(!thread1_1.is_ready());
        HPX_TEST(!thread1_2.is_ready());
        HPX_TEST(!thread2_1.is_ready());

        map->add(id2, event2);

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
        HPX_TEST(!thread1_1.is_ready());
        HPX_TEST(!thread1_2.is_ready());
        HPX_TEST(thread2_1.is_ready());

        map->add(id, event);

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
        HPX_TEST(thread1_1.is_ready());
        HPX_TEST(thread1_2.is_ready());
        HPX_TEST(thread2_1.is_ready());

        HPX_TEST_EQ(thread1_1.get(), event);
        HPX_TEST_EQ(thread1_2.get(), event);
        HPX_TEST_EQ(thread2_1.get(), event2);
        
        HPX_TEST_EQ(count_deleted(), 0);
        map->remove(id.get_gid());
        map->remove(id2.get_gid());
        HPX_TEST_EQ(count_deleted(), 2);
    }
    

    // Test deletion.
    delete map;

}


