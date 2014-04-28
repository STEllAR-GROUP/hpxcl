// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include "timer.hpp"
#include "work_queue.hpp"
//#include "fifo.hpp"

#include <string>

void async_func(work_queue<int> *wq, int i)
{

    hpx::cout << "Worker # " << i << " started." << hpx::endl;
    
    int workload;

    while(wq->request(&workload))
    {
        hpx::cout << "Worker # " << i << ": " << workload << hpx::endl;
        wq->deliver(workload);
    }

    hpx::cout << "Worker # " << i << " ended." << hpx::endl;

}


int hpx_main(int argc, char* argv[])
{

    // Print help message on wrong argument count
    if(argc < 2)
    {
        hpx::cerr << "Usage: " << argv[0] << " matrixsize" << hpx::endl;
        return hpx::finalize();
    }


    {

        work_queue<int> wq;

        for(int i = 0; i < 10; i++)
        {
            hpx::apply(&async_func, &wq, i);
        }

        wq.add_work(5);
        wq.add_work(7);
        wq.add_work(3);
        wq.finish();

        int t;
        while(wq.retrieve_finished_work(&t))
            hpx::cout << "Finished: " << t << hpx::endl;

        hpx::cout << "Everything done." << hpx::endl;

    }

    hpx::cout << "Program finished." << hpx::endl;
   
    // End the program
    return hpx::finalize();

}

