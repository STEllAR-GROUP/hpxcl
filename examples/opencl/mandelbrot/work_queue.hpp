// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_WORK_QUEUE_H_
#define MANDELBROT_WORK_QUEUE_H_

#include "fifo.hpp"


/**
 * @brief A thread safe multi-consumer-buffer
 *
 * The purpose of this class is to receive workloads from one master and 
 * queue it until one of many workers request it.
 * The workers will then compute the workload and return it to this class,
 * so the master can receive the computed workload
 *
 * Therefore, this class needs to be completely threadsafe.
 */
template <typename T>
class work_queue
{

public:

    /**
     * @brief Sends an undone workload to a worker.
     *
     * Gets called by the workers.
     *
     * @param wp Returns a workload that needs computation
     * @return False on end of work
     */
    bool request(T* wp);

    /**
     * @brief Hands in a finished workload from a worker
     *
     * Gets called by the workers.
     *
     * @param done_workload The ready computed workload 
     */
    void deliver(const T &done_workload);

    /**
     * @brief Adds undone workloads to the work pool.
     *
     * Gets called by the master.
     * 
     * Calling this function after finish() will lead to
     * undefined behaviour.
     *
     * @param undone_workload A new workload
     */
    void add_work(const T &undone_workload);

    /**
     * @brief Retrieves a finished work packet
     *  
     * Gets called by the master.
     *
     * @param done_workload Returns a finished workload
     * @return false to signal all work done
     */
    bool retrieve_finished_work(T* done_workload);

    /**
     * @brief Signals that all work is done
     *
     * Gets called by the master.
     */
    void finish();
public:
    work_queue();

private:
    // holds the undone work
    fifo<T> unfinished_work;

    // holds the done work
    fifo<T> finished_work;
    
    // saves how much work is left
    std::atomic_size_t num_work;

    // is true as the end-of-work-signal arrives
    std::atomic_bool finished;

};


template<typename T>
work_queue<T>::work_queue()
{

    num_work = 0;
    finished = false;
}

template<typename T>
void work_queue<T>::add_work(const T &undone_workload)
{
    
    // Make sure queue is not funished yet
    BOOST_ASSERT(!finished);

    // Add the workload packet
    unfinished_work.push(undone_workload);

}

template<typename T>
bool work_queue<T>::request(T* undone_workload)
{

    // Store number of workloads that are currently active
    // needs to be done before retrieving work packet to prefent race condition
    num_work++;

    // get new work packet
    if(!unfinished_work.pop(undone_workload))
    {
        // set input queue state to finished.
        // from now on we will only wait for returned packets.
        // as soon as the num_work is zero, we know everything is done.
        finished = true;

        // decrease work counter as we couldn't get a work packet
        size_t work_left = --num_work;

        // if all work packets got returned, close the result queue
        if(work_left == 0)
        {
            finished_work.finish();
        }

        // stop worker, no more work to be done
        return false;
    }

    // successfully aquired new work packet.
    return true;

}

template<typename T>
void work_queue<T>::deliver(const T &done_workload)
{
    
    // add to finished queue
    finished_work.push(done_workload);

    // check wether all work packets got returned
    size_t work_left = --num_work;

    // if all work packets got returned, close the result queue
    if(finished && work_left == 0)
    {
        finished_work.finish();
    }

}


template<typename T>
bool work_queue<T>::retrieve_finished_work(T* done_workload)
{
    
    return finished_work.pop(done_workload);

}

template<typename T>
void work_queue<T>::finish()
{

    // close unfinished work queue
    unfinished_work.finish();
    // Working with atomics, no sync necessary
   // finished = true;

}









#endif 
