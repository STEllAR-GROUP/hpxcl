// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTILS_LOCAL_FIFO_
#define HPX_UTILS_LOCAL_FIFO_

#include <queue>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/condition_variable.hpp>

template <typename T>
class fifo
{

    typedef hpx::lcos::local::spinlock lock_type;
    typedef hpx::lcos::local::condition_variable cond_type;

public:
    // push an item to the queue
    void push(const T &);
    // take an item from the queue, will return false on end-of-program.
    // blocks.
    bool pop(T*);
    // signal end of program
    void finish();

public:
    fifo();
    ~fifo();

private:
    std::queue<T> queue;
    lock_type     lock;
    cond_type     cond_var;

    volatile bool finished;


};

template<typename T>
fifo<T>::fifo()
{
    finished = false;
}

template<typename T>
fifo<T>::~fifo()
{
    finish();
}

template<typename T>
void fifo<T>::push(const T &item)
{

    // lock class
    boost::lock_guard<lock_type> locallock(lock);

    // check wether fifo is already in finished state
    if(finished)
    {
        HPX_THROW_EXCEPTION(hpx::invalid_status, "fifo::push()",
                            "fifo::finish() already called!");
    }

    // push item
    queue.push(item);

    // signal waiting threads that new item is available
    cond_var.notify_one();

}

template<typename T>
bool fifo<T>::pop(T* item)
{

    // lock class
    boost::lock_guard<lock_type> locallock(lock);

    // wait for queue to not be empty
    while(queue.empty())
    {

        // check wether fifo is already in finished state
        if(finished)
            return false;

        // wait for something to change
        cond_var.wait(lock);

    }

    // Retrieve element from queue
    *item = queue.front();

    // Remove element from queue
    queue.pop();

    // Return success
    return true;

}

template<typename T>
void fifo<T>::finish()
{

    // lock class
    boost::lock_guard<lock_type> locallock(lock);

    finished = true;

    cond_var.notify_all();

}

#endif

