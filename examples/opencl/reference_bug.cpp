// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/apply.hpp>

#include <boost/atomic.hpp>



class ref_ctr{
    
    private:
        static boost::atomic<int> num_refs;
    
    public:
        ref_ctr()
        {
            std::cout << this << " ref_ctr() :  " << (1 + num_refs.fetch_add(1, boost::memory_order_seq_cst)) << std::endl;
        }
    
        ref_ctr(const ref_ctr &other)
        {
            std::cout << this << " ref_ctr(&) : " << (1 + num_refs.fetch_add(1, boost::memory_order_seq_cst)) << std::endl;
        }
    
        ref_ctr(ref_ctr &&other)
        {
            std::cout << this << " ref_ctr(m) : " << (1 + num_refs.fetch_add(1, boost::memory_order_seq_cst)) << std::endl;
            
        }


        ~ref_ctr()
        {
            std::cout << this << " ~ref_ctr() : " << (- 1 + num_refs.fetch_sub(1, boost::memory_order_seq_cst)) << std::endl;
        }
    
        int get_num_refs()
        {
            return num_refs.load(boost::memory_order_seq_cst);
        }
    
};

boost::atomic<int> ref_ctr::num_refs(0);

static void test(ref_ctr ctr)
{

    std::cout << &ctr << " test() :     " << ctr.get_num_refs() << std::endl;

}

// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{
    {
       
        ref_ctr ctr;

        hpx::apply(hpx::util::bind(&test, ctr));
        
        std::cout << "before scope end" << std::endl;

    }
    
    std::cout << "after scope end" << std::endl;


    // End the program
    return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[]){

    // initialize HPX, run hpx_main
    hpx::start(argc, argv);

    // wait for hpx::finalize being called
    return hpx::stop();
}


