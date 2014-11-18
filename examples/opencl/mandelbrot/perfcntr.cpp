#include "perfcntr.hpp"


namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

void
perfcntr_t::init(std::vector<std::string> gpu_names_)
{
    gpu_names = gpu_names_;
    num_gpus = gpu_names_.size();
    
    for(int i = 0; i<num_gpus; i++)
    {
        num_pixels_calculated[i].store(0);
    }

    start_time = boost::posix_time::microsec_clock::universal_time();
}

void
perfcntr_t::submit(size_t gpuid, unsigned long num_pixels)
{
    num_pixels_calculated[gpuid].fetch_add(num_pixels);
}

std::vector<unsigned long>
perfcntr_t::get_counters()
{
    std::vector<unsigned long> res(num_gpus);
    for(size_t i = 0; i < num_gpus; i++)
    {
        res[i] = num_pixels_calculated[i].load();
    }
    return res;
}

std::vector<std::string>
perfcntr_t::get_gpu_names()
{
    return gpu_names;
}

void
perfcntr_t::set_gpu_name(size_t gpuid, std::string gpu_name)
{
    gpu_names[gpuid] = gpu_name;
}

long
perfcntr_t::get_current_time()
{
    boost::posix_time::ptime now =
        boost::posix_time::microsec_clock::universal_time();

    boost::posix_time::time_duration diff = now - start_time;

    return (long)diff.total_milliseconds();
}


perfcntr_t perfcntr;

}}}}
