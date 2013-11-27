#include <hpx/hpx_init.hpp>
#include "cuda_components/managed_cuda_component.hpp"
#include <iostream>

int main(int argc,char** argv)
{
    boost::program_options::options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING "[options]");
    desc_commandline.add_options()
        ("info",
            boost::program_options::value<bool>()->default_value(false)
        );

	return hpx::init(desc_commandline,argc,argv);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        bool info = vm["info"].as<bool>();
        typedef cuda_hpx::server::managed_cuda_component cuda_component_type;
        typedef cuda_component_type::argument_type argument_type;

        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        cuda_hpx::managed_cuda_component client(
        hpx::components::new_<cuda_component_type>(localities.back()));
        if(info)
        {
            client.get_cuda_info();
        }
        cuda_hpx::managed_cuda_component client2;
        client2.create(hpx::find_here());
    }
    return hpx::finalize();
}
