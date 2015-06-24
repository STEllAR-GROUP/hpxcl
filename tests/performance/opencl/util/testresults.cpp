// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "testresults.hpp"

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>

#include <algorithm>

using hpx::opencl::tests::performance::testresults;

void
testresults::set_enabled_tests( std::vector<std::string> enabled_tests )
{

    enabled_tests_set.insert(enabled_tests.begin(), enabled_tests.end());

}

void
testresults::set_output_json( bool enable )
{

    output_json = enable;

}

void
testresults::start_test( std::string name,
                         std::string unit,
                         std::map<std::string, std::string> atts )
{

    std::cerr << "Running '" << name << "' " << std::flush;

    // If test is allowed to run
    if( enabled_tests_set.empty() ||
                      enabled_tests_set.find(name) != enabled_tests_set.end()) {

        testseries new_test;
    
        new_test.series_name = name;
        new_test.atts = atts;
        new_test.unit = unit;
    
        results.push_back(new_test);

        current_test_valid = true;

    } else {
    
        std::cerr << "- disabled" << std::endl; 
        current_test_valid = false;

    }
}

void
testresults::add( double result )
{
    std::cerr << "." << std::flush;
    results.back().test_entries.push_back(result);
    if(results.back().test_entries.size() >= 10){
        std::cerr << std::endl;
    }
}

bool
testresults::needs_more_testing()
{
    if(!current_test_valid)
        return false;

    if(results.back().test_entries.size() >= 10)
        return false;

    return true;
}

std::ostream&
hpx::opencl::tests::performance::operator<<(std::ostream& os, const testresults& result)
{
    // print headline
    os << std::endl;
    os << "test\tatts\tunits\tmedian\tmean\tstddev\tmin\tmax\ttrial0\ttrial1\t"
       << "trial2\ttrial3\ttrial4\ttrial5\ttrial6\ttrial7\ttrial8\ttrial9"
       << std::endl;

    for(const auto& row : result.results){

        if(row.test_entries.size() != 10)
        {
            os << "ERROR! test_entries.size() != 10!" << std::endl;
            break;
        }

        os << row.series_name << "\t";

        // TODO atts
        os << "TODO" << "\t";

        os << row.unit << "\t";

        os << row.get_median() << "\t";
        os << row.get_mean() << "\t";
        os << row.get_stddev() << "\t";
        os << row.get_min() << "\t";
        os << row.get_max() << "\t";

        bool is_first = true;
        for(const auto& res : row.test_entries){
            if(is_first){
                is_first = false;
            } else {
                os << "\t";
            }
            os << res;
        }

        os << std::endl;

    }

    return os;
}

double
testresults::testseries::get_min() const
{
    double min = test_entries[0];
    
    for(const double& val : test_entries){
        if(val < min)
            min = val;
    }

    return min;
}

double
testresults::testseries::get_max() const
{
    double max = test_entries[0];
    
    for(const double& val : test_entries){
        if(val > max)
            max = val;
    }

    return max;
}

double
testresults::testseries::get_stddev() const
{
    double mean = get_mean();

    double sum = 0;

    for(const double& val : test_entries){
        double diff = val - mean;
        sum += (diff*diff);
    }

    return sum/test_entries.size();
}

double
testresults::testseries::get_mean() const
{

    double sum = 0;
    
    for(const double& val : test_entries){
        sum += val;
    }

    return sum/test_entries.size();
}

double
testresults::testseries::get_median() const
{

    double median;
    std::size_t size = test_entries.size();

    if(size == 0)
        return 0;

    if(size == 1)
        return test_entries[0];

    std::vector<double> sorted_entries(test_entries);
    std::sort(sorted_entries.begin(), sorted_entries.end());

    if (size  % 2 == 0)
    {
        median = (sorted_entries[size / 2 - 1] + sorted_entries[size / 2]) / 2;
    }
    else 
    {
        median = sorted_entries[size / 2];
    }

    return median;
}
