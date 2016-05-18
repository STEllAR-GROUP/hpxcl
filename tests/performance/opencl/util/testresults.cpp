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
testresults::set_output_json()
{
    output_format = JSON;
}

void
testresults::set_output_tabbed()
{
    output_format = TABBED;
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

static std::size_t
get_column_width( const std::vector<std::vector<std::string> > & matrix,
                  std::size_t column )
{
    std::size_t max_width = 0;

    for(const auto& row : matrix){

        if(row.size() <= column)
            continue;

        if(row[column].length() > max_width)
            max_width = row[column].length();

    }

    return max_width;

}

static std::string
double_to_str(double num)
{
    std::stringstream str;
    str << num;
    return str.str();
}

void
testresults::print_default( std::ostream& os ) const
{
    std::vector<std::vector<std::string> > output;

    std::vector<std::string> headline;
    headline.push_back("test");
    headline.push_back("atts");
    headline.push_back("units");
    headline.push_back("median");
    headline.push_back("mean");
    headline.push_back("stddev");
    headline.push_back("min");
    headline.push_back("max");
    headline.push_back("trial0");
    headline.push_back("trial1");
    headline.push_back("trial2");
    headline.push_back("trial3");
    headline.push_back("trial4");
    headline.push_back("trial5");
    headline.push_back("trial6");
    headline.push_back("trial7");
    headline.push_back("trial8");
    headline.push_back("trial9");
    output.push_back(headline);

    // fill with results
    for(const auto& row : results){

        if(row.test_entries.size() != 10)
        {
            os << "ERROR! test_entries.size() != 10!" << std::endl;
            break;
        }

        std::vector<std::string> line;

        line.push_back(row.series_name);
        line.push_back(row.get_atts());
        line.push_back(row.unit);
        line.push_back(double_to_str(row.get_median()));
        line.push_back(double_to_str(row.get_mean()));
        line.push_back(double_to_str(row.get_stddev()));
        line.push_back(double_to_str(row.get_min()));
        line.push_back(double_to_str(row.get_max()));

        for(const auto& res : row.test_entries){
            line.push_back(double_to_str(res));
        }

        output.push_back(line);
    }

    // compute widths of columns
    std::vector<std::size_t> column_widths;
    for(std::size_t col = 0; col < headline.size(); col++){
        column_widths.push_back(get_column_width(output,col));
    }

    // print
    for(const auto& row : output){
        for(std::size_t i = 0; i < row.size() && i < column_widths.size(); i++){
            if(i != 0)
                os << " ";

            os << row[i];

            for(std::size_t j = row[i].length(); j < column_widths[i]; j++){
                os << " ";
            }
        }
        os << std::endl;
    }

}

void
testresults::print_tabbed( std::ostream& os ) const
{
    // print headline
    os << "test\tatts\tunits\tmedian\tmean\tstddev\tmin\tmax\ttrial0\ttrial1\t"
       << "trial2\ttrial3\ttrial4\ttrial5\ttrial6\ttrial7\ttrial8\ttrial9"
       << std::endl;

    for(const auto& row : results){

        if(row.test_entries.size() != 10)
        {
            os << "ERROR! test_entries.size() != 10!" << std::endl;
            break;
        }

        os << row.series_name << "\t";

        os << row.get_atts() << "\t";

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

}

void
testresults::print_json( std::ostream& os ) const
{
    os << "{" << std::endl;
    os << "  \"tests\": {" << std::endl;

    for(std::size_t i = 0; i < results.size(); i++){
        const auto& row = results[i];
        os << "    \"" << row.series_name << "\": {" << std::endl;

        HPX_ASSERT(row.test_entries.size() == 10);

        // atts
        os << "      \"atts\": {" << std::endl;
        for( auto it = row.atts.begin(); it != row.atts.end(); ){
            os << "        \"" << it->first << "\": \"" << it->second << "\"";

            it++;

            if(it != row.atts.end())
                os << ",";

            os << std::endl;
        }
        os << "      }," << std::endl;

        // unit
        os << "      \"unit\": \"" << row.unit << "\"," << std::endl;

        // stats
        os << "      \"median\": " << row.get_median() << "," << std::endl;
        os << "      \"mean\":   " << row.get_mean() << "," << std::endl;
        os << "      \"stddev\": " << row.get_stddev() << "," << std::endl;
        os << "      \"min\":    " << row.get_min() << "," << std::endl;
        os << "      \"max\":    " << row.get_max() << "," << std::endl;

        // trials
        os << "      \"trials\": [" << std::endl;
        for(std::size_t j = 0; j < row.test_entries.size(); j++){
            const auto& res = row.test_entries[j];

            os << "        " << res;

            if(j < row.test_entries.size() - 1)
                os << "," << std::endl;
            else
                os << std::endl;
        }
        os << "      ]" << std::endl;

        if(i < results.size() - 1)
            os << "    }," << std::endl;
        else
            os << "    }" << std::endl;
    }

    os << "  }" << std::endl;
    os << "}" << std::endl;
}

std::ostream&
hpx::opencl::tests::performance::operator<<(std::ostream& os, const testresults& result)
{

    switch(result.output_format){
        case testresults::DEFAULT: result.print_default(os);    break;
        case testresults::TABBED:  result.print_tabbed(os);     break;
        case testresults::JSON:    result.print_json(os);       break;
        default:
            HPX_ASSERT(false);
            break;
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

std::string
testresults::testseries::get_atts() const
{

    if(atts.empty())
        return "-";

    // get all keys
    std::vector<std::string> keys;
    for( const auto& it : atts ){
        keys.push_back(it.first);
    }

    // sort keys
    std::sort(keys.begin(), keys.end());

    // print values
    std::string result = "";
    for( const auto& key : keys ){
        if(result.size() > 0)
            result += " ";

        result += key;
        result += "=";
        result += atts.at(key);
    }

    return result;

}
