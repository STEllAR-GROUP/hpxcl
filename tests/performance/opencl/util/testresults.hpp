// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <map>
#include <vector>
#include <iostream>


#ifndef HPX_OPENCL_TESTS_PERFORMANCE_TESTRESULTS_HPP_
#define HPX_OPENCL_TESTS_PERFORMANCE_TESTRESULTS_HPP_

namespace hpx{ namespace opencl{ namespace tests{ namespace performance{

    class testresults
    {
        private:
            class testseries{
                public:
                    std::vector<double> test_entries;
                    std::string series_name;
                    std::map<std::string, std::string> atts;
                    std::string unit;
                    double get_median() const;
                    double get_mean() const;
                    double get_stddev() const;
                    double get_min() const;
                    double get_max() const;
            };

            std::vector<testseries> results;

        public:
            void start_test( std::string name,
                             std::string unit,
                             std::map<std::string, std::string> atts
                                = std::map<std::string, std::string>() );

            bool add( double result );

        private:
            friend std::ostream& operator<<( std::ostream& os,
                                             const testresults& result );

    };

    std::ostream& operator<<(std::ostream& os, const testresults& result);

}}}}
    
    
#endif
