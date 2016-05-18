// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <map>
#include <set>
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
                    std::string get_atts() const;
            };

            std::vector<testseries> results;

        public:
            void set_enabled_tests( std::vector<std::string> enabled_tests );

            void set_output_json();
            void set_output_tabbed();

            void start_test( std::string name,
                             std::string unit,
                             std::map<std::string, std::string> atts
                                = std::map<std::string, std::string>() );

            void add( double result );

            bool needs_more_testing();

        private:
            void print_default( std::ostream& os ) const;
            void print_tabbed( std::ostream& os ) const;
            void print_json( std::ostream& os ) const;

            friend std::ostream& operator<<( std::ostream& os,
                                             const testresults& result );

            bool current_test_valid = false;

            // Settings
            std::set<std::string> enabled_tests_set;

            enum output_formats { DEFAULT, TABBED, JSON };
            output_formats output_format = DEFAULT;

    };

    std::ostream& operator<<(std::ostream& os, const testresults& result);

}}}}
    
    
#endif
