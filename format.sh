#!/bin/bash

##############################################################################
#  Copyright (c) 2019 Prashant K. Jha
#  Copyright (c) 2019 Patrick Diehl
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
##############################################################################

if [ ! -f ".clang-format" ]; then

	echo "Generate clang format configuration file"
	clang-format -style=Google -dump-config > .clang-format 
    sed -i 's/SortIncludes:    true/SortIncludes:    false/g' .clang-format
fi

echo "Formating header files"
find -name "*.hpp" -exec clang-format  -i "{}" ";"

echo "Formating source files"
find -name "*.cpp" -exec clang-format -i "{}" ";"
