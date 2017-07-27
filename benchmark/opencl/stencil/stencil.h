// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define SINGLE

//To switch between single and double precision
#ifdef SINGLE
#define TYPE float
#else
#define TYPE double
#endif