// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define SINGLE
#define EPS 10e-5

//###########################################################################
//Switching between single precision and double precision
//###########################################################################

#ifdef SINGLE
#define TYPE float
#else
#define TYPE double
#endif
