/*
 * utils.hpp
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstdlib>
#include <ctime>

template<typename T>
void fillRandomVector(T* matrix, size_t size){

	for (size_t i = 0; i < size; i++){
		srand(time(NULL));
		matrix[i] =  (T)(0.5) * ((T) rand()) / (T) RAND_MAX;
	}


}


#endif /* UTILS_HPP_ */
