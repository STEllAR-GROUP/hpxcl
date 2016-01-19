/*
 * utils.hpp
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstdlib>
#include <ctime>

template<typename T>
void fillRandomVector(T* matrix, size_t size) {
	srand(time(NULL));

	for (size_t i = 0; i < size; i++) {

		matrix[i] = (T) (0.5) * ((T) rand()) / (T) RAND_MAX;
	}

}

template<typename T>
bool checkStencil(T*in,T*out,T* s, size_t size)
{
	bool check = true;
	for (size_t i = 1; i < size -1; ++i)
	{
		T res = s[0] * in[i - 1] + s[1] * in[i] + s[2] * in[i + 1];

		if(res-out[i] >= EPS) check = false;


	}

	return check;
}

#endif /* UTILS_HPP_ */
