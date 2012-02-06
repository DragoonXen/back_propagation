/*
 * matrix.h
 *
 *  Created on: Feb 4, 2012
 *      Author: dragoon
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>

namespace matrix {

class Matrix {

public:
	Matrix(size_t rows_count, size_t columns_count);
	virtual ~Matrix();

	Matrix* clone();

	size_t rows_count();
	size_t columns_count();

	double* operator [] (const size_t index);
	Matrix& operator +=(const double &value);
	Matrix& operator -=(const double &value);

	static void multiply_row_with_rows_to_row(const double *first, Matrix &second, double *destination);
	static Matrix* create_rand_matrix(size_t rows_count, size_t columns_count);

private:
	void init();

	size_t rows_count_;
	size_t columns_count_;
	double** matrix_;
};

} /* namespace matrix */
#endif /* MATRIX_H_ */
