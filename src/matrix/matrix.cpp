/*
 * matrix.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: dragoon
 */

#include "matrix.h"
#include <stdlib.h>
#include <algorithm>

namespace matrix {

using std::copy;

Matrix::Matrix(size_t rows_count, size_t columns_count) {
	rows_count_ = rows_count;
	columns_count_ = columns_count;
	init();
}

Matrix::~Matrix() {
	if (rows_count_) {
		for (size_t i = 0; i != rows_count_; i++) {
			delete[] matrix_[i];
		}
		delete[] matrix_;
	}
}

Matrix* Matrix::clone() {
	Matrix *cloned_matrix = new Matrix(rows_count_, columns_count_);
	cloned_matrix->matrix_ = new double*[rows_count_];
	for (size_t i = 0; i != rows_count_; i++) {
		cloned_matrix->matrix_[i] = new double[columns_count_];
		copy(matrix_[i], matrix_[i] + columns_count_, cloned_matrix->matrix_[i]);
	}
	return cloned_matrix;
}

size_t Matrix::rows_count() {
	return rows_count_;
}
size_t Matrix::columns_count() {
	return columns_count_;
}

double* Matrix::operator [](const size_t index) {
	return matrix_[index];
}

Matrix& Matrix::operator +=(const double &value) {
	for (size_t i = 0; i != rows_count_; i++) {
		for (size_t j = 0; j != columns_count_; j++) {
			matrix_[i][j] += value;
		}
	}
	return *this;
}

Matrix& Matrix::operator -=(const double &value) {
	(*this) += -value;
	return *this;
}

/*
 * Non standart multiplication operator. Multiply row on rows, return one row
 */
void Matrix::multiply_row_with_rows_to_row(const double *first, Matrix &second,
		double *destination) {
	for (size_t i = 0; i != second.rows_count_; i++) {
		destination[i] = 0;
		for (size_t j = 0; j != second.columns_count_; j++) {
			destination[i] += first[j] * second[i][j];
		}
	}
}

Matrix* Matrix::create_rand_matrix(size_t rows_count, size_t columns_count) {
	Matrix *result_matrix = new Matrix(rows_count, columns_count);
	for (size_t i = 0; i != rows_count; i++) {
		for (size_t j = 0; j != columns_count; j++) {
			(*result_matrix)[i][j] = rand() / (double) RAND_MAX;
		}
	}
	return result_matrix;
}

void Matrix::init() {
	matrix_ = new double*[rows_count_];
	for (size_t i = 0; i != rows_count_; i++) {
		matrix_[i] = new double[columns_count_];
	}
}

} /* namespace matrix */
