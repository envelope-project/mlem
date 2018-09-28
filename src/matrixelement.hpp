/**
	matrixelement.hpp

	Created on: Oct 15, 2009
		Author: kuestner
*/

#pragma once

#include <stdint.h>
#include <iostream>


template <typename T> class MatrixElement {
public:
	MatrixElement() : row_(0), column_(0), value_(0.0) {}
	MatrixElement(uint32_t row, uint32_t column, T value) :
		row_(row), column_(column), value_(value) {}
	uint32_t row() const { return row_; }
	uint32_t column() const { return column_; }
	T value() const { return value_; }
	void setValue(T value) { value_ = value; }
private:
	uint32_t row_;
	uint32_t column_;
	T value_;
};

/// Formatted iostream output
template <typename T>
std::ostream& operator<<(std::ostream& os, const MatrixElement<T>& e)
{
	os<<"("<<e.row()<<", "<<e.column()<<", "<<e.value()<<")";
	return os;
}

template <typename T> class RowElement {
public:
	explicit RowElement(uint32_t column = 0, T value = 0) :
		column_(column), value_(value) {}
	uint32_t column() const { return column_; }
	T value() const { return value_; }
private:
	uint32_t column_;
	T value_;
};

template <typename T>
std::ostream& operator<< (std::ostream &out, const RowElement<T>& elem )
{
    out << "(" << elem.column() << ", " << elem.value() << ")";
    return out;
}
