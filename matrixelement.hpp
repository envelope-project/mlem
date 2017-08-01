/**
    Copyright © 2017 Tilman Kuestner
    Authors: Tilman Kuestner
           Dai Yang
           Josef Weidendorfer

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
    OTHER DEALINGS IN THE SOFTWARE.

    The above copyright notice and this permission notice shall be 
    included in all copies and/or substantial portions of the Software,
    including binaries.
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
