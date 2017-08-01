/**
    Copyright © 2017 Tilman Kuestner
    Authors: Tilman Kuestner
           Dai Yang
           Josef Weidendorfer

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

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

#include "matrixelement.hpp"
#include "scannerconfig.hpp"

# include <boost/iterator/iterator_facade.hpp>

#include <string>
#include <assert.h>
#include <stdint.h>

#define MAX_MAPPINGS 28


/// A Matrix in csr4 format
class Csr4Matrix {
public:
    Csr4Matrix(const std::string& fn);
    ~Csr4Matrix();

    uint32_t rows() const { return nRows; }
    uint32_t columns() const { return nColumns; }
    uint64_t elements() const { return nnz; }
    uint32_t elementsInRow(uint32_t rowNr) const;
    const SymConfig symconfig() const { return symcfg; }
    const ScannerConfig scannerConfig() const { return scancfg; }

    // Allow access to raw pointers
    const RowElement<float>* getData() const { return data; }
    const uint64_t* getRowIdx() const { return rowidx; }

    void setData(RowElement<float>* d) { data = d; }

    class RowIterator // a const iterator
            : public std::iterator<std::forward_iterator_tag, RowElement<float> > {
    public:
        explicit RowIterator(RowElement<float>* pelem_) : pelem(pelem_) {}
        RowIterator(const RowIterator& i) : pelem(i.pelem) {}
        RowIterator& operator=(const RowIterator& rhs) {
            pelem = rhs.pelem;
            return (*this);
        }
        bool operator==(const RowIterator& rhs) {
            return (pelem == rhs.pelem);
        }
        bool operator!=(const RowIterator& rhs) {
            return (pelem != rhs.pelem);
        }
        RowIterator& operator++() { // prefix increment
            ++pelem;
            return (*this);
        }
        RowIterator operator++(int) { // postfix increment
            RowIterator tmp(*this);
            ++(*this); // call prefix increment
            return tmp;
        }
        const RowElement<float>& operator*() const {
            return *pelem;
        }
        const RowElement<float>* operator->() const {
            return pelem;
        }

    private:
        RowElement<float>* pelem;
    };


    template <class Value>
    class node_iter : public boost::iterator_facade <
            node_iter<Value>,
            Value,
            //boost::forward_traversal_tag
            //boost::random_access_traversal_tag,
            std::random_access_iterator_tag,
            Value&
            >
    {
    private:
        typedef Value& Ref;
        typedef boost::iterator_facade<node_iter<Value>, Value,
        boost::random_access_traversal_tag, Ref> base_type;

    public:
        typedef node_iter<Value> this_type;
        typedef typename base_type::difference_type difference_type;
        typedef typename base_type::reference reference;

        explicit node_iter(Value* p) : p_elem(p) {}

        const Value* get() const { return p_elem; }

    private:
        friend class boost::iterator_core_access;

        reference dereference() const { return *p_elem; }

        bool equal(node_iter<Value> const& other) const {
            return this->p_elem == other.p_elem;
        }

        void increment() { ++p_elem; }

        void decrement() { --p_elem; }

        void advance(difference_type n) { p_elem += n; }

        difference_type distance_to(const this_type &rhs) const {
            return rhs.p_elem - p_elem;
        }

        Value* p_elem;
    };
    typedef node_iter<RowElement<float> > node_iterator;
    typedef node_iter<const RowElement<float> > node_const_iterator;

    node_iterator beginRow2(uint32_t rowNr) const {
        assert(this->currentMap >=0);
        assert(this->data);
        assert(rowNr >= this->mappings[this->currentMap].mapRowStart);
        assert(rowNr < this->mappings[this->currentMap].mapRowStart +
                this->mappings[this->currentMap].mapRowCount);
        if (rowNr == 0) return node_iterator(data);
        else return node_iterator(&data[rowidx[rowNr - 1] -
                this->mappings[this->currentMap].mapFirstElement]);
    }
    node_iterator endRow2(uint32_t rowNr) const {
        assert(this->currentMap >=0);
        assert(this->data);
        assert(rowNr >= this->mappings[this->currentMap].mapRowStart);
        assert(rowNr < this->mappings[this->currentMap].mapRowStart +
                this->mappings[this->currentMap].mapRowCount);
        return node_iterator(&data[rowidx[rowNr] -
                this->mappings[this->currentMap].mapFirstElement]);
    }

    node_const_iterator beginRowConst2(uint32_t rowNr) const {
        if (rowNr == 0) return node_const_iterator(data);
        else return node_const_iterator(&data[rowidx[rowNr - 1]]);
    }
    node_const_iterator endRowConst2(uint32_t rowNr) const {
        return node_const_iterator(&data[rowidx[rowNr]]);
    }

    RowIterator beginRow(uint32_t rowNr) const;
    RowIterator endRow(uint32_t rowNr) const;

    void mapRow(uint32_t row) const;
    void mapRows(uint32_t start, uint32_t count) const;

private:
    static const uint32_t minHeaderLength = 28;
    static const uint32_t minScanConfigBytes = 60;
    static const uint8_t xsymmask = 1;
    static const uint8_t ysymmask = 2;
    static const uint8_t zsymmask = 4;
    int fileDescriptor;
    std::string filename;
    uint32_t scanConfigBytes;
    off64_t fileSize;
    uint8_t flags;
    uint32_t nRows;
    uint32_t nColumns;
    uint64_t nnz;
    uint64_t* rowidx;
    uint64_t pageSize;
    SymConfig symcfg;
    ScannerConfig scancfg;

    struct tag_mappings{
        uint64_t mapSize, mapFirstElement, off;
        uint32_t mapRowStart, mapRowCount;
        char* map;
#ifdef _HPC_
        char* map2;
#endif
    };
    mutable struct tag_mappings mappings[MAX_MAPPINGS];
    mutable int activeMappings;
    mutable int currentMap ;
    // Current file mapping: can change in const functions
    mutable RowElement<float>* data;

};
