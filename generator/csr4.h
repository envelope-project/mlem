/**
  csr4gen

  Copyright © 2017 Thorsten Fuchs

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
 * @file csr4.h
 * @author Thorsten Fuchs
 * @date 13 Jun 2017
 * @brief Header for CSR4 matrix class
*/

#ifndef CSR4_H
#define CSR4_H

#include <vector>
#include <utility>
#include <stdint.h>
#include <cstring>
#include <iostream>
#include <assert.h>

#include "scannerConfig.h"
#include "csr4errors.h"

namespace CSR {

  /// Header declaration for csr4 matrix
  typedef struct CSR4Header {
    char magic[4];                           /**< Must be set to 'PCSR' for CSR4 */
    uint8_t version;                         /**< Must be set to 4 for CSR4 */
    uint8_t flags;                           /**< Currently unused field */
    uint16_t _padding1;                      /**< Padding for alignment, ignore! */
    uint32_t numRows;                        /**< Number of rows in matrix */
    uint32_t numCols;                        /**< Number of columns in matrix */
    uint64_t nnz;                            /**< Number of non-zero elements in matrix */
    uint32_t scanConfigBytes;                /**< Size of scanner configuration in bytes */
    uint32_t _padding2;                      /**< Padding for alignment, ignore! */
  } CSR4Header;

  /// Compressed sparse matrix class
  /** The CSR4 class stores header, scannerConfig and matrix data */
  template<typename T>
  class CSR4 {
    public:
      CSR4();
      ~CSR4();

      uint32_t rowLength(uint32_t index) const;
      void reset();
      CsrResult resizeRowIndices();
      CsrResult resizeData();

      // Getter
      bool isPopulated() const { return populated_; }
      const std::vector<uint64_t>& getRowIndices() const { return rowIndices_; }
      uint64_t getRowIndex(uint32_t idx) const { return rowIndices_[idx]; }
      uint64_t getRowIndicesByteSize() const {
        return rowIndices_.size() * sizeof(uint64_t);
      }
      const std::vector< std::pair<uint32_t,T> >& getData() const { return data_; }
      const std::pair<uint32_t,T>& getDatum(uint64_t idx) const { return data_[idx]; }
      uint64_t getDataByteSize() const {
        return data_.size() * sizeof(std::pair<uint32_t,T>);
      }
      uint64_t getNNZ() const { return header_.nnz; }
      uint32_t getNumRows() const { return header_.numRows; }
      uint32_t getNumCols() const { return header_.numCols; }
      const CSR4Header& getHeader() const { return header_; }
      uint8_t getHeaderLength() const { return headerLength; }
      const ScannerConfig& getScanConfig() const { return config_; }
      uint32_t getScanConfigBytes() const { return header_.scanConfigBytes; }

      // Setter
      void setRowIndex(uint32_t idx, uint64_t value) {
        rowIndices_[idx] = value;
      }
      void setDatum(uint64_t idx, const std::pair<uint32_t,T>& datum) {
        data_[idx] = datum;
      }
      void setPopulated(bool v) { populated_ = v; }
      void setNNZ(uint32_t v) { header_.nnz = v; }
      void setNumRows(uint32_t v) { header_.numRows = v; }
      void setNumCols(uint32_t v) { header_.numCols = v; }
      void setScanConfig(const ScannerConfig& config) {
        config_ = config;
        header_.scanConfigBytes = config.getSize();
      }
      void setHeader(const CSR4Header& header) { header_ = header; }
      void setRowIndices(std::vector<uint64_t> other) {
        rowIndices_ = std::move(other);
      }
      void setData(std::vector< std::pair<uint32_t,T> > other) {
        data_ = std::move(other);
      }

      static const uint8_t headerLength = 28;
      static const uint32_t minScanBytes = 60;

    private:
      std::vector<uint64_t> rowIndices_;
      std::vector< std::pair<uint32_t,T> > data_;

      CSR4Header header_;
      ScannerConfig config_;

      bool populated_;
  };

  template<typename T>
  CSR4<T>::CSR4()
    : populated_{false}
  {

  }

  template<typename T>
  CSR4<T>::~CSR4<T>() {

  }

  template<typename T>
  uint32_t CSR4<T>::rowLength(uint32_t index) const {
    assert(isPopulated());
    assert(index < header_.numRows);

    if(index == 0)
      return rowIndices_[0];

    return rowIndices_[index] - rowIndices_[index-1];
  }

  template<typename T>
  void CSR4<T>::reset() {
    rowIndices_.clear();
    data_.clear();
    populated_ = false;
    strncpy(header_.magic, "PCSR", 4);
    header_.version = 4;
    header_.numRows = 0;
    header_.numCols = 0;
    header_.scanConfigBytes = 0;
    header_.flags = 0;
  }

  template<typename T>
  CsrResult CSR4<T>::resizeRowIndices() {
    try {
      rowIndices_.resize(header_.numRows);
    } catch (const std::bad_alloc& e) {
      std::cerr <<
        "Failed to allocate rowIndices: " << e.what() << std::endl;
      return CSR_MEM_ALLOC_ERR;
    }
    return CSR_SUCCESS;
  }

  template<typename T>
  CsrResult CSR4<T>::resizeData() {
    try {
      data_.resize(header_.nnz);
    } catch (const std::bad_alloc& e) {
      std::cerr <<
        "Failed to allocate data: " << e.what() << std::endl;
      return CSR_MEM_ALLOC_ERR;
    }
    return CSR_SUCCESS;
  }
}

#endif
