/**
  csr4gen

  Copyright © 2017 Thorsten Fuchs

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
 * @file csr4errors.h
 * @author Thorsten Fuchs
 * @date 4 Jul 2017
*/

#ifndef CSR4ERRORS_H
#define CSR4ERRORS_H

namespace CSR {
  /// Enum of return codes.
  /** Return codes that are used by functions in CSR namespace. */
  typedef enum CsrResult {
    CSR_SUCCESS = 0,        /**< Generic success return value */
    CSR_ERROR = -1,         /**< Generic error return vaulue */
    CSR_IO_OPEN_ERR = -2,   /**< IO open error */
    CSR_IO_CLOSE_ERR = -3,  /**< IO close error */
    CSR_IO_SAVE_ERR = -4,   /**< IO save error */
    CSR_NOT_POP_ERR = -5,   /**< CSR not populated */
    CSR_MEM_ALLOC_ERR = -6, /**< Memory allocation operation failed */
    CSR_MEM_ACC_ERR = -7,   /**< Memory access failed */
    CSR_IO_ERR = -8,        /**< Generic IO error */
    CSR_IO_LOAD_ERR = -9,   /**< IO load error */
  } CsrResult;
}

#endif
