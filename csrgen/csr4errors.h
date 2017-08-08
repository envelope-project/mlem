/**
    Copyright © 2017 Thorsten Fuchs
    Copyright © 2017 LRR, TU Muenchen
    Authors: Thorsten Fuchs

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
