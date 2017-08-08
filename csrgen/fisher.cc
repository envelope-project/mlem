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

#include <random>
#include <algorithm>
#include <assert.h>

#include "fisher.h"

void fisher_shuffle(
    uint32_t* array,
    uint32_t len,
    uint32_t sortlen
) {
  assert(len >= sortlen);

  static std::mt19937 rng;
  uint32_t j;

  for(uint32_t i=0; i<len; ++i) {
    j = rng() % (i+1);
    array[i] = array[j];
    array[j] = i;
  }

  if(sortlen <= 1)
    return;
  
  std::sort(&array[0],&array[sortlen-1]);
}
