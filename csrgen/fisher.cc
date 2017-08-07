/**
  csr4gen

  Copyright © 2017 Thorsten Fuchs

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
 * @file fisher.cc
 * @author Thorsten Fuchs
 * @date 18 Jun 2017
 * @brief Provide Fisher-Yates Shuffle with additional sorting capability 
 *        of the first n elements
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
