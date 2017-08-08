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

#ifndef CONFIGLOADER_H
#define CONFIGLOADER_H

#include <cstring>
#include <memory>

#include "scannerConfig.h"

namespace libconfig {
  class Config;
}

namespace CSR {
  class ConfigLoader {
    public:
      ConfigLoader(const std::string& filename);
      ~ConfigLoader();
      const ScannerConfig getConfig();

    private:
      void readGeometry();
      void readGrid();

      std::string filename;
      std::unique_ptr<libconfig::Config> conf;
      ScannerConfig scanConf;
  };
}

#endif
