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
