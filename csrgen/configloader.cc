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

#include <iostream>
#include <libconfig.h++>

#include "configloader.h"

using namespace CSR;

ConfigLoader::ConfigLoader(const std::string& filename)
  : filename{filename},
    conf{nullptr}
{
  using namespace libconfig;

  try {
    conf.reset(new Config());
    conf->readFile(filename.c_str());
    readGeometry();
    readGrid();
  }
  catch (const FileIOException& fioex) {
    std::cerr << "I/O error while reading config file " << filename << std::endl;
    throw fioex;
  }
  catch (const ParseException& pex) {
    std::cerr << "Error parsing config file at "
      << pex.getFile() << ":" << pex.getLine()
      << " - " << pex.getError() << std::endl;
    throw pex;
  }
  catch (const SettingTypeException& e) {
    std::cerr << "Error reading " << e.getPath()
      << " - wrong number format" << std::endl;
    throw e;
  }
  catch (const SettingNotFoundException& e) {
    std::cerr << "Error reading " << e.getPath()
      << " - setting not found" << std::endl;
    throw e;
  }
  catch (const SettingException& e) {
    std::cerr << "Error reading " << e.getPath()
      << " - unspecified exception" << std::endl;
    throw e;
  }
}

ConfigLoader::~ConfigLoader() {

}

const ScannerConfig ConfigLoader::getConfig() {
  return scanConf;
}

void ConfigLoader::readGeometry() {
  using namespace libconfig;

  const Setting& root = conf->getRoot()["detector_geometry"];
  // Read configuration of layers
  const Setting& layers = root["layers"];
  for(int32_t i=0; i<layers.getLength(); ++i) {
    float w = layers[i]["crystal_size"][0];
    float h = layers[i]["crystal_size"][1];
    float d = layers[i]["crystal_size"][2];
    float r = layers[i]["front_radius"];
    float mu = layers[i]["attenuation_coeff"];
    scanConf.addLayer(LayerConfig(w, h, d, r, mu));
  }
  // Read configuration of blocks
  const Setting& blocks = root["blocks"];
  uint32_t nBlocks = blocks["number"];
  scanConf.setNBlocks(nBlocks);
  for(int32_t i=0; i<blocks["gaps"].getLength(); ++i) {
    float gap = blocks["gaps"][i];
    scanConf.addBlockGap(gap);
  }
  // Read configuration of rings
  const Setting& rings = root["rings"];
  for(int32_t i=0; i<rings["gaps"].getLength(); ++i) {
    float gap = rings["gaps"][i];
    scanConf.addRingGap(gap);
  }
}

void ConfigLoader::readGrid() {
  using namespace libconfig;

  const Setting& root = conf->getRoot()["voxel_grid"];
  if(root["voxel_size"].getLength() != 3)
    throw std::runtime_error("Error reading size of voxel");
  if(root["dimension"].getLength() != 3)
    throw std::runtime_error("Error reading dimensions of voxel grid");
  float w = root["voxel_size"][0];
  float h = root["voxel_size"][1];
  float d = root["voxel_size"][2];
  uint32_t nx = root["dimension"][0];
  uint32_t ny = root["dimension"][1];
  uint32_t nz = root["dimension"][2];
  scanConf.setVoxelSize(Vector3D<float>(w, h, d));
  scanConf.setGridDim(Vector3D<uint32_t>(nx, ny, nz));
}
