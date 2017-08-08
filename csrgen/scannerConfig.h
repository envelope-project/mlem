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


#ifndef SCANNERCONFIG_H
#define SCANNERCONFIG_H

#include <cstdint>
#include <vector>

#include "vector3d.h"

namespace CSR {
  class SymConfig {
    public:
      SymConfig() : x{false}, y{false}, z{false} {}
      SymConfig(bool x, bool y, bool z) : x{x}, y{y}, z{z} {}
      bool x, y, z;
  };

  class LayerConfig {
    public:
      LayerConfig() {};
      LayerConfig(
          float width
          , float height
          , float depth
          , float radius
          , float mu
      ) : width{width}, height{height}, depth{depth},radius{radius}, mu{mu} {}

      float width, height, depth, radius, mu;
  };

  class ScannerConfig {
    public:
      ScannerConfig() {}
      ~ScannerConfig() {}
      // Getter
      uint32_t nLayers()    const { return layers_.size(); }
      uint32_t nBlocks()    const { return nBlocks_; }
      uint32_t blocksize()  const { return blockGaps_.size() + 1; }
      uint32_t nRings()     const { return ringGaps_.size() + 1; }
      uint32_t nBlockGaps() const { return blockGaps_.size(); }
      uint32_t nRingGaps()  const { return ringGaps_.size(); }
      const std::vector<LayerConfig>& getLayers() const { return layers_; }
      const std::vector<float>& getBlockGaps() const { return blockGaps_; }
      const std::vector<float>& getRingGaps() const { return ringGaps_; }
      const LayerConfig& getLayer(uint32_t idx) const { return layers_[idx]; }
      float getBlockGap(uint32_t idx) const { return blockGaps_[idx]; }
      float getRingGap(uint32_t idx) const { return ringGaps_[idx]; }
      const Vector3D<uint32_t>& getGridDim() const { return gridDim_; }
      const Vector3D<float>& getVoxelSize() const { return voxelSize_; }
      // Setter
      void addLayer(const LayerConfig& lc) { layers_.push_back(lc); }
      void addBlockGap(float gap) { blockGaps_.push_back(gap); }
      void addRingGap(float gap) { ringGaps_.push_back(gap); }
      void setNBlocks(uint32_t n) { nBlocks_ = n; }
      // NOTE: Removed passing as const reference
      void setGridDim(Vector3D<uint32_t> dims) { gridDim_ = dims; }
      void setVoxelSize(Vector3D<float> size) { voxelSize_ = size; }
      // Computation
      const Vector3D<uint32_t> getCoords(uint32_t voxelNr) const;
      uint32_t getVoxelNr(const Vector3D<uint32_t>& coords) const;
      uint32_t nCrystals() const;
      uint32_t nLors() const;
      uint32_t nVoxels() const;
      uint32_t getSize() const;

    private:
      // Detector
      std::vector<LayerConfig> layers_;
      std::vector<float> blockGaps_;
      std::vector<float> ringGaps_;
      uint32_t nBlocks_;
      // Voxel grid
      Vector3D<uint32_t> gridDim_;
      Vector3D<float> voxelSize_;
  };
}

#endif
