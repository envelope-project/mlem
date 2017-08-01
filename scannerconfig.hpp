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

#include <stdint.h>
#include <vector>

#include "vector3d.hpp"


/// Scanner symmetry configuration
/**
        \p x is true if the symmtry with respect to the yz plane has been used to
        reduce matrix size, i.e. matrix elements have been omitted when saving on
        disk and must be regenerated on-the-fly
*/
class SymConfig {
public:
    SymConfig() : x(false), y(false), z(false) {}
    SymConfig(bool x_, bool y_, bool z_) : x(x_), y(y_), z(z_) {}
    bool x, y, z;
};

std::ostream& operator<<(std::ostream& os, const SymConfig& sc);


/// Describes a layer consisting of the same type of crystals
class LayerConfig
{
public:
    /**
        @param width  The extent of the crystal along the x-axis
        @param height The extent of the crystal along the y-axis
        @param depth  The extent of the crystal alfong the z-axis
        @param depth  The distance of the layer to the scanner's center (z-axis)
        @param mu     The attenuation coefficient of the crystals in this layer
    */
    LayerConfig(float width, float height, float depth, float radius,
                float mu)
        : width(width), height(height), depth(depth), radius(radius), mu(mu) {}
    float width;
    float height;
    float depth;
    float radius;
    float mu;
};


/// Data structure describing the scannner's geometric configuration
class ScannerConfig
{
public:
    ScannerConfig()  {}
    ~ScannerConfig() {}
    // Getter
    uint32_t nLayers()    const { return layers.size(); }
    uint32_t nBlocks()    const { return nblocks; }
    uint32_t blocksize()  const { return blockGaps.size() + 1; }
    uint32_t nRings()     const { return ringGaps.size() + 1; }
    uint32_t nBlockGaps() const { return blockGaps.size(); }
    uint32_t nRingGaps()  const { return ringGaps.size(); }
    const std::vector<LayerConfig>& getLayers() const { return layers; }
    const std::vector<float>& getBlockGaps() const { return blockGaps; }
    const std::vector<float>& getRingGaps() const { return ringGaps; }
    const LayerConfig& getLayer(uint32_t i) const { return layers[i]; }
    float getBlockGap(uint32_t i) const { return blockGaps[i]; }
    float getRingGap(uint32_t i) const { return ringGaps[i]; }
    const IVec3& getGridDim() const { return griddim; }
    const Vector3d<float>& getVoxelSize() const { return voxelsize; }
    // Setter
    void addLayer(const LayerConfig& lc) { layers.push_back(lc); }
    void addBlockGap(float gap) { blockGaps.push_back(gap); }
    void addRingGap(float gap) { ringGaps.push_back(gap); }
    void setNBlocks(uint32_t n) { nblocks = n; }
    void setGridDim(const IVec3& dim) { griddim = dim; }
    void setVoxelSize(const Vector3d<float> size) { voxelsize = size; }
    // Computations
    const IVec3 getCoords(uint32_t voxelNr) const;
    uint32_t getVoxelNr(const IVec3& coords) const;
    uint32_t nCrystals() const {
        return blocksize() * nBlocks() * nLayers() * nRings();
    }
    uint32_t nLors() const {
        return nCrystals() * nCrystals();
    }
    uint32_t nVoxels() const {
        return griddim.x * griddim.y * griddim.z;
    }
private:
    // Detector
    std::vector<LayerConfig> layers;
    std::vector<float> blockGaps;
    std::vector<float> ringGaps;
    uint32_t nblocks;
    // Voxel grid
    IVec3 griddim;
    Vector3d<float> voxelsize;
};
