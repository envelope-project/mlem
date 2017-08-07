#include "scannerConfig.h"

using namespace CSR;

const Vector3D<uint32_t> ScannerConfig::getCoords(uint32_t voxelNr) const {
  Vector3D<uint32_t> coords;
  coords.z() = voxelNr / (gridDim_.x() * gridDim_.y());
  voxelNr -= coords.z() * (gridDim_.x() * gridDim_.y());
  coords.y() = voxelNr / gridDim_.x();
  voxelNr -= coords.y() * gridDim_.x();
  coords.x() = voxelNr;

  return coords;
}

uint32_t ScannerConfig::getVoxelNr(const Vector3D<uint32_t>& coords) const {
  return coords.x()
          + coords.y() * gridDim_.x()
          + coords.z() * gridDim_.x() * gridDim_.y();
}

uint32_t ScannerConfig::nCrystals() const {
  return blocksize() * nBlocks() * nLayers() * nRings();
}

uint32_t ScannerConfig::nLors() const {
  return nCrystals() * nCrystals();
}

uint32_t ScannerConfig::nVoxels() const {
  return gridDim_.x() * gridDim_.y() * gridDim_.z();
}

uint32_t ScannerConfig::getSize() const {
  uint32_t size = 0;
  size += 40; // Size of the fixed entries in scannerConfig
  size += layers_.size() * sizeof(float) * 5;
  size += blockGaps_.size() * sizeof(float);
  size += ringGaps_.size() * sizeof(float);
  return size;
}
