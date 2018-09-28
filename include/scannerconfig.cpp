/**
    scannerconfig.cpp

    Created on: Jul 26, 2010
            Author: kuestner
*/

#include "vector3d.hpp"
#include "scannerconfig.hpp"


/// Formatted iostream output
std::ostream& operator<<(std::ostream& os, const SymConfig& sc)
{
    os<<"("<<sc.x<<", "<<sc.y<<", "<<sc.z<<")";
    return os;
}

const IVec3 ScannerConfig::getCoords(uint32_t voxelNr) const
{
    IVec3 coords;
    coords.z = voxelNr / (griddim.x * griddim.y);
    voxelNr -= coords.z * (griddim.x * griddim.y);
    coords.y = voxelNr / griddim.x;
    voxelNr -= coords.y * griddim.x;
    coords.x = voxelNr;
    return coords;
}

uint32_t ScannerConfig::getVoxelNr(const IVec3& coords) const
{
    return coords.x
        + coords.y * griddim.x
        + coords.z * griddim.x * griddim.y;
}
