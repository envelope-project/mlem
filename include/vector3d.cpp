/**
 * vector3d.cpp
 *
 *  Created on: May 26, 2010
 *      Author: kuestner
 */

#include "vector3d.hpp"

/// Formatted iostream output
std::ostream& operator<<(std::ostream& os, const IVec3& v)
{
    os<<"("<<v.x<<", "<<v.y<<", "<<v.z<<")";
    return os;
}
