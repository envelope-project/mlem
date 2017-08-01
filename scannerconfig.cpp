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
