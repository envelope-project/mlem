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


#ifndef VECTOR3D_H
#define VECTOR3D_H

namespace CSR {
  template<typename T>
  class Vector3D {
    public:
      Vector3D() : x_{T()}, y_{T()}, z_{T()} {}
      Vector3D(T x, T y, T z) : x_{x}, y_{y}, z_{z} {}
      Vector3D(const Vector3D<T>& rhs) : x_{rhs.x()}, y_{rhs.y()}, z_{rhs.z()} {}

      // Getter&Setter
      T& x() { return x_; }
      T& y() { return y_; }
      T& z() { return z_; }
      // Getter for const
      const T& x() const { return x_; };
      const T& y() const { return y_; };
      const T& z() const { return z_; };

      Vector3D<T>& operator=(const Vector3D<T>& rhs) {
        x_ = rhs.x();
        y_ = rhs.y();
        z_ = rhs.z();
        return *this;
      }

    private:
      T x_, y_, z_;
  };
}

#endif
