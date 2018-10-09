/**
    Copyright © 2017 Technische Universitaet Muenchen
    Authors: Tilman Kuestner
           Dai Yang
           Josef Weidendorfer

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
#include <cmath>
#include <ostream>


/// Integer vector in 3d space
class IVec3 {
public:
    IVec3() : x(0), y(0), z(0) {}
    IVec3(const IVec3& rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}
    IVec3(uint32_t x, uint32_t y, uint32_t z) : x(x), y(y), z(z) {}
    uint32_t x, y, z;
};

std::ostream& operator<<(std::ostream& os, const IVec3& v);


/// Float or double vector in 3d space
template <typename T>
class Vector3d
{
public:
    T x, y, z;

    Vector3d() : x(0.0), y(0.0), z(0.0) {}

    Vector3d(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    template <typename S>
    Vector3d<T>(const Vector3d<S>& rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}

    template <typename S>
    Vector3d<T>& operator=(const Vector3d<S>& rhs) {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

    Vector3d& operator+=(const Vector3d& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    const Vector3d operator+(const Vector3d& rhs) const {
        return Vector3d(*this) += rhs;
    }

    const Vector3d operator-() const {
        return Vector3d(-x, -y, -z);
    }

    Vector3d& operator-=(const Vector3d& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    const Vector3d operator-(const Vector3d& rhs) const {
        return Vector3d(*this) -= rhs;
    }

    /// Dot product / inner product
    T operator*=(const Vector3d& rhs) {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    }

    T operator*(const Vector3d& rhs) const {
        return Vector3d(*this) *= rhs;
    }

    /// Right scalar multiplication
    Vector3d& operator*=(const T rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    const Vector3d operator*(const T rhs) const {
        return Vector3d(*this) *= rhs;
    }

    /// Euclidean norm
    T norm() const {
        return std::sqrt(sqdnorm());
    }

    /// Square of Euclidean norm
    T sqdnorm() const {
        return x*x + y*y + z*z;
    }

    /// Make unit vector
    void normalize() {
        T l = sqdnorm();
        if (l != 0.0) {
            l = 1.0 / std::sqrt(l);
            x *= l;
            y *= l;
            z *= l;
        }
    }

    /// Return unit vector
    const Vector3d normalized() const {
        T l = sqdnorm();
        if (l == 0.0)
            return Vector3d();
        else {
            l = 1.0 / std::sqrt(l);
            return Vector3d(x*l, y*l, z*l);
        }
    }
};

/// Cross product
template <typename T>
const Vector3d<T> cross_product(const Vector3d<T>& lhs, const Vector3d<T>& rhs)
{
    return Vector3d<T>(
        lhs.y * rhs.z - lhs.z * rhs.y,
        -(lhs.x * rhs.z - lhs.z * rhs.x),
        lhs.x * rhs.y - lhs.y * rhs.x
    );
}

/// Formatted iostream output
template <typename T>
std::ostream& operator<<(std::ostream& os, const Vector3d<T>& v)
{
    os<<"("<<v.x<<", "<<v.y<<", "<<v.z<<")";
    return os;
}

/// Transform spherical coordinates to cartesian coordinates
template <typename T>
inline
Vector3d<T> sphereToCart(T phi, T theta, T r = 1.0)
{
    T sp = sin(phi);
    T cp = cos(phi);
    T st = sin(theta);
    T ct = cos(theta);
    return Vector3d<T>(r * st * cp, r * st * sp, r * ct);
}
