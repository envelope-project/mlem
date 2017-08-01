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

#include <cstring>
#include <errno.h>
#include <fstream>
#include <stdexcept>

template <typename T, typename S = T>
class Vector {
public:
    typedef T value_type;

    Vector(size_t size = 0, const T& initValue = T()) : buffer(0), nElements(size) {
        if (nElements == 0) return;
        buffer = new T[nElements];
        std::fill(buffer, buffer + nElements, initValue);
    }

    Vector(const std::string& filename) : buffer(0), nElements(0) {
        std::ifstream file(filename.c_str(), std::ifstream::in | std::ifstream::binary);
        if (!file.good())
            throw std::runtime_error(std::string("Cannot open file ") + filename);

        // Get file size
        file.seekg(0, std::ifstream::end);
        std::ifstream::pos_type bytes = file.tellg();
        file.seekg(0, std::ifstream::beg); // according to old C++ standard, eof may not be reset

        if (bytes % sizeof(T) != 0)
            throw std::runtime_error(std::string("Cannot open file ") + filename);

        nElements = bytes / sizeof(T);
        buffer = new T[nElements];

        for (int c=0; !file.eof(); ++c)
            file.read(reinterpret_cast<char*>(&buffer[c]), sizeof(T));

        file.close();
    }

    ~Vector() { delete[] buffer; }

    size_t size() const { return nElements; }

    const T* ptr() const { return buffer; }
    T* ptr() { return buffer; }

    const T& operator[] (int index) const { return buffer[index]; }
    T& operator[] (int index) { return buffer[index]; }

    const T& at(int index) const {
        if (!(0 <= index && index < nElements))
            throw std::out_of_range(std::string("vector out of range"));
        return buffer[index];
    }

    T& at(int index) {
        if (!(0 <= index && index < nElements))
            throw std::out_of_range(std::string("vector out of range"));
        return buffer[index];
    }

    void writeToFile(const std::string& filename) {
        std::ofstream file(filename.c_str(), std::ofstream::out | std::ofstream::binary);
        if (!file.good())
            throw std::runtime_error(std::string("Cannot open file ") + filename +
                                     std::string("; reason: ") + std::string(strerror(errno)));

        for (size_t i=0; i<nElements; ++i)
            file.write(reinterpret_cast<char*>(&buffer[i]), sizeof(T));

        file.close();
    }

    static int GetFd(std::filebuf& filebuf) {
        class exFd : public std::filebuf {
        public:
            int handle() { return _M_file.fd(); }
        };
        return static_cast<exFd&>(filebuf).handle();
    }

    void writeToFileSync(const std::string& filename){
        std::ofstream file(filename.c_str(), std::ofstream::out | std::ofstream::binary);
        if (!file.good())
            throw std::runtime_error(std::string("Cannot open file ") + filename +
                                     std::string("; reason: ") + std::string(strerror(errno)));

        for (size_t i=0; i<nElements; ++i)
            file.write(reinterpret_cast<char*>(&buffer[i]), sizeof(T));

        file.flush();
        fsync(GetFd(*file.rdbuf()));
        file.close();
    }

    void readFromFile(const std::string& filename) {
        size_t nElements;
        std::ifstream file(filename.c_str(), std::ifstream::in | std::ifstream::binary);
        if (!file.good())
            throw std::runtime_error(std::string("Cannot open file ") + filename);

        // Get file size
        file.seekg(0, std::ifstream::end);
        std::ifstream::pos_type bytes = file.tellg();
        file.seekg(0, std::ifstream::beg); // according to old C++ standard, eof may not be reset

        if (bytes % sizeof(T) != 0)
            throw std::runtime_error(std::string("Cannot open file ") + filename);

        nElements = bytes / sizeof(T);
        if(nElements != this->nElements){
            throw std::runtime_error(std::string("Mismatch element numger")
                                     + filename);
        }

        for (int c=0; !file.eof(); ++c)
            file.read(reinterpret_cast<char*>(&buffer[c]), sizeof(T));

        file.close();
    }

protected:
    T* buffer;
    size_t nElements;
};
