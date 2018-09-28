/**
 * vector.hpp
 *
 * Created on: Jun 6 2013
 *     Author: kuestner
 *
 * A simple fixed length vector/array
 */

#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <cstring>
#include <errno.h>
#include <fstream>
#include <stdexcept>

#ifdef __clang__
FILE* c_file( std::filebuf& fb );
#endif

template <typename T, typename S = T>
class Vector {
public:
    typedef T value_type;

    Vector(size_t size = 0, const T& initValue = T()) : buffer(0), nElements(size) {
        if (nElements == 0) return;
        buffer = new T[nElements]; // TODO 
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
        buffer = new T[nElements]; // TODO

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

#ifndef __clang__
    static int GetFd(std::filebuf& filebuf) {
        class exFd : public std::filebuf {
        public:
            int handle() { return _M_file.fd(); }
        };
        return static_cast<exFd&>(filebuf).handle();
    }
#else
    static int GetFd(std::filebuf& filebuf){
        return (int)(size_t)c_file(filebuf);
    }
#endif
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




#ifdef __clang__
// Generate a static data member of type Tag::type in which to store
// the address of a private member. It is crucial that Tag does not
// depend on the /value/ of the the stored address in any way so that
// we can access it from ordinary code without directly touching
// private data.
template < class Tag >
struct stowed
{
  static typename Tag::type value;
};

template < class Tag >
typename Tag::type stowed< Tag >::value;

// Generate a static data member whose constructor initializes
// stowed< Tag >::value. This type will only be named in an explicit
// instantiation, where it is legal to pass the address of a private
// member.
template < class Tag, typename Tag::type x >
struct stow_private
{
  stow_private() { stowed< Tag >::value = x; }
  static stow_private instance;
};
template < class Tag, typename Tag::type x >
stow_private< Tag, x > stow_private< Tag, x >::instance;

struct filebuf_file { typedef FILE*( std::filebuf::*type ); };
template struct stow_private< filebuf_file, &std::filebuf::__file_ >;

FILE* c_file( std::filebuf& fb )
{
  return fb.*stowed< filebuf_file >::value;
}
#endif

#endif
