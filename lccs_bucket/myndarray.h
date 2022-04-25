#pragma once

#include <vector>
#include <array>

template<int D, class Scalar> 
class NDArrayView;

// -----------------------------------------------------------------------------
//  a light-weighted compact n-dimensional array for D >= 3
// 
//  interface
//  operator[]
//  to_ptr()
//  constructor/make_array
//  foreach loop (begin/end)
// ----------------------------------------------------------------------------- 
template<int D, class Scalar> 
class NDArray {
public:
    std::array<size_t, D> lens;
    std::array<size_t, D+1> elemLens;

    NDArray(): lens({0}), buffer(nullptr), bufferRaw(nullptr) {}

    // -------------------------------------------------------------------------
    NDArray(const std::array<size_t, D> &dims):lens(dims), bufferRaw(nullptr) {
        elemLens[D] = 1;
        for (int i = D-1; i >= 0; --i) {
            elemLens[i] = elemLens[i+1] * lens[i];
        }
        buffer = new Scalar[elemLens[0]];
    }

    // -------------------------------------------------------------------------
    ~NDArray() {
        delete[] buffer;
        if (bufferRaw != nullptr) delete[] bufferRaw;
    }

    // -------------------------------------------------------------------------
    void resize(const std::array<size_t, D> &dims) {
        for (int i = 0; i < D; ++i) lens[i] = dims[i];

        elemLens[D] = 1;
        for(int i = D-1; i >= 0; --i) {
            elemLens[i] = elemLens[i+1] * lens[i];
        }

        if (buffer) delete[] buffer;
        buffer = new Scalar[elemLens[0]];
        
        if (bufferRaw != nullptr) {
            delete[] bufferRaw; bufferRaw = nullptr;
        }
    }

    // -------------------------------------------------------------------------
    NDArrayView<D-1, Scalar> operator[](int i) {
        return NDArrayView<D-1, Scalar>(buffer + i*elemLens[1], &elemLens[2]);
    }

    // -------------------------------------------------------------------------
    Scalar* begin() { return buffer; }
    Scalar* end()   { return buffer + elemLens[0]; }

    using PtrType =  typename NDArrayView<D-1, Scalar>::PtrType*;
    using ConstPtrType = typename NDArrayView<D-1, Scalar>::ConstPtrType*;

    // -------------------------------------------------------------------------
    PtrType to_ptr() {
        if (bufferRaw == nullptr) {
            int size = 0;
            int curp = 1;
            for (int i = 0; i < D-1; ++i) {
                size += curp*lens[i];
                curp *= lens[i];
            }
            bufferRaw = new void*[size];

            size = 0;
            curp = 1;
            for (int i = 0; i < D-2; ++i) {
                void **basei = bufferRaw + size;
                size += curp*lens[i];
                curp *= lens[i];

                void **baseip1 = bufferRaw + size;
                for (int j = 0; basei+j < baseip1; ++j) {
                    basei[j] = (void*) (baseip1 + lens[i+1]*j);
                }
            }

            // for second last dimension
            void **basei   = bufferRaw + size;
            void **baseip1 = bufferRaw + size + curp*lens[D-2];
            for (int j = 0; basei+j < baseip1; ++j) {
                basei[j] = (void*) (buffer + lens[D-1]*j);
            }
        }
        return (PtrType)bufferRaw;
    }

    // -------------------------------------------------------------------------
    inline ConstPtrType to_cptr() { return (ConstPtrType) to_ptr(); }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        return uint64_t(elemLens[0]) * uint64_t(sizeof(Scalar)) + 
            uint64_t(sizeof(*this));
    }

protected:
    Scalar *buffer;
    void   **bufferRaw; // will be used only when convert to raw_ptr is called
};

// -----------------------------------------------------------------------------
//  partial specification
// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArray<0, Scalar> {
public:
    using PtrType =  Scalar&;
    using ConstPtrType = const Scalar&;
};

// -----------------------------------------------------------------------------
//  D = 1
// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArray<1, Scalar> {
public:
    static const int D = 1;
    std::array<size_t, D> lens;
    std::array<size_t, D+1> elemLens;

    NDArray(): lens({0}), buffer(nullptr) {}

    // -------------------------------------------------------------------------
    NDArray(const std::array<size_t, D> &dims) : lens(dims) {
        elemLens[D] = 1;
        elemLens[0] = lens[0];
        buffer = new Scalar[elemLens[0]];
    }

    // -------------------------------------------------------------------------
    ~NDArray() { delete[] buffer; }

    // -------------------------------------------------------------------------
    void resize(const std::array<size_t, D> &dims) {
        for(int i=0;i<D;i++){
            lens[i] = dims[i];
        }
        elemLens[D] = 1;
        for(int i=D-1;i>=0;--i){
            elemLens[i] = elemLens[i+1]*lens[i];
        }

        if(buffer){
            delete[] buffer;
        }
        buffer = new Scalar[elemLens[0]];
    }

    // -------------------------------------------------------------------------
    void resize(size_t dim) { resize({dim}); }

    Scalar& operator[](int i) { return buffer[i]; } 

    Scalar* begin() { return buffer; }
    Scalar* end()   { return buffer + elemLens[0]; }

    using PtrType = Scalar*;
    using ConstPtrType = const Scalar*;

    // -------------------------------------------------------------------------
    PtrType to_ptr() { return (PtrType)buffer; }

    // -------------------------------------------------------------------------
    inline ConstPtrType to_cptr() { return (ConstPtrType) to_ptr(); }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        return uint64_t(elemLens[0])*uint64_t(sizeof(Scalar)) + 
            uint64_t(sizeof(*this));
    }

protected:
    Scalar *buffer;
};

// -----------------------------------------------------------------------------
//  D = 2
// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArray<2, Scalar> {
public:
    static const int D = 2;
    std::array<size_t, D> lens;
    std::array<size_t, D+1> elemLens;

    NDArray(): lens({0}), buffer(nullptr), bufferRaw(nullptr) {}

    // -------------------------------------------------------------------------
    NDArray(const std::array<size_t, D> &dims):lens(dims), bufferRaw(nullptr) {
        elemLens[D] = 1;
        for (int i = D-1; i >= 0; --i) {
            elemLens[i] = elemLens[i+1] * lens[i];
        }
        buffer = new Scalar[elemLens[0]];
    }

    // -------------------------------------------------------------------------
    ~NDArray() {
        delete[] buffer;
        if (bufferRaw != nullptr) delete[] bufferRaw;
    }

    // -------------------------------------------------------------------------
    void resize(const std::array<size_t, D> &dims) {
        for (int i = 0; i < D; ++i) lens[i] = dims[i];

        elemLens[D] = 1;
        for (int i = D-1; i >= 0; --i) {
            elemLens[i] = elemLens[i+1] * lens[i];
        }

        if(buffer) delete[] buffer;
        buffer = new Scalar[elemLens[0]];

        if (bufferRaw != nullptr) {
            delete[] bufferRaw; bufferRaw = nullptr;
        }
    }

    // -------------------------------------------------------------------------
    Scalar* operator[](int i) { return (Scalar*)(buffer + i*elemLens[1]); }

    Scalar* begin() { return buffer; }
    Scalar* end()   { return buffer + elemLens[0]; }

    using PtrType = typename NDArray<D-1, Scalar>::PtrType*;
    using ConstPtrType = typename NDArray<D-1, Scalar>::ConstPtrType*;

    // -------------------------------------------------------------------------
    PtrType to_ptr() {
        if (bufferRaw == nullptr) {
            int size = 0;
            int curp = 1;
            for (int i = 0; i < D-1; ++i) {
                size += curp*lens[i];
                curp *= lens[i];
            }
            bufferRaw = new void*[size];

            size = 0;
            curp = 1;
            for (int i = 0; i < D-2; ++i) {
                void **basei = bufferRaw + size;
                size += curp*lens[i];
                curp *= lens[i];

                void **baseip1 = bufferRaw + size;
                for (int j = 0; basei+j < baseip1; ++j) {
                    basei[j] = (void*) (baseip1 + lens[i+1]*j);
                }
            }

            // for second last dimension
            void **basei   = bufferRaw + size;
            void **baseip1 = bufferRaw + size + curp*lens[D-2];
            for (int j = 0; basei+j < baseip1; ++j) {
                basei[j] = (void*) (buffer + lens[D-1]*j);
            }
        }
        return (PtrType)bufferRaw;
    }

    // -------------------------------------------------------------------------
    inline ConstPtrType to_cptr() { return (ConstPtrType) to_ptr(); }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        return uint64_t(elemLens[0]) * uint64_t(sizeof(Scalar)) + 
            uint64_t(sizeof(*this));
    }

protected:
    Scalar *buffer;
    void   **bufferRaw; // will be used only when convert to raw_ptr is called
};

// -----------------------------------------------------------------------------
//  for D >= 3 
// -----------------------------------------------------------------------------
template<int D, class Scalar> 
class NDArrayView {
public:
    NDArrayView(Scalar *p, size_t *elemLensp):p(p), elemLensp(elemLensp) {}

    // -------------------------------------------------------------------------
    NDArrayView<D-1, Scalar> operator[](int i) {
        return NDArrayView<D-1, Scalar>(p + (*elemLensp)*i, elemLensp+1);
    }

    // -------------------------------------------------------------------------
    typedef typename NDArrayView<D-1, Scalar>::PtrType* PtrType;
    typedef typename NDArrayView<D-1, Scalar>::ConstPtrType* ConstPtrType;

    Scalar* begin() { return p; }
    Scalar* end()   { return p + *(elemLensp-1); }

protected:
    Scalar *p;
    size_t *elemLensp;
};

// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArrayView<0, Scalar> {
public:
    typedef Scalar PtrType;
    typedef const Scalar ConstPtrType;
};

// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArrayView<1, Scalar> {
public:
    NDArrayView(Scalar *p, size_t *elemLensp):p(p), elemLensp(elemLensp){}

    // -------------------------------------------------------------------------
    Scalar& operator[](int i) { return p[i]; }
    
    // -------------------------------------------------------------------------
    typedef Scalar* PtrType;
    typedef const Scalar* ConstPtrType;

    // -------------------------------------------------------------------------
    operator PtrType() const { return p; }

    Scalar* begin() { return p; }
    Scalar* end()   { return p + *(elemLensp-1); }

protected:
    Scalar *p;
    size_t *elemLensp;
};

// -----------------------------------------------------------------------------
template<class Scalar> 
class NDArrayView<2, Scalar> {
public:
    static const int D = 2;
    
    NDArrayView(Scalar *p, size_t *elemLensp):p(p), elemLensp(elemLensp){}

    // -------------------------------------------------------------------------
    Scalar* operator[](int i) { return p + (*elemLensp)*i; }

    // -------------------------------------------------------------------------
    typedef typename NDArrayView<D-1, Scalar>::PtrType* PtrType;
    typedef typename NDArrayView<D-1, Scalar>::ConstPtrType* ConstPtrType;

    Scalar* begin() { return p; }
    Scalar* end()   { return p + *(elemLensp-1); }

protected:
    Scalar *p;
    size_t *elemLensp;
};

// -----------------------------------------------------------------------------
inline NDArray<1, double> make_darray(size_t sz0)
{
    return NDArray<1, double>({sz0});
}

// -----------------------------------------------------------------------------
inline NDArray<2, double> make_darray(size_t sz0, size_t sz1)
{
    return NDArray<2, double>({sz0, sz1});
}

// -----------------------------------------------------------------------------
inline NDArray<3, double> make_darray(size_t sz0, size_t sz1, size_t sz2)
{
    return NDArray<3, double>({sz0, sz1, sz2});
}

// -----------------------------------------------------------------------------
inline NDArray<4, double> make_darray(size_t sz0, size_t sz1, size_t sz2, size_t sz3)
{
    return NDArray<4, double>({sz0, sz1, sz2, sz3});
}

// -----------------------------------------------------------------------------
inline NDArray<1, float> make_farray(size_t sz0)
{
    return NDArray<1, float>({sz0});
}

// -----------------------------------------------------------------------------
inline NDArray<2, float> make_farray(size_t sz0, size_t sz1)
{
    return NDArray<2, float>({sz0, sz1});
}

// -----------------------------------------------------------------------------
inline NDArray<3, float> make_farray(size_t sz0, size_t sz1, size_t sz2)
{
    return NDArray<3, float>({sz0, sz1, sz2});
}

// -----------------------------------------------------------------------------
inline NDArray<4, float> make_farray(size_t sz0, size_t sz1, size_t sz2, size_t sz3)
{
    return NDArray<4, float>({sz0, sz1, sz2, sz3});
}

// -----------------------------------------------------------------------------
inline NDArray<1, int> make_iarray(size_t sz0)
{
    return NDArray<1, int>({sz0});
}

// -----------------------------------------------------------------------------
inline NDArray<2, int> make_iarray(size_t sz0, size_t sz1)
{
    return NDArray<2, int>({sz0, sz1});
}

// -----------------------------------------------------------------------------
inline NDArray<3, int> make_iarray(size_t sz0, size_t sz1, size_t sz2)
{
    return NDArray<3, int>({sz0, sz1, sz2});
}

// -----------------------------------------------------------------------------
inline NDArray<4, int> make_iarray(size_t sz0, size_t sz1, size_t sz2, size_t sz3)
{
    return NDArray<4, int>({sz0, sz1, sz2, sz3});
}
