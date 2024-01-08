#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#include "main.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

template <typename T> class GPU_Array_T {
public:
  GPU_Array_T(long long size) {
    m_size = size * sizeof(T);
    gpuErrchk(cudaMalloc(&m_data, m_size));
  }

  GPU_Array_T(long long size, T *data) : GPU_Array_T(size) { copy_HtoD(data); }

  GPU_Array_T(std::vector<T> data) : GPU_Array_T(data.size()) {
    copy_HtoD(data);
  }

  void copy_HtoD(const T *data) {
    gpuErrchk(cudaMemcpy(m_data, data, m_size, cudaMemcpyHostToDevice));
  }

  void copy_HtoD(const std::vector<T> &data) {
    gpuErrchk(cudaMemcpy(m_data, &data[0], m_size, cudaMemcpyHostToDevice));
  }

  void copy_HtoD_range(const T *src, const int start,
                       const int number_elements) {
    gpuErrchk(cudaMemcpy(m_data + start, src, number_elements * sizeof(T),
                         cudaMemcpyHostToDevice));
  }

  void copy_DtoH(T *dst) {
    gpuErrchk(cudaMemcpy(dst, m_data, m_size, cudaMemcpyDeviceToHost));
  }

  void memset(int value) { gpuErrchk(cudaMemset(m_data, 0, m_size)); }

  ~GPU_Array_T() { gpuErrchk(cudaFree(m_data)); }

  operator T *() { return m_data; }

  operator const T *() const { return m_data; }

  T *m_data;
  long long m_size;
};

#ifdef USE_DOUBLE
typedef cuDoubleComplex cuComplexType;
#define MULTIPLY_COMPLEX_CUDA cuCmul
#define DIVIDE_COMPLEX_CUDA cuCdiv
#define CREATE_COMPLEX_CUDA make_cuDoubleComplex
#define ADD_COMPLEX_CUDA cuCadd
#define SUBTRACT_COMPLEX_CUDA cuCsub
#define CUDA_REAL cuCreal
#define CUDA_IMAG cuCimag
#define CONJUGATE_CUDA cuConj
#define ABSOLUTE_CUDA cuCabs
#else
typedef cuFloatComplex cuComplexType;
#define MULTIPLY_COMPLEX_CUDA cuCmulf
#define DIVIDE_COMPLEX_CUDA cuCdivf
#define CREATE_COMPLEX_CUDA make_cuFloatComplex
#define ADD_COMPLEX_CUDA cuCaddf
#define SUBTRACT_COMPLEX_CUDA cuCsubf
#define CUDA_REAL cuCrealf
#define CUDA_IMAG cuCimagf
#define CONJUGATE_CUDA cuConjf
#define ABSOLUTE_CUDA cuCabsf
#endif

typedef GPU_Array_T<int> GPU_Array_int;
typedef GPU_Array_T<FLOATING_TYPE> GPU_Array_float;
typedef GPU_Array_T<cuComplexType> GPU_Array_complex;

inline GPU_Array_complex *
create_GPU_Array_complex(const std::vector<C_FLOATING_TYPE> &src) {
  return new GPU_Array_complex(src.size(), (cuComplexType *)(&src[0]));
}

inline GPU_Array_complex *
create_GPU_Array_complex(const std::vector<C_LONG_DOUBLE> &src_c) {
  std::vector<C_FLOATING_TYPE> src;
  for (auto &d : src_c) {
    src.push_back((C_FLOATING_TYPE)d);
  }
  return create_GPU_Array_complex(src);
}

inline GPU_Array_complex *create_GPU_Array_complex(C_FLOATING_TYPE *src,
                                                   unsigned int size) {
  return new GPU_Array_complex(size, (cuComplexType *)(src));
}

#endif
