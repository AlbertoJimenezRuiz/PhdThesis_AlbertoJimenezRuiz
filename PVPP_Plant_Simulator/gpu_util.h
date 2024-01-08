#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#include "main.h"

#ifdef __CUDACC__
#if __CUDA_ARCH__ < 600
__device__ inline double custom_atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
__device__ inline float custom_atomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}
#else
__device__ inline FLOATING_TYPE custom_atomicAdd(FLOATING_TYPE *address,
                                                 FLOATING_TYPE val) {
  return atomicAdd(address, val);
}
#endif

#else
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    abortmsg("GPUassert: Error %d: %s. File: %s Line: %d \n", code,
             cudaGetErrorString(code), file, line);
  }
}

template <typename T> class GPU_array {
public:
  GPU_array() {
    m_size = 0;
    m_data = NULL;
  }

  GPU_array(const GPU_array &) = delete;

  GPU_array(long long size) {
    m_size = size * sizeof(T);
    m_size_no_sizeofT = size;

    gpuErrchk(cudaMalloc(&m_data, m_size));
  }

  GPU_array(long long size, const T *data) : GPU_array(size) {
    Copy_HtoD(data);
  }

  GPU_array(std::vector<T> vector_data) : GPU_array(vector_data.size()) {
    Copy_HtoD(vector_data);
  }

  void Copy_HtoD(const T *src) {
    gpuErrchk(cudaMemcpy(m_data, src, m_size, cudaMemcpyHostToDevice));
  }

  void Copy_HtoD(const std::vector<T> &src) {
    gpuErrchk(cudaMemcpy(m_data, &src[0], m_size, cudaMemcpyHostToDevice));
  }

  void Copy_DtoH(T *dst) {
    gpuErrchk(cudaMemcpy(dst, m_data, m_size, cudaMemcpyDeviceToHost));
  }

  void Copy_DtoH(std::vector<T> &dst) {
    gpuErrchk(cudaMemcpy(&dst[0], m_data, m_size, cudaMemcpyDeviceToHost));
  }

  T get_last_DtoH() {
    T temp;
    gpuErrchk(cudaMemcpy(&temp, m_data + m_size_no_sizeofT - 1, sizeof(T),
                         cudaMemcpyDeviceToHost));
    return temp;
  }

  void memset(int value) { gpuErrchk(cudaMemset(m_data, value, m_size)); }

  ~GPU_array() {
    if (m_data)
      gpuErrchk(cudaFree(m_data));
  }

  operator T *() { return m_data; }

  operator const T *() const { return m_data; }

  T *m_data;
  long long m_size;
  long long m_size_no_sizeofT;
};

typedef GPU_array<bool> GPU_array_bool;
typedef GPU_array<int> GPU_array_int;
typedef GPU_array<uint8_t> GPU_array_uint8;
typedef GPU_array<FLOATING_TYPE> GPU_array_f;

void create_csr_boolean_CUDA(vector<vector<int>> matr,
                             GPU_array_int *&d_Start_Matrix_Rows,
                             GPU_array_int *&d_Positions_Column_Raw);

#endif
