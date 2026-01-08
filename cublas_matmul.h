#ifndef CUBLAS_MATMUL_H
#define CUBLAS_MATMUL_H

#include <cstddef>
#include <vector>
#include <string>

// Needed for cudaStream_t in the public API.
#include <cuda_runtime.h>

namespace cublas_test {

// Matrix initialization modes
enum class InitMode {
    ZEROS,      // All zeros
    ONES,       // All ones
    RANDOM,     // Random values in [0, 1)
    IDENTITY,   // Identity matrix (if square)
    CUSTOM      // Custom single value
};

// Matrix class for managing host and device memory
class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();

    // Disable copy (manage CUDA memory)
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Enable move
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Initialize matrix with different modes
    void init(InitMode mode, float custom_value = 0.0f);
    
    // Copy data between host and device
    void copyToDevice();
    void copyToHost();

    // Accessors
    float* hostData() { return h_data_; }
    const float* hostData() const { return h_data_; }
    float* deviceData() { return d_data_; }
    const float* deviceData() const { return d_data_; }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }

    // Utility
    void print(const std::string& name = "", size_t max_rows = 8, size_t max_cols = 8) const;
    
private:
    size_t rows_;
    size_t cols_;
    float* h_data_;  // Host data
    float* d_data_;  // Device data
};

// cuBLAS Matrix Multiplication wrapper
// Computes: C = alpha * A * B + beta * C
// A is (M x K), B is (K x N), C is (M x N)
class CublasMatMul {
public:
    CublasMatMul();
    ~CublasMatMul();

    // Disable copy
    CublasMatMul(const CublasMatMul&) = delete;
    CublasMatMul& operator=(const CublasMatMul&) = delete;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    void multiply(const Matrix& A, const Matrix& B, Matrix& C,
                  float alpha = 1.0f, float beta = 0.0f);

    // Set a CUDA stream for this cuBLAS handle.
    // If take_ownership is true, the stream will be destroyed in the destructor.
    void setStream(cudaStream_t stream, bool take_ownership = false);

    cudaStream_t getStream() const { return stream_; }

    // Get timing of last operation (in milliseconds)
    float getLastTimeMs() const { return last_time_ms_; }

    // Get GFLOPS of last operation
    float getLastGflops() const { return last_gflops_; }

private:
    void* cublas_handle_;
    cudaStream_t stream_;
    bool owns_stream_;
    float last_time_ms_;
    float last_gflops_;
};

// Utility functions
std::string initModeToString(InitMode mode);
InitMode stringToInitMode(const std::string& str);

// Verify matrix multiplication result (CPU reference)
bool verifyResult(const Matrix& A, const Matrix& B, const Matrix& C,
                  float alpha = 1.0f, float beta = 0.0f, float tolerance = 1e-3f);

} // namespace cublas_test

#endif // CUBLAS_MATMUL_H

