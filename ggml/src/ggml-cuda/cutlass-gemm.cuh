#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

// 定义 GEMM 配置
using ElementInput = cutlass::half_t;  // 输入数据类型 (FP16)
using ElementOutput = cutlass::half_t; // 输出数据类型 (FP16)
using ElementCompute = cutlass::half_t;

using GemmTN = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,    // A: RowMajor
    cutlass::half_t, cutlass::layout::ColumnMajor, // B: ColumnMajor
    float, cutlass::layout::ColumnMajor, // C/D: ColumnMajor
    cutlass::half_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm60,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::Convert<
      float,
      1,
      cutlass::half_t
    >
>;

// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t HgemmNT(
  int M,
  int N,
  int K,
  half const *A,
  int lda,
  half const *B,
  int ldb,
  float *C,
  int ldc,
  cudaStream_t stream
) {

  GemmTN gemm_operator;

  // Cast the pointers to cutlass::half_t
  using TensorRefA = cutlass::TensorRef<const cutlass::half_t, RowMajor>;
  using TensorRefB = cutlass::TensorRef<const cutlass::half_t, ColumnMajor>;
  using TensorRefC = cutlass::TensorRef<float, ColumnMajor>;
  using TensorRefD = cutlass::TensorRef<float, ColumnMajor>;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  TensorRefA tensor_a(reinterpret_cast<cutlass::half_t const *>(A), lda);
  TensorRefB tensor_b(reinterpret_cast<cutlass::half_t const *>(B), ldb);
  TensorRefC tensor_c(C, ldc);
  TensorRefD tensor_d(C, ldc); // Destination matrix D shares the same memory as C

  // Construct the Arguments object
  GemmTN::Arguments args(problem_size,// Gemm Problem dimensions
                         tensor_a,    // Tensor-ref for source matrix A
                         tensor_b,    // Tensor-ref for source matrix B
                         tensor_c,    // Tensor-ref for source matrix C
                         tensor_d    // Tensor-ref for destination matrix D
                );

  // Launch the CUTLASS GEMM kernel.
  cutlass::Status status = gemm_operator(args, nullptr, stream);

  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}
