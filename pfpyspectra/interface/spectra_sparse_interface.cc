/*
 * Copyright 2020 Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <utility>

#include <Spectra/MatOp/SparseGenComplexShiftSolve.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/MatOp/SymShiftInvert.h>

#include <Spectra/GenEigsComplexShiftSolver.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/SymEigsSolver.h>
// #include <Spectra/SymGEigsSolver.h>
#include <Spectra/SymGEigsShiftSolver.h>
// #include <Spectra/DavidsonSymEigsSolver.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// BUG: 稀疏矩阵类型 接口不对
// using ComplexMatrix = Eigen::MatrixXcd;
// using ComplexVector = Eigen::VectorXcd;

// // using Matrix = Eigen::MatrixXd;
// using Matrix = Eigen::SparseMatrixXd;

// // using Vector = Eigen::VectorXd;
// using Vector = Eigen::SparseVectorXd;

// using Eigen::Index;

// NOTE: 返回类型是对的
using ComplexMatrix = Eigen::MatrixXcd;
using ComplexVector = Eigen::VectorXcd;
using SpMatrix = Eigen::SparseMatrix<double>;
using SpVector = Eigen::SparseVector<double>;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

using Eigen::Index;

Spectra::SortRule string_to_sortrule(const std::string &name) {
  std::unordered_map<std::string, Spectra::SortRule> rules = {
      {"LargestMagn", Spectra::SortRule::LargestMagn},
      {"LargestReal", Spectra::SortRule::LargestReal},
      {"LargestImag", Spectra::SortRule::LargestImag},
      {"LargestAlge", Spectra::SortRule::LargestAlge},
      {"SmallestMagn", Spectra::SortRule::SmallestMagn},
      {"SmallestReal", Spectra::SortRule::SmallestReal},
      {"SmallestImag", Spectra::SortRule::SmallestImag},
      {"SmallestAlge", Spectra::SortRule::SmallestAlge},
      {"BothEnds", Spectra::SortRule::BothEnds}};
  auto it = rules.find(name);
  if (it != rules.cend()) {
    return it->second;
  } else {
    std::ostringstream oss;
    oss << "There is no selection rule named: " << name << "\n"
        << "Available selection rules:\n";
    for (const auto &pair : rules) {
      oss << pair.first << "\n";
    }
    throw std::runtime_error(oss.str());
  }
}

/// \brief Run the computation and throw and error if it fails
template <typename ResultVector, typename ResultMatrix, typename Solver>
std::pair<ResultVector, ResultMatrix>
compute_and_check(Solver &eigs, const std::string &selection) {
  // Initialize and compute
  eigs.init();
  // Compute using the user provided selection rule
  eigs.compute(string_to_sortrule(selection));

  // Retrieve results
  if (eigs.info() == Spectra::CompInfo::Successful) {
    // 特征值是向量，特征向量是矩阵
    return std::make_pair(eigs.eigenvalues(), eigs.eigenvectors());
  } else {
    throw std::runtime_error(
        "The Spectra SymEigsSolver calculation has failed!");
  }
}


// TAG: 1 普通矩阵
// FIXME: 传入矩阵为？稀疏矩阵
/// \brief Call the Spectra::GenEigsSolver eigensolver
std::pair<ComplexVector, ComplexMatrix>
geneigssolver(const SpMatrix &mat, Index nvalues, Index nvectors,
              const std::string &selection) {
  using SparseOp = Spectra::SparseGenMatProd<double>;

  // Construct matrix operation object using the wrapper class SparseSymMatProd
  SparseOp op(mat);
  // Spectra::SparseGenMatProd<double> op(mat);
  Spectra::GenEigsSolver<SparseOp> eigs(op, nvalues, nvectors);

  // printf("调用了GenEigsSolver<SparseOp>\n");
  return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

/// \brief Call the Spectra::GenEigsRealShiftSolver eigensolver
std::pair<ComplexVector, ComplexMatrix>
geneigsrealshiftsolver(const SpMatrix &mat, Index nvalues, Index nvectors,
                       double sigma, const std::string &selection) {
  using SparseOp = Spectra::SparseGenRealShiftSolve<double>;
  SparseOp op(mat);
  Spectra::GenEigsRealShiftSolver<SparseOp> eigs(op, nvalues, nvectors, sigma);
  return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

/// \brief Call the Spectra::GenEigsComplexShiftSolver eigensolver
std::pair<ComplexVector, ComplexMatrix>
geneigscomplexshiftsolver(const SpMatrix &mat, Index nvalues, Index nvectors,
                          double sigmar, double sigmai,
                          const std::string &selection) {
  using SparseOp = Spectra::SparseGenComplexShiftSolve<double>;
  SparseOp op(mat);
  Spectra::GenEigsComplexShiftSolver<SparseOp> eigs(op, nvalues, nvectors,
                                                    sigmar, sigmai);
  return compute_and_check<ComplexVector, ComplexMatrix>(eigs, selection);
}

// TAG: 2 对称矩阵
// FIXME: 对称矩阵返回类型怎么不一样
/// \brief Call the Spectra::SparseSymMatProd eigensolver
std::pair<Vector, Matrix> symeigssolver(const SpMatrix &mat, Index nvalues,
                                        Index nvectors,
                                        const std::string &selection) {
  using SparseSym = Spectra::SparseSymMatProd<double>;
  // Construct matrix operation object using the wrapper class
  SparseSym op(mat);
  Spectra::SymEigsSolver<SparseSym> eigs(op, nvalues, nvectors);

  return compute_and_check<Vector, Matrix>(eigs, selection);
}

/// \brief Call the Spectra::SymEigsShiftSolver eigensolver
std::pair<Vector, Matrix> symeigsshiftsolver(const SpMatrix &mat, Index nvalues,
                                             Index nvectors, double sigma,
                                             const std::string &selection) {
  using SparseSymShift = Spectra::SparseSymShiftSolve<double>;
  // Construct matrix operation object using the wrapper class
  SparseSymShift op(mat);
  Spectra::SymEigsShiftSolver<SparseSymShift> eigs(op, nvalues, nvectors,
                                                   sigma);

  return compute_and_check<Vector, Matrix>(eigs, selection);
}

/// \brief Call the Spectra::SymGEigsShiftSolver eigensolver
std::pair<Vector, Matrix> symgeneigsshiftsolver(const SpMatrix &mat_A,
                                                const SpMatrix &mat_B,
                                                Index nvalues, Index nvectors,
                                                double sigma,
                                                const std::string &selection) {
  using SymShiftInvert =
      Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
  using SparseSym = Spectra::SparseSymMatProd<double>;

  // Construct matrix operation object using the wrapper class
  SymShiftInvert op_A(mat_A, mat_B);
  SparseSym op_B(mat_B);
  Spectra::SymGEigsShiftSolver<SymShiftInvert, SparseSym,
                               Spectra::GEigsMode::ShiftInvert>
      eigs(op_A, op_B, nvalues, nvectors, sigma);

  return compute_and_check<Vector, Matrix>(eigs, selection);
}

// NOTE: 稀疏矩阵只支持copy
PYBIND11_MODULE(spectra_sparse_interface, m) {
  m.doc() = "Interface to the C++ spectra library, see: "
            "https://github.com/yixuan/spectra";

  m.def("general_eigensolver", &geneigssolver);

  m.def("general_real_shift_eigensolver", &geneigsrealshiftsolver);

  m.def("general_complex_shift_eigensolver", &geneigscomplexshiftsolver);

  m.def("symmetric_eigensolver", &symeigssolver);

  m.def("symmetric_shift_eigensolver", &symeigsshiftsolver);

  m.def("symmetric_generalized_shift_eigensolver", &symgeneigsshiftsolver);
}
