# Curriculum

Eight modules, ordered so each one only needs earlier modules plus math from the sibling repos.
Status: ☐ planned · ◐ in progress · ☑ done.

**Prerequisite shorthand** (paths inside the sibling repos):

- **C:** [LearningCalculus](https://github.com/dnlwrthstr/LearningCalculus), `hypersklill/…` or `coursera/…`
- **L:** [LearningLinearAlgebra](https://github.com/dnlwrthstr/LearningLinearAlgebra), `hyperskill/…` or `coursera/…`
- **N:** [LearningNeuralNetworks](https://github.com/dnlwrthstr/LearningNeuralNetworks), `prerequisites/…`
- **Q:** [quantum-computing-course](https://github.com/dnlwrthstr/quantum-computing-course), `notebooks/…`
- **—**: needs only earlier notebooks in this repository.

---

## 00 Foundations

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Why complex numbers exist: $x^2 + 1 = 0$ and the cubic that forced them | C: `hypersklill/00_basics/09_introduction_to_polynomials` | ◐ |
| 02 | Cartesian form and arithmetic: add, subtract, multiply, $i^n$ | — | ☐ |
| 03 | Conjugate, modulus and division | — | ☐ |
| 04 | The complex plane: numbers as points and vectors, the triangle inequality | C: `hypersklill/00_basics/06_cartesian_coordinates`, L: `hyperskill/01_vector_and_vector_spaces/011_vector_operations` | ☐ |

## 01 Polar form and Euler's formula

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Polar form, modulus and argument (principal value, `atan2`) | C: `hypersklill/00_basics/07_trigonometric_functions` | ☐ |
| 02 | Multiplication is rotation and scaling | C: `hypersklill/00_basics/08_trigonometric_identities` | ☐ |
| 03 | Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ from Taylor series | C: `hypersklill/02_derivaties/06_derivative_of_exponential_functions` | ☐ |
| 04 | De Moivre's theorem and trig identities for free | — | ☐ |
| 05 | Roots of unity and $n$-th roots | — | ☐ |

## 02 Polynomials and roots

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Quadratics with negative discriminant; conjugate-root theorem | C: `hypersklill/00_basics/09_introduction_to_polynomials` | ☐ |
| 02 | The fundamental theorem of algebra (with a winding-number picture) | — | ☐ |
| 03 | Finding roots numerically: Newton's method in $\mathbb{C}$ and Newton fractals | C: `coursera/05_newtons_method/00_newtons_method` | ☐ |
| 04 | Partial fractions over $\mathbb{C}$ | C: `hypersklill/04_integrals/06_partial_fraction_expansion` | ☐ |

## 03 Complex functions

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Functions $\mathbb{C} \to \mathbb{C}$ as maps: grids, domain colouring | C: `hypersklill/00_basics/03_functions_in_mathematics` | ☐ |
| 02 | The complex exponential and its periodicity | C: `hypersklill/02_derivaties/06_derivative_of_exponential_functions` | ☐ |
| 03 | The complex logarithm, branches and branch cuts | C: `hypersklill/02_derivaties/05_derivative_of_logarithmic_functions` | ☐ |
| 04 | Complex powers $z^w$ and principal values | — | ☐ |
| 05 | Complex sine, cosine and hyperbolic functions | C: `hypersklill/00_basics/08_trigonometric_identities` | ☐ |
| 06 | Möbius transformations | L: `hyperskill/02_matrices/023_matrix_multiplication` | ☐ |

## 04 Complex linear algebra

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Complex numbers as $2\times 2$ real matrices | L: `coursera/02_vector_and_linear_transformation/06_matrices_as_linear_transformations` | ☐ |
| 02 | Complex eigenvalues of real matrices: rotations and spirals | L: `coursera/04_eigenvalues_and_eigenvectors/02_eigenvalues_and_eigenvectors`, `13_discrete_dynamical_systems` | ☐ |
| 03 | Complex vectors, the Hermitian inner product, norms | L: `hyperskill/01_vector_and_vector_spaces/014_vector_dot_product` | ☐ |
| 04 | Hermitian and unitary matrices | L: `advanced_concepts/04_orthogonal_unitary_transformations` | ☐ |
| 05 | The DFT matrix as a unitary change of basis | L: `hyperskill/11_advanced_concepts/orthogonality_and_fourier_analysis` | ☐ |

## 05 Complex differentiation

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Limits and continuity in $\mathbb{C}$ | C: `hypersklill/01_limits/01_limit_of_a_function` | ☐ |
| 02 | The complex derivative and the Cauchy–Riemann equations | C: `coursera/02_gradients/01_partial_derivatives_part_1` | ☐ |
| 03 | Holomorphic functions; why $\bar z$ and $\lvert z\rvert$ are not | — | ☐ |
| 04 | Conformal maps: angles are preserved | — | ☐ |
| 05 | Harmonic functions and the link to 2-D physics | C: `coursera/05_newtons_method/03_the_hessian` | ☐ |

## 06 Complex integration

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Contour integrals | C: `hypersklill/04_integrals/00_introduction_to_integrals` | ☐ |
| 02 | Cauchy's theorem | — | ☐ |
| 03 | Cauchy's integral formula | — | ☐ |
| 04 | Power series and radius of convergence | C: `hypersklill/01_limits/00_limits_of_sequences` | ☐ |
| 05 | Laurent series and singularities | — | ☐ |
| 06 | The residue theorem | — | ☐ |
| 07 | Real integrals via residues | C: `hypersklill/04_integrals/04_trigonometric_substitution` | ☐ |

## 07 Applications

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | Phasors and AC circuits: impedance as a complex number | 01/03 | ☐ |
| 02 | Linear ODEs: the damped oscillator via $e^{\lambda t}$ | C: `hypersklill/02_derivaties/09_higher_order_derivatives` | ☐ |
| 03 | Fourier series and the DFT/FFT | N: `prerequisites/09_convolution` | ☐ |
| 04 | Signals and filters: the $z$-transform, poles and zeros | 06/05 | ☐ |
| 05 | Fractals: Mandelbrot and Julia sets | 02/03 | ☐ |
| 06 | Qubits: amplitudes, phases and the Bloch sphere | Q: `notebooks/00_complex_linear_algebra`, `01_the_qubit` | ☐ |

---

## Reference courses

The plan follows the standard order of an introductory complex analysis course:

- Tristan Needham, *Visual Complex Analysis* — the geometric viewpoint used throughout.
- Elias Stein & Rami Shakarchi, *Complex Analysis* — modules 05 and 06.
- Petra Bonfert-Taylor, *Introduction to Complex Analysis* (Coursera, Wesleyan) — overall sequencing.
