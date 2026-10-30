# Curriculum

Eight modules, 42 notebooks, ordered so each one only needs earlier modules plus math from the sibling repos.
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
| 01 | [Why complex numbers exist: $x^2 + 1 = 0$ and the cubic that forced them](./modules/00_foundations/01_why_complex_numbers_exist.ipynb) | C: `hypersklill/00_basics/09_introduction_to_polynomials` | ☑ |
| 02 | [Cartesian form and arithmetic: add, subtract, multiply, $i^n$](./modules/00_foundations/02_cartesian_form_and_arithmetic.ipynb) | — | ☑ |
| 03 | [Conjugate, modulus and division](./modules/00_foundations/03_conjugate_modulus_and_division.ipynb) | — | ☑ |
| 04 | [The complex plane: numbers as points and vectors, the triangle inequality](./modules/00_foundations/04_the_complex_plane.ipynb) | C: `hypersklill/00_basics/06_cartesian_coordinates`, L: `hyperskill/01_vector_and_vector_spaces/011_vector_operations` | ☑ |

## 01 Polar form and Euler's formula

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Polar form, modulus and argument (principal value, `atan2`)](./modules/01_polar_form_and_euler/01_polar_form_modulus_and_argument.ipynb) | C: `hypersklill/00_basics/07_trigonometric_functions` | ☑ |
| 02 | [Multiplication is rotation and scaling](./modules/01_polar_form_and_euler/02_multiplication_is_rotation_and_scaling.ipynb) | C: `hypersklill/00_basics/08_trigonometric_identities` | ☑ |
| 03 | [Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$ from Taylor series](./modules/01_polar_form_and_euler/03_eulers_formula.ipynb) | C: `hypersklill/02_derivaties/06_derivative_of_exponential_functions` | ☑ |
| 04 | [De Moivre's theorem and trig identities for free](./modules/01_polar_form_and_euler/04_de_moivres_theorem.ipynb) | — | ☑ |
| 05 | [Roots of unity and $n$-th roots](./modules/01_polar_form_and_euler/05_roots_of_unity_and_nth_roots.ipynb) | — | ☑ |

## 02 Polynomials and roots

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Quadratics with negative discriminant; conjugate-root theorem](./modules/02_polynomials_and_roots/01_quadratics_and_conjugate_roots.ipynb) | C: `hypersklill/00_basics/09_introduction_to_polynomials` | ☑ |
| 02 | [The fundamental theorem of algebra (with a winding-number picture)](./modules/02_polynomials_and_roots/02_fundamental_theorem_of_algebra.ipynb) | — | ☑ |
| 03 | [Finding roots numerically: Newton's method in $\mathbb{C}$ and Newton fractals](./modules/02_polynomials_and_roots/03_newtons_method_in_the_complex_plane.ipynb) | C: `coursera/05_newtons_method/00_newtons_method` | ☑ |
| 04 | [Partial fractions over $\mathbb{C}$](./modules/02_polynomials_and_roots/04_partial_fractions_over_c.ipynb) | C: `hypersklill/04_integrals/06_partial_fraction_expansion` | ☑ |

## 03 Complex functions

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Functions $\mathbb{C} \to \mathbb{C}$ as maps: grids, domain colouring](./modules/03_complex_functions/01_complex_functions_as_maps.ipynb) | C: `hypersklill/00_basics/03_functions_in_mathematics` | ☑ |
| 02 | [The complex exponential and its periodicity](./modules/03_complex_functions/02_the_complex_exponential.ipynb) | C: `hypersklill/02_derivaties/06_derivative_of_exponential_functions` | ☑ |
| 03 | [The complex logarithm, branches and branch cuts](./modules/03_complex_functions/03_the_complex_logarithm_and_branch_cuts.ipynb) | C: `hypersklill/02_derivaties/05_derivative_of_logarithmic_functions` | ☑ |
| 04 | [Complex powers $z^w$ and principal values](./modules/03_complex_functions/04_complex_powers.ipynb) | — | ☑ |
| 05 | [Complex sine, cosine and hyperbolic functions](./modules/03_complex_functions/05_complex_trig_and_hyperbolic_functions.ipynb) | C: `hypersklill/00_basics/08_trigonometric_identities` | ☑ |
| 06 | [Möbius transformations](./modules/03_complex_functions/06_mobius_transformations.ipynb) | L: `hyperskill/02_matrices/023_matrix_multiplication` | ☑ |

## 04 Complex linear algebra

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Complex numbers as $2\times 2$ real matrices](./modules/04_complex_linear_algebra/01_complex_numbers_as_matrices.ipynb) | L: `coursera/02_vector_and_linear_transformation/06_matrices_as_linear_transformations` | ☑ |
| 02 | [Complex eigenvalues of real matrices: rotations and spirals](./modules/04_complex_linear_algebra/02_complex_eigenvalues_of_real_matrices.ipynb) | L: `coursera/04_eigenvalues_and_eigenvectors/02_eigenvalues_and_eigenvectors`, `13_discrete_dynamical_systems` | ☑ |
| 03 | [Complex vectors, the Hermitian inner product, norms](./modules/04_complex_linear_algebra/03_complex_vectors_and_inner_products.ipynb) | L: `hyperskill/01_vector_and_vector_spaces/014_vector_dot_product` | ☑ |
| 04 | [Hermitian and unitary matrices](./modules/04_complex_linear_algebra/04_hermitian_and_unitary_matrices.ipynb) | L: `advanced_concepts/04_orthogonal_unitary_transformations` | ☑ |
| 05 | [The DFT matrix as a unitary change of basis](./modules/04_complex_linear_algebra/05_the_dft_matrix.ipynb) | L: `hyperskill/11_advanced_concepts/orthogonality_and_fourier_analysis` | ☑ |

## 05 Complex differentiation

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Limits and continuity in $\mathbb{C}$](./modules/05_complex_differentiation/01_limits_and_continuity.ipynb) | C: `hypersklill/01_limits/01_limit_of_a_function` | ☑ |
| 02 | [The complex derivative and the Cauchy–Riemann equations](./modules/05_complex_differentiation/02_the_complex_derivative_and_cauchy_riemann.ipynb) | C: `coursera/02_gradients/01_partial_derivatives_part_1` | ☑ |
| 03 | [Holomorphic functions; why $\bar z$ and $\lvert z\rvert$ are not](./modules/05_complex_differentiation/03_holomorphic_functions.ipynb) | — | ☑ |
| 04 | [Conformal maps: angles are preserved](./modules/05_complex_differentiation/04_conformal_maps.ipynb) | — | ☑ |
| 05 | [Harmonic functions and the link to 2-D physics](./modules/05_complex_differentiation/05_harmonic_functions.ipynb) | C: `coursera/05_newtons_method/03_the_hessian` | ☑ |

## 06 Complex integration

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Contour integrals](./modules/06_complex_integration/01_contour_integrals.ipynb) | C: `hypersklill/04_integrals/00_introduction_to_integrals` | ☑ |
| 02 | [Cauchy's theorem](./modules/06_complex_integration/02_cauchys_theorem.ipynb) | — | ☑ |
| 03 | [Cauchy's integral formula](./modules/06_complex_integration/03_cauchys_integral_formula.ipynb) | — | ☑ |
| 04 | [Power series and radius of convergence](./modules/06_complex_integration/04_power_series_and_radius_of_convergence.ipynb) | C: `hypersklill/01_limits/00_limits_of_sequences` | ☑ |
| 05 | [Laurent series and singularities](./modules/06_complex_integration/05_laurent_series_and_singularities.ipynb) | — | ☑ |
| 06 | [The residue theorem](./modules/06_complex_integration/06_the_residue_theorem.ipynb) | — | ☑ |
| 07 | [Real integrals via residues](./modules/06_complex_integration/07_real_integrals_via_residues.ipynb) | C: `hypersklill/04_integrals/04_trigonometric_substitution` | ☑ |

## 07 Applications

| # | Notebook | Prerequisite | Status |
|---|---|---|---|
| 01 | [Phasors and AC circuits: impedance as a complex number](./modules/07_applications/01_phasors_and_ac_circuits.ipynb) | 01/03 | ☑ |
| 02 | [Linear ODEs: the damped oscillator via $e^{\lambda t}$](./modules/07_applications/02_linear_odes_and_the_damped_oscillator.ipynb) | C: `hypersklill/02_derivaties/09_higher_order_derivatives` | ☑ |
| 03 | [Fourier series and the DFT/FFT](./modules/07_applications/03_fourier_series_and_the_dft.ipynb) | N: `prerequisites/09_convolution` | ☑ |
| 04 | [Signals and filters: the $z$-transform, poles and zeros](./modules/07_applications/04_z_transform_poles_and_zeros.ipynb) | 06/05 | ☑ |
| 05 | [Fractals: Mandelbrot and Julia sets](./modules/07_applications/05_mandelbrot_and_julia_sets.ipynb) | 02/03 | ☑ |
| 06 | [Qubits: amplitudes, phases and the Bloch sphere](./modules/07_applications/06_qubits_amplitudes_and_phases.ipynb) | Q: `notebooks/00_complex_linear_algebra`, `01_the_qubit` | ☑ |

---

## Reference courses

The plan follows the standard order of an introductory complex analysis course:

- Tristan Needham, *Visual Complex Analysis* — the geometric viewpoint used throughout.
- Elias Stein & Rami Shakarchi, *Complex Analysis* — modules 05 and 06.
- Petra Bonfert-Taylor, *Introduction to Complex Analysis* (Coursera, Wesleyan) — overall sequencing.
