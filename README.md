# Learning Complex Numbers

A collection of Jupyter notebooks dedicated to learning complex numbers, from $i^2 = -1$ through
Euler's formula and complex functions to contour integration and the residue theorem, with
applications in signals, circuits, fractals and quantum computing. Each notebook follows the same
layered style as the sibling repositories: intuition, formal theory, worked examples, Python code,
visualizations, exercises.

It builds on the sibling repositories:

- [LearningCalculus](../LearningCalculus/): trigonometry, limits, derivatives, Taylor series, integrals.
- [LearningLinearAlgebra](../LearningLinearAlgebra/): vectors, matrices, eigenvalues, unitary transformations.
- [LearningProbability](../LearningProbability/): the notebook format this series copies.

## Project Structure

- [`CURRICULUM.md`](./CURRICULUM.md): the full module plan, with the prerequisite for each notebook and its status.
- `modules/NN_<topic>/`: one numbered folder per module, notebooks numbered inside.
- `GUIDELINES.md`: the notebook specification, shared with the other Learning* repos, plus a complex-numbers section at the end.

| # | Module | What it covers |
|---|---|---|
| 00 | [Foundations](./modules/00_foundations/) | why $i$ exists, Cartesian arithmetic, conjugate and modulus, the complex plane |
| 01 | [Polar form and Euler](./modules/01_polar_form_and_euler/) | modulus and argument, multiplication as rotation, $e^{i\theta}$, De Moivre, roots of unity |
| 02 | [Polynomials and roots](./modules/02_polynomials_and_roots/) | conjugate roots, fundamental theorem of algebra, Newton's method in $\mathbb{C}$ |
| 03 | [Complex functions](./modules/03_complex_functions/) | domain colouring, exp, log and branch cuts, complex powers and trig, Möbius maps |
| 04 | [Complex linear algebra](./modules/04_complex_linear_algebra/) | $2\times 2$ matrix form, complex eigenvalues, Hermitian and unitary matrices, the DFT |
| 05 | [Complex differentiation](./modules/05_complex_differentiation/) | Cauchy–Riemann, holomorphic functions, conformal maps, harmonic functions |
| 06 | [Complex integration](./modules/06_complex_integration/) | contour integrals, Cauchy's theorem and formula, Laurent series, residues |
| 07 | [Applications](./modules/07_applications/) | phasors, ODEs, Fourier, $z$-transform, fractals, qubits |

## Getting Started

You need a Python 3 environment. Install the dependencies:

```bash
pip install -r requirements.txt
```

Notebook outputs are stripped automatically on commit. After cloning, register the git filter once:

```bash
nbstripout --install --attributes .gitattributes
```

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

## License

This project is for educational purposes.
