#!/usr/bin/env python3
"""Concept sheets for modules 04 (complex linear algebra) and 05 (complex differentiation).

One page per notebook that summarizes its central idea. Every number is computed here from the
same matrices, vectors, functions, points and random seeds the notebook uses, and is checked
against the notebook's printed outputs with asserts.
Images are written to modules/04_complex_linear_algebra/img/ and modules/05_complex_differentiation/img/.

Run from anywhere:  python scripts/make_modules_04_05_concept_images.py
"""
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.lines import Line2D

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID, SEQ,
                       new_figure, grid, panel, heat, trapezoid)
from viz_style import save as _save

MODULE_04 = REPO / "modules" / "04_complex_linear_algebra"
MODULE_05 = REPO / "modules" / "05_complex_differentiation"
save_04 = partial(_save, folder=MODULE_04 / "img")
save_05 = partial(_save, folder=MODULE_05 / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
PURPLE = "#8a5cc2"
# diverging map for signed values: blue (negative) - surface - orange (positive)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", SURFACE, "#f2a07c", "#c4481a"])


# ================================================================ helpers
def arrow(ax, start, end, color, width=2.2, zorder=4, label=None, style="-|>"):
    """Arrow between two points given as complex numbers."""
    start, end = complex(start), complex(end)
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag), zorder=zorder,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=width, mutation_scale=15,
                                shrinkA=0, shrinkB=0))
    if label is not None:
        ax.plot([], [], color=color, linewidth=width, label=label)


def argand(ax, xlim, ylim, title, xlabel="Re", ylabel="Im", grid_on=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.grid(grid_on)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def boxed_legend(ax, **kw):
    kw.setdefault("fontsize", 9)
    return ax.legend(framealpha=0.92, frameon=True, edgecolor="none", facecolor=SURFACE, **kw)


def unit_circle(n=300):
    return np.exp(1j * np.linspace(0, 2 * np.pi, n))


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=0)


def cfmt(z, digits=4):
    """Format a complex number like '1.9757+0.5735i'."""
    z = complex(z)
    re, im = round(z.real, digits) + 0.0, round(z.imag, digits) + 0.0
    if im == 0:
        return f"{re:g}".replace("-", "−")
    if re == 0:
        return {1: "i", -1: "−i"}.get(im, f"{im:g}i".replace("-", "−"))
    return f"{re:g}{'+' if im > 0 else '−'}{abs(im):g}i".replace("-", "−")


# ================================================================ module 04, notebook 01
def to_matrix(z):
    z = complex(z)
    return np.array([[z.real, -z.imag], [z.imag, z.real]]) + 0.0


def sheet_04_01_complex_numbers_as_matrices():
    J = to_matrix(1j)
    assert to_matrix(3 + 4j).tolist() == [[3, -4], [4, 3]]
    assert (J @ J).tolist() == [[-1, 0], [0, -1]]
    assert (to_matrix(3 + 4j) @ np.array([1.0, 2.0])).tolist() == [-5.0, 10.0]
    assert np.allclose(to_matrix(1 + 2j) @ to_matrix(3 - 1j), to_matrix(5 + 5j))
    assert np.isclose(np.linalg.det(to_matrix(3 + 4j)), 25)
    assert np.allclose(np.linalg.matrix_power(to_matrix(1 + 1j), 8), 16 * np.eye(2))

    # 4.2: the dictionary on rng(0) numbers
    rng = np.random.default_rng(0)
    z = complex(*rng.normal(size=2))
    w = complex(*rng.normal(size=2))
    assert str(z) == "(0.1257302210933933-0.1321048632913019j)"
    assert str(w) == "(0.6404226504432821+0.10490011715303971j)"
    checks = [np.allclose(to_matrix(z + w), to_matrix(z) + to_matrix(w)),
              np.allclose(to_matrix(z * w), to_matrix(z) @ to_matrix(w)),
              np.allclose(to_matrix(z) @ to_matrix(w), to_matrix(w) @ to_matrix(z)),
              np.allclose(to_matrix(np.conj(z)), to_matrix(z).T),
              np.isclose(np.linalg.det(to_matrix(z)), abs(z) ** 2),
              np.allclose(to_matrix(1 / z), np.linalg.inv(to_matrix(z))),
              np.allclose(to_matrix(z) @ to_matrix(z).T, abs(z) ** 2 * np.eye(2))]
    assert all(checks) and len(checks) == 7

    # 4.6: which matrices commute with J
    J_int = np.array([[0, -1], [1, 0]])
    candidates = {"M(2−3i)  [[2,3],[−3,2]]": np.array([[2, 3], [-3, 2]]),
                  "symmetric [[2,3],[3,2]]": np.array([[2, 3], [3, 2]]),
                  "reflection diag(1,−1)": np.array([[1, 0], [0, -1]]),
                  "shear [[1,1],[0,1]]": np.array([[1, 1], [0, 1]]),
                  "stretch diag(2,1)": np.array([[2, 0], [0, 1]])}
    commutes = {name: np.array_equal(m @ J_int, J_int @ m) for name, m in candidates.items()}
    assert list(commutes.values()) == [True, False, False, False, False]

    fig = new_figure("Multiplying by a + bi is the matrix [[a, −b], [b, a]]: a rotation-scaling",
                     r"$M(a+bi) = aI + bJ$,   $J = M(i)$ is the 90° turn with $J^2 = -I$;   "
                     r"$M(zw) = M(z)M(w)$,  $\det M(z) = |z|^2$,  $M(re^{i\varphi}) = r\,R(\varphi)$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 house under w = 1 + i
    ax = fig.add_subplot(gs[0, 0])
    house = np.array([0, 1, 1 + 1j, 0.5 + 1.5j, 1j, 0])
    w0 = 1 + 1j
    image_complex = w0 * house
    image_matrix = to_matrix(w0) @ np.vstack([house.real, house.imag])
    assert np.allclose(image_matrix[0] + 1j * image_matrix[1], image_complex)
    ax.fill(house.real, house.imag, color=BLUE_LIGHT, alpha=0.6, zorder=2)
    ax.plot(house.real, house.imag, "o-", color=BLUE, markersize=4, label="house $z$", zorder=3)
    ax.plot(image_complex.real, image_complex.imag, "-", color=ORANGE, linewidth=6, alpha=0.45,
            label=r"$wz$ (complex multiply)", zorder=3)
    ax.plot(image_matrix[0], image_matrix[1], "--", color=INK, linewidth=1.4,
            label=r"$M(w)\,(x, y)^T$", zorder=4)
    arrow(ax, 0, w0, ORANGE, width=2)
    ax.text(1.06, 0.82, r"$w = 1+i$", color=ORANGE, fontsize=11)
    argand(ax, (-2.5, 2.0), (-0.5, 3.0), r"Same image twice: $\times\,(1+i)$ = rotate 45°, scale $\sqrt{2}$")
    boxed_legend(ax, loc="upper left")

    # 5.2 powers of J
    ax = fig.add_subplot(gs[0, 1])
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    labels = ["$I$", "$J$", "$J^2 = -I$", "$J^3 = -J$"]
    colors = [BLUE, ORANGE, AQUA, PURPLE]
    power = np.eye(2)
    for label, color in zip(labels, colors):
        image = power @ square
        corner = power @ np.array([1.0, 0.0])
        ax.fill(image[0], image[1], color=color, alpha=0.28, label=label, zorder=2)
        ax.plot(image[0], image[1], color=color, linewidth=1.2, zorder=2)
        ax.plot(*corner, "o", color=color, markersize=8, markeredgecolor=INK, zorder=4)
        power = J @ power
    assert np.allclose(power, np.eye(2))     # J^4 = I
    argand(ax, (-1.5, 1.5), (-1.5, 1.95), r"Powers of $J = M(i)$: quarter turns of the unit square")
    boxed_legend(ax, loc="upper center", ncol=4, fontsize=8.6)
    ax.set_yticks(np.arange(-1, 1.01, 0.5))
    ax.text(1.0, -1.35, "dot = image of 1", fontsize=9, color=INK2, ha="center")

    panel(fig.add_subplot(gs[0, 2]), "The dictionary", [
        r"$a + bi \;\leftrightarrow\; aI + bJ$:   $1 \leftrightarrow I$,   $i \leftrightarrow J$",
        r"$z + w,\; zw \;\leftrightarrow\; M(z) + M(w),\; M(z)M(w)$",
        r"$\bar z \leftrightarrow M(z)^T$,     $|z|^2 \leftrightarrow \det M(z)$",
        r"$1/z \leftrightarrow M(z)^{-1} = M(z)^T / \det M(z)$",
        "",
        "M(3+4i) = [[3, −4], [4, 3]], det = 25 = |3+4i|²",
        "`M(3+4i) @ (1, 2) -> (-5, 10)`",
        "   = (3+4i)(1+2i) = −5 + 10i",
        "`M(1+2i) @ M(3-i) -> [[5,-5],[5,5]]`",
        "   = M(5+5i) = (1+2i)(3−i)",
        "All 7 rows hold for random z, w (seed 0).",
    ], spacing=0.083)

    # 5.3 grids: complex number vs shear
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.35)
    grid_values = np.linspace(0, 1, 5)
    for k, (title, matrix) in enumerate([("$M(1.2+0.5i)$:\nsquares stay square", to_matrix(1.2 + 0.5j)),
                                         ("shear [[1, 1], [0, 1]]:\nnot a complex number",
                                          np.array([[1.0, 1.0], [0.0, 1.0]]))]):
        ax = fig.add_subplot(sub[0, k])
        for value in grid_values:
            for line in (np.array([[value, value], [0, 1]]), np.array([[0, 1], [value, value]])):
                ax.plot(*line, color=GRID, linewidth=1, zorder=1)
                image = matrix @ line
                ax.plot(*image, color=BLUE, linewidth=1.5, zorder=2)
        ax.plot(*(matrix @ np.array([1, 0])), "o", color=ORANGE, zorder=3)
        argand(ax, (-1, 2.2), (-0.6, 2.2), "")
        ax.set_title(title, fontsize=11.5)
    assert np.isclose(np.linalg.det(to_matrix(1.2 + 0.5j)), abs(1.2 + 0.5j) ** 2)

    # 5.4 powers of a rotation-scaling matrix
    ax = fig.add_subplot(gs[1, 1])
    for w, color in [(0.95 * np.exp(0.4j), BLUE), (np.exp(0.4j), ORANGE)]:
        points = [np.array([1.0, 0.0])]
        for _ in range(40):
            points.append(to_matrix(w) @ points[-1])
        points = np.array(points)
        powers = w ** np.arange(41)
        assert np.allclose(points[:, 0] + 1j * points[:, 1], powers)
        ax.plot(points[:, 0], points[:, 1], "o-", markersize=3.5, linewidth=1, color=color,
                label=rf"$|w| = {abs(w):.2f}$, $\arg w = 0.4$")
    argand(ax, (-1.25, 1.25), (-1.25, 1.7), r"Iterating $M(w)$ walks through the points $w^k$")
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4)
    ax.set_yticks(np.arange(-1, 1.01, 0.5))

    panel(fig.add_subplot(gs[1, 2]), "Which real matrices are complex?", [
        "[[p, q], [r, s]] = M(z) ⟺ p = s, r = −q ⟺ AJ = JA",
        "",
        "`commutes with J?`",
        *[f"`  {name:24s} {'yes' if ok else 'no'}`" for name, ok in commutes.items()],
        "",
        r"Polar form: $M(re^{i\varphi}) = r\,R(\varphi)$,  $e^{\varphi J} = R(\varphi)$",
        r"$M(1+i) = \sqrt{2}\,R(\pi/4)$, so $M(1+i)^8 = 16\,I$",
    ], spacing=0.081, fontsize=12)
    save_04(fig, name="01_complex_numbers_as_matrices.png")


# ================================================================ module 04, notebook 02
def sheet_04_02_complex_eigenvalues_of_real_matrices():
    A = np.array([[0.5, -0.6], [0.75, 1.1]])
    eigenvalues = np.linalg.eigvals(A)
    assert np.allclose(sorted(eigenvalues, key=lambda v: -v.imag), [0.8 + 0.6j, 0.8 - 0.6j])
    assert np.isclose(np.trace(A), 1.6) and np.isclose(np.linalg.det(A), 1.0)

    # 4.3 real canonical form exactly as the notebook builds it
    values, vectors = np.linalg.eig(A)
    index = int(np.argmin(values.imag))
    lam = values[index]
    v = vectors[:, index]
    P_numpy = np.column_stack([v.real, v.imag])
    assert np.allclose(P_numpy.round(6) + 0.0, [[0.298142, 0.596285], [-0.745356, 0.0]])
    C = np.array([[lam.real, lam.imag], [-lam.imag, lam.real]])
    assert np.allclose(C, [[0.8, -0.6], [0.6, 0.8]])
    phi = np.angle(np.conj(lam))
    assert f"{phi:.6f}" == "0.643501" and f"{np.degrees(phi):.4f}" == "36.8699"
    P = np.array([[2.0, 0.0], [-1.0, 2.0]])
    P_inv = np.linalg.inv(P)
    assert np.allclose(A @ P, P @ C) and np.allclose(A @ P, [[1.6, -1.2], [0.4, 2.2]])

    # 4.5 canonical sizes change by exactly |lambda|
    x0 = np.array([2.0, 0.0])
    printed = {0.9: [2.0, 1.622, 1.971, 2.083, 1.705], 1.0: [2.0, 1.803, 2.433, 2.857, 2.599],
               1.1: [2.0, 1.983, 2.944, 3.803, 3.805]}
    for s, plain in printed.items():
        x, sizes, plains = x0.copy(), [np.linalg.norm(P_inv @ x0)], [2.0]
        for _ in range(10):
            x = (s * A) @ x
            sizes.append(np.linalg.norm(P_inv @ x))
            plains.append(np.linalg.norm(x))
        ratios = np.array(sizes[1:]) / np.array(sizes[:-1])
        assert np.allclose(ratios, s) and np.allclose(np.round(plains[:5], 3), plain)

    # 4.2 random real 5x5 (seed 3): conjugate pairs
    rng = np.random.default_rng(3)
    eig5 = np.linalg.eigvals(rng.normal(size=(5, 5)))
    assert [f"{v:.4f}" for v in eig5] == ["-3.5437+0.0000j", "1.9757+0.5735j", "1.9757-0.5735j",
                                          "0.4065+0.0000j", "-0.2997+0.0000j"]
    # 4.6 hidden invariant plane
    B = np.array([[1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    S = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    A3 = S @ B @ np.linalg.inv(S)
    eig3 = np.linalg.eigvals(A3)
    assert np.allclose(sorted(eig3, key=lambda z: (z.real, z.imag)), [0.5, 1 - 1j, 1 + 1j])

    fig = new_figure("Complex eigenvalues of a real matrix are a hidden rotation",
                     r"$\lambda = \frac{\mathrm{tr}A}{2} \pm i\sqrt{\det A - \frac{(\mathrm{tr}A)^2}{4}}$,  "
                     r"$|\lambda|^2 = \det A$;   $A = PCP^{-1}$, $P = [\mathrm{Re}\,v,\ \mathrm{Im}\,v]$, "
                     r"$C = M(a+bi)$:  $\arg\lambda$ = turn per step, $|\lambda|$ = growth")
    gs = grid(fig, top=0.84, wspace=0.3)

    angles = np.linspace(0, 2 * np.pi, 300)
    radius = np.linalg.norm(P_inv @ x0)
    ellipse = P @ (radius * np.vstack([np.cos(angles), np.sin(angles)]))

    # 5.1 spiral in, ellipse, spiral out
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ellipse[0], ellipse[1], "--", color=MUTED, linewidth=1.2, label="invariant ellipse of $A$")
    for s, steps, color in [(0.9, 30, AQUA), (1.0, 10, INK2), (1.1, 12, ORANGE)]:
        points = [x0]
        for _ in range(steps):
            points.append((s * A) @ points[-1])
        points = np.array(points)
        style = dict(linestyle="none") if s == 1.0 else dict(linewidth=1.1)
        ax.plot(points[:, 0], points[:, 1], "o", markersize=3.8, color=color,
                label=rf"$x_{{k+1}} = {s}A\,x_k$,  $|\lambda| = {s}$", **style)
        if s != 1.0:
            ax.plot(points[:, 0], points[:, 1], "-", linewidth=1, color=color)
    arrow(ax, 0, P[0, 0] + 1j * P[1, 0], BLUE, width=2.4)
    arrow(ax, 0, P[0, 1] + 1j * P[1, 1], PURPLE, width=2.4)
    ax.text(-6.6, 6.2, r"arrows: Re $v$ = (2, −1)", color=BLUE, fontsize=10, va="top")
    ax.text(-6.6, 5.2, r"            Im $v$ = (0, 2)", color=PURPLE, fontsize=10, va="top")
    ax.plot(*x0, "o", color=INK, markersize=7, zorder=5)
    argand(ax, (-7, 7), (-7, 7), "Same start, three scalings of A", xlabel="$x_1$", ylabel="$x_2$")
    boxed_legend(ax, loc="lower left", fontsize=8.6)

    # 5.2 canonical coordinates
    ax = fig.add_subplot(gs[0, 1])
    w = 0.8 + 0.6j
    for s, color in [(0.9, AQUA), (1.0, INK2), (1.1, ORANGE)]:
        x = x0.copy()
        canonical = [P_inv @ x]
        for _ in range(14):
            x = (s * A) @ x
            canonical.append(P_inv @ x)
        canonical = np.array(canonical)
        complex_orbit = (s * w) ** np.arange(15) * complex(*canonical[0])
        assert np.allclose(canonical[:, 0] + 1j * canonical[:, 1], complex_orbit)
        ax.plot(canonical[:, 0], canonical[:, 1], "o-", color=color, markersize=3.8, linewidth=1,
                label=f"$P^{{-1}}x_k$, s = {s}")
        ax.plot(complex_orbit.real, complex_orbit.imag, "x", color=INK, markersize=5)
    ax.plot([], [], "x", color=INK, label=r"$(s\,w)^k u_0$, $w = 0.8+0.6i$")
    argand(ax, (-4.6, 4.6), (-4.6, 4.6), r"In the basis Re $v$, Im $v$: multiply by $0.8+0.6i$")
    boxed_legend(ax, loc="lower left", fontsize=8.6)

    panel(fig.add_subplot(gs[0, 2]), "Example: A = [[0.5, −0.6], [0.75, 1.1]]", [
        r"tr $A$ = 1.6,  det $A$ = 1,  $1.6^2 = 2.56 < 4$",
        r"$\lambda = 0.8 \pm 0.6i$,   $|\lambda| = 1$,   $\varphi$ = 0.6435 rad = 36.87°",
        r"$\lambda = 0.8 - 0.6i$:  $v = (2,\ -1+2i)$",
        r"$P$ = [[2, 0], [−1, 2]],  $C$ = [[0.8, −0.6], [0.6, 0.8]]",
        r"$AP = PC$ = [[1.6, −1.2], [0.4, 2.2]]  ✓",
        "",
        r"Each step: $\|P^{-1}x_k\|$ × exactly $s$ (0.9, 1, 1.1);",
        r"plain $\|x_k\|$ wobbles: 2, 1.803, 2.433, 2.857 (s = 1)",
        "One turn ≈ 360 / 36.87 ≈ 9.8 steps.",
    ], spacing=0.087, fontsize=12)

    # 5.4 eigenvalues in the complex plane
    ax = fig.add_subplot(gs[1, 0])
    circle = unit_circle()
    ax.plot(circle.real, circle.imag, color=MUTED, linewidth=1.2, label="unit circle")
    for s, color in [(0.9, AQUA), (1.0, INK2), (1.1, ORANGE)]:
        ev = np.linalg.eigvals(s * A)
        for value in ev:
            arrow(ax, 0, value, color, width=1.6)
        ax.plot(ev.real, ev.imag, "o", color=color, markersize=7, label=f"eigenvalues of {s}A")
    arc = 0.3 * np.exp(1j * np.linspace(0, np.angle(0.8 + 0.6j), 30))
    ax.plot(arc.real, arc.imag, color=INK2, linewidth=1.2)
    ax.annotate("36.87°", (0.34, 0.07), fontsize=10, color=INK2)
    argand(ax, (-1.3, 1.3), (-1.3, 1.3), r"Conjugate pairs $0.8s \pm 0.6s\,i$")
    boxed_legend(ax, loc="lower left", fontsize=8.6)

    # 5.3 trace-determinant plane
    ax = fig.add_subplot(gs[1, 1])
    trace = np.linspace(-3, 3, 400)
    ax.fill_between(trace, trace ** 2 / 4, 1, where=trace ** 2 / 4 < 1, color=AQUA_LIGHT, alpha=0.8,
                    label=r"complex, $|\lambda| < 1$: spiral in", zorder=1)
    ax.fill_between(trace, np.maximum(trace ** 2 / 4, 1), 2.5, color=ORANGE_LIGHT, alpha=0.7,
                    label=r"complex, $|\lambda| > 1$: spiral out", zorder=1)
    ax.plot(trace, trace ** 2 / 4, color=INK, linewidth=1.4, label=r"$\det = \mathrm{tr}^2/4$", zorder=2)
    ax.plot([-2, 2], [1, 1], color=BLUE, linewidth=3, label=r"$|\lambda| = 1$: closed ellipses", zorder=2)
    for s in (0.9, 1.0, 1.1):
        ax.plot(1.6 * s, s ** 2, "o", color=INK, zorder=3)
        ax.annotate(f"{s}A", (1.6 * s, s ** 2), textcoords="offset points", xytext=(-8, 0), fontsize=9.5,
                    ha="right", va="center")
    ax.text(2.55, 0.3, "real\neigenvalues", fontsize=9, color=INK2, ha="center")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True)
    ax.set_xlabel(r"$\mathrm{tr}\,A$")
    ax.set_ylabel(r"$\det A$")
    ax.set_title("Trace–determinant plane: where spirals live")
    boxed_legend(ax, loc="upper left", fontsize=8.6)

    panel(fig.add_subplot(gs[1, 2]), "Pairs and invariant planes", [
        r"$Av = \lambda v$ with $A$ real  ⟹  $A\bar v = \bar\lambda\,\bar v$",
        "Random real 5×5 (seed 3):",
        "`  -3.5437, 1.9757±0.5735i, 0.4065, -0.2997`",
        "",
        r"In $\mathbb{R}^n$: span{Re $v$, Im $v$} is an invariant plane.",
        "B = [[1,−1,0],[1,1,0],[0,0,0.5]] hidden as SBS⁻¹:",
        r"eigenvalues $1 \pm i$ and 0.5; the plane comes back",
        "with action [[1, −1], [1, 1]] (residual ~6e−16).",
        "",
        r"$|\lambda| < 1$ in,  $= 1$ on an ellipse,  $> 1$ out",
    ], spacing=0.083, fontsize=12)
    save_04(fig, name="02_complex_eigenvalues_of_real_matrices.png")


# ================================================================ module 04, notebook 03
def gram_schmidt(vectors, correct_order=True):
    orthonormal = []
    for v in vectors:
        w = v.astype(complex)
        for e in orthonormal:
            coefficient = np.vdot(e, v) if correct_order else np.vdot(v, e)
            w = w - coefficient * e
        orthonormal.append(w / np.linalg.norm(w))
    return np.column_stack(orthonormal)


def sheet_04_03_complex_vectors_and_inner_products():
    u = np.array([1 + 1j, 2 - 1j])
    v = np.array([3, 1j])
    assert np.vdot(u, v) == 2 - 1j and np.vdot(v, u) == 2 + 1j and u @ v == 4 + 5j
    assert np.vdot(u, u).real == 7 and np.vdot(v, v).real == 10
    zz = np.array([1, 1j])
    assert zz @ zz == 0 and np.vdot(zz, zz) == 2

    rng = np.random.default_rng(1)
    u1 = rng.normal(size=4) + 1j * rng.normal(size=4)
    v1 = rng.normal(size=4) + 1j * rng.normal(size=4)
    assert np.round(np.vdot(u1, v1), 6) == -0.468002 - 2.091862j
    assert np.round(np.vdot(u1, 1j * u1), 6) == 4.246829j

    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(10_000):
        a = rng.normal(size=5) + 1j * rng.normal(size=5)
        b = rng.normal(size=5) + 1j * rng.normal(size=5)
        worst = max(worst, abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert round(worst, 4) == 0.9285

    v1g, v2g, v3g = np.array([1, 1j, 0]), np.array([1, 1, 1j]), np.array([0, 1j, 1 + 1j])
    E = gram_schmidt([v1g, v2g, v3g])
    assert np.allclose(E[:, 1] * np.sqrt(2), [0.5 + 0.5j, 0.5 - 0.5j, 1j])
    assert np.allclose(E.conj().T @ E, np.eye(3))
    E_wrong = gram_schmidt([v1g, v2g, v3g], correct_order=False)
    gram_wrong = E_wrong.conj().T @ E_wrong
    assert np.allclose(np.round(gram_wrong, 4), [[1, -0.7071j, 0.1147 + 0.1147j],
                                                 [0.7071j, 1, 0], [0.1147 - 0.1147j, 0, 1]])

    fig = new_figure("Measure complex vectors with ⟨u, v⟩ = Σ ūⱼvⱼ: conjugate one slot",
                     r"$\langle v, v\rangle = \sum |v_j|^2 \geq 0$,   $\langle v, u\rangle = \overline{\langle u, v\rangle}$,   "
                     r"$|\langle u, v\rangle| \leq \|u\|\,\|v\|$;   Re$\langle u, v\rangle$ = real dot product in $\mathbb{R}^{2n}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 inner product in C^1
    ax = fig.add_subplot(gs[0, 0])
    w = 1.5 * np.exp(0.4j)
    z = 1.2 * np.exp(1.5j)
    inner = np.conj(w) * z
    assert f"{abs(inner):.4f}" == "1.8000" and f"{np.angle(inner):.4f}" == "1.1000"
    for number, label, color in [(w, "$w$", BLUE), (z, "$z$", ORANGE),
                                 (inner, r"$\langle w, z\rangle = \bar{w}z$", INK)]:
        arrow(ax, 0, number, color, width=2.2)
        ax.text(number.real * 1.06 + 0.03, number.imag * 1.06 + 0.04, label, color=color, fontsize=12)
    arc = np.linspace(np.angle(w), np.angle(z), 50)
    ax.plot(0.55 * np.cos(arc), 0.55 * np.sin(arc), color=MUTED, linewidth=1.5)
    arc = np.linspace(0, np.angle(inner), 50)
    ax.plot(0.4 * np.cos(arc), 0.4 * np.sin(arc), color=INK, linestyle=":", linewidth=1.5)
    ax.plot([inner.real, inner.real], [0, inner.imag], "--", color=INK2, linewidth=1)
    ax.text(inner.real + 0.04, -0.2, "Re = dot product\nof the arrows", color=INK2, fontsize=9, va="top")
    argand(ax, (-0.8, 2.2), (-0.65, 2.2), r"$\bar w z$: modulus $|w||z|$ = 1.8, arg = 1.5 − 0.4")

    # 5.2 why conjugate
    ax = fig.add_subplot(gs[0, 1])
    phi = np.linspace(0, np.pi, 200)
    plain = 1 + np.exp(2j * phi)
    sc = ax.scatter(plain.real, plain.imag, c=phi, cmap=SEQ, s=11, zorder=3)
    ax.plot([], [], "o", color=BLUE, label=r"$z \cdot z = 1 + e^{2i\varphi}$, $0 \leq \varphi \leq \pi$")
    ax.plot(2, 0, "o", color=ORANGE, markersize=13, zorder=4, label=r"$\langle z, z\rangle = 2$ for every $\varphi$")
    ax.plot(0, 0, "x", color=INK, markersize=11, markeredgewidth=2, zorder=4)
    ax.annotate(r"$z \cdot z = 0$ at $\varphi = \pi/2$", (0, 0), textcoords="offset points",
                xytext=(8, -18), fontsize=9.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(r"$\varphi$")
    cbar.outline.set_visible(False)
    argand(ax, (-0.5, 2.5), (-1.5, 1.5), r"$z = (1, e^{i\varphi})$: plain square vs norm")
    boxed_legend(ax, loc="upper left", fontsize=8.8)

    panel(fig.add_subplot(gs[0, 2]), "Example: u = (1+i, 2−i), v = (3, i)", [
        r"$\langle u, v\rangle = (1-i)\cdot 3 + (2+i)\cdot i = 2 - i$",
        r"$\langle v, u\rangle = 2 + i$  (the conjugate)",
        r"$\|u\|^2 = 2 + 5 = 7$,   $\|v\|^2 = 9 + 1 = 10$",
        r"Cauchy–Schwarz: $\sqrt{5} \approx 2.24 \leq \sqrt{70} \approx 8.37$",
        "",
        "`np.vdot(u, v) -> (2-1j)    # conjugates u`",
        "`u @ v         -> (4+5j)    # no conjugate`",
        "`z = (1, i):  z @ z -> 0j,  vdot -> (2+0j)`",
        "",
        r"$\langle u, iu\rangle = i\|u\|^2$: $u \perp iu$ in $\mathbb{R}^{2n}$ but",
        "not orthogonal in ℂⁿ (same complex line).",
    ], spacing=0.078, fontsize=12)

    # 5.3 Cauchy-Schwarz: the unit disk
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(5)
    normalised = []
    for _ in range(3000):
        a = rng.normal(size=3) + 1j * rng.normal(size=3)
        b = rng.normal(size=3) + 1j * rng.normal(size=3)
        normalised.append(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    normalised = np.array(normalised)
    a = rng.normal(size=3) + 1j * rng.normal(size=3)
    scalars = np.exp(1j * np.linspace(0, 2 * np.pi, 12, endpoint=False)) * 1.7
    dependent = np.array([np.vdot(a, c * a) / (np.linalg.norm(a) * np.linalg.norm(c * a)) for c in scalars])
    assert round(np.abs(normalised).max(), 4) == 0.9943 and np.allclose(np.abs(dependent), 1)
    circle = unit_circle()
    ax.scatter(normalised.real, normalised.imag, s=4, alpha=0.45, color=BLUE, label="3000 random pairs in ℂ³",
               zorder=2)
    ax.plot(dependent.real, dependent.imag, "o", color=ORANGE, markersize=7, zorder=3,
            label=r"$v = c\,u$ (dependent)")
    ax.plot(circle.real, circle.imag, color=INK, linewidth=1.2)
    argand(ax, (-1.25, 1.25), (-1.25, 1.55), r"$\langle u, v\rangle / (\|u\|\,\|v\|)$ stays in the unit disk")
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, markerscale=2)
    ax.set_yticks(np.arange(-1, 1.01, 0.5))
    ax.text(0, -1.17, f"max modulus of random pairs: {np.abs(normalised).max():.4f}", ha="center",
            fontsize=9, color=INK2)

    # 5.4 orthonormal pair: products close a triangle
    ax = fig.add_subplot(gs[1, 1])
    omega = np.exp(2j * np.pi / 3)
    e1 = np.array([1, 1, 1]) / np.sqrt(3)
    e2 = np.array([1, omega, omega ** 2]) / np.sqrt(3)
    products = np.conj(e1) * e2
    assert abs(np.vdot(e1, e2)) < 1e-12 and np.isclose(np.linalg.norm(e2), 1)
    position = 0
    centroid = np.mean(np.cumsum(products))
    for j, product in enumerate(products):
        arrow(ax, position, position + product, [BLUE, ORANGE, AQUA][j], width=2.6)
        mid = position + product / 2
        out = (mid - centroid) / abs(mid - centroid)
        ax.text((mid + 0.09 * out).real, (mid + 0.09 * out).imag,
                rf"$\bar e_{{1,{j + 1}}}\,e_{{2,{j + 1}}}$", fontsize=12, color=[BLUE, ORANGE, AQUA][j],
                ha="center", va="center")
        position = position + product
    ax.plot(0, 0, "o", color=INK, markersize=7, zorder=5)
    ax.text(-0.02, -0.06, "start = end", fontsize=9.5, color=INK2)
    argand(ax, (-0.25, 0.6), (-0.2, 0.45), r"$e_1 = (1,1,1)/\sqrt{3} \perp e_2 = (1,\omega,\omega^2)/\sqrt{3}$")
    ax.text(0.175, 0.41, r"$\langle e_1, e_2\rangle$ = sum of the three products = 0", fontsize=10,
            color=INK2, ha="center")

    # 4.5 Gram-Schmidt with the wrong coefficient order
    ax = fig.add_subplot(gs[1, 2])
    gram_labels = [[cfmt(value, 2) if abs(value) > 1e-9 else "0" for value in row]
                   for row in np.round(gram_wrong, 4)]
    heat(ax, np.abs(gram_wrong), ["e₁", "e₂", "e₃"], ["e₁", "e₂", "e₃"],
         "Gram–Schmidt with ⟨v, e⟩ instead of ⟨e, v⟩:\nGram matrix Eᴴ E ≠ I (right order gives I)",
         "", "", labels=gram_labels, vmax=1.0,
         highlight_cells=[(0, 1), (1, 0), (0, 2), (2, 0)], fontsize=11)
    ax.set_aspect("equal")
    save_04(fig, name="03_complex_vectors_and_inner_products.png")


# ================================================================ module 04, notebook 04
def sheet_04_04_hermitian_and_unitary_matrices():
    A = np.array([[2, 1 - 1j], [1 + 1j, 3]])
    eig_A, U_A = np.linalg.eigh(A)
    assert np.allclose(eig_A, [1, 4]) and np.allclose(U_A @ np.diag(eig_A) @ U_A.conj().T, A)
    v_hand, w_hand = np.array([-1 + 1j, 1]), np.array([1 - 1j, 2])
    assert np.allclose(A @ v_hand, v_hand) and np.allclose(A @ w_hand, 4 * w_hand) and np.vdot(v_hand, w_hand) == 0
    x_ex = np.array([1, 1j])
    assert np.vdot(x_ex, A @ x_ex) == 7

    rng = np.random.default_rng(1)
    B = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H = (B + B.conj().T) / 2
    eig_H, _ = np.linalg.eigh(H)
    assert np.allclose(eig_H.round(6), [-2.011979, -0.960211, 1.63983, 2.751587])
    x = rng.normal(size=4) + 1j * rng.normal(size=4)
    assert f"{np.vdot(x, H @ x) / np.vdot(x, x):.4f}" == "-0.1473+0.0000j"

    not_hermitian = np.array([[1.0, 5.0], [-5.0, 2.0]])
    assert np.allclose(sorted(np.linalg.eigvals(not_hermitian).round(4), key=lambda z: z.imag),
                       [1.5 - 4.9749j, 1.5 + 4.9749j])
    assert np.allclose(np.linalg.eigh(not_hermitian)[0].round(4), [-3.5249, 6.5249])

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    ev_s, U_s = np.linalg.eigh(np.pi / 4 * sigma_x)
    U_ex4 = U_s @ np.diag(np.exp(1j * ev_s)) @ U_s.conj().T
    assert np.allclose(U_ex4 * np.sqrt(2), [[1, 1j], [1j, 1]])

    fig = new_figure("Hermitian matrices are the real numbers, unitary ones the unit circle",
                     r"$A^H = A$ ⟹ real $\lambda$, orthogonal eigenvectors, $A = U\Lambda U^H$;   "
                     r"$U^HU = I$ ⟹ $\|Ux\| = \|x\|$, $|\lambda| = 1$;   $e^{iH}$ is unitary")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 where the eigenvalues live
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(4)
    herm, unit, general = [], [], []
    for _ in range(40):
        Bm = (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))) / np.sqrt(6)
        herm.extend(np.linalg.eigvalsh((Bm + Bm.conj().T) / 2))
        Q, _ = np.linalg.qr(Bm)
        unit.extend(np.linalg.eigvals(Q))
        general.extend(np.linalg.eigvals(Bm))
    herm, unit, general = np.array(herm), np.array(unit), np.array(general)
    assert np.abs(np.abs(unit) - 1).max() < 1e-13
    circle = unit_circle()
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=1.5)
    ax.scatter(general.real, general.imag, s=8, color=MUTED, alpha=0.45, label="general complex $B$", zorder=2)
    ax.scatter(unit.real, unit.imag, s=11, color=ORANGE, label=r"unitary $Q$: $|\lambda| = 1$", zorder=3)
    ax.scatter(herm.real, np.zeros_like(herm), s=60, color=BLUE, marker="|",
               label=r"Hermitian $(B + B^H)/2$: $\lambda$ real", zorder=4)
    argand(ax, (-2.2, 2.2), (-2.2, 2.2), "40 random 6×6 matrices of each kind")
    boxed_legend(ax, loc="upper left", fontsize=8.6)

    # 5.2 quadratic form
    ax = fig.add_subplot(gs[0, 1])
    A_tri = np.array([[1, 2 + 1j], [0, 4]])
    rng = np.random.default_rng(6)
    xs = rng.normal(size=(2, 3000)) + 1j * rng.normal(size=(2, 3000))
    xs = xs / np.linalg.norm(xs, axis=0)
    form_h = np.sum(np.conj(xs) * (A @ xs), axis=0)
    form_t = np.sum(np.conj(xs) * (A_tri @ xs), axis=0)
    assert np.abs(form_h.imag).max() < 1e-14
    assert form_h.real.min().round(3) == 1.0 and form_h.real.max().round(3) == 4.0
    ax.scatter(form_t.real, form_t.imag, s=3, alpha=0.3, color=MUTED, label="non-Hermitian [[1, 2+i], [0, 4]]")
    ax.scatter(form_h.real, form_h.imag, s=4, color=BLUE, label="Hermitian [[2, 1−i], [1+i, 3]]")
    ax.plot([1, 4], [0, 0], "o", color=ORANGE, markersize=8, label="eigenvalues 1 and 4 (both)")
    argand(ax, (-0.3, 5.3), (-1.9, 1.9), r"$\langle x, Ax\rangle$ for 3000 unit vectors $x \in \mathbb{C}^2$")
    boxed_legend(ax, loc="upper left", fontsize=8.6, markerscale=2.5)

    panel(fig.add_subplot(gs[0, 2]), "Example: A = [[2, 1−i], [1+i, 3]]", [
        r"$\det(A - \lambda I) = \lambda^2 - 5\lambda + 4 = (\lambda - 1)(\lambda - 4)$",
        r"$\lambda = 1$: $v = (-1+i,\ 1)$,   $\lambda = 4$: $w = (1-i,\ 2)$",
        r"$\langle v, w\rangle = (-1-i)(1-i) + 2 = 0$",
        r"$A = U\,\mathrm{diag}(1, 4)\,U^H$,  $U = [v/\sqrt{3},\ w/\sqrt{6}]$",
        "",
        r"$x = (1, i)$: $\langle x, Ax\rangle = 7$, real;  Rayleigh 7/2 ∈ [1, 4]",
        "",
        "eigh trap: it mirrors the lower triangle.",
        "`[[1, 5], [-5, 2]]: eig  -> 1.5 ± 4.9749i`",
        "`                   eigh -> -3.5249, 6.5249`",
    ], spacing=0.084, fontsize=12)

    # 5.3 unitary time evolution
    Hm = np.array([[1.0, 0.5 - 0.5j, 0.0], [0.5 + 0.5j, 0.0, 0.8j], [0.0, -0.8j, -1.0]])
    ev, U = np.linalg.eigh(Hm)
    x0 = np.array([1, 0, 0], dtype=complex)
    times = np.linspace(0, 8, 400)
    states = np.array([U @ (np.exp(-1j * t * ev) * (U.conj().T @ x0)) for t in times])
    probabilities = np.abs(states) ** 2
    assert np.abs(probabilities.sum(axis=1) - 1).max() < 1e-14

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=1.5)
    short = times[times <= 1.5]
    for value, color in zip(ev, [BLUE, ORANGE, AQUA]):
        path = np.exp(-1j * short * value)
        ax.plot(path.real, path.imag, color=color, linewidth=2.6, label=rf"$e^{{-it\lambda}}$, $\lambda = {value:.2f}$")
        ax.plot(path[-1].real, path[-1].imag, "o", color=color, markersize=7)
    argand(ax, (-1.3, 1.3), (-1.3, 1.3), r"Eigenvalues of $e^{-itH}$ run on the circle ($t \leq 1.5$)")
    boxed_legend(ax, loc="center", fontsize=8.6)

    ax = fig.add_subplot(gs[1, 1])
    for j, color in enumerate([BLUE, ORANGE, AQUA]):
        ax.plot(times, probabilities[:, j], color=color, linewidth=1.8, label=f"$|x_{j + 1}(t)|^2$")
    ax.plot(times, probabilities.sum(axis=1), "--", color=INK, linewidth=1.6, label=r"sum $= \|x(t)\|^2$")
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlim(0, 8)
    ax.grid(True)
    ax.set_xlabel("t")
    ax.set_ylabel("squared modulus")
    ax.set_title(r"$x(t) = e^{-itH}x_0$: components slosh, norm stays 1")
    boxed_legend(ax, loc="upper center", ncol=4, fontsize=8.4, columnspacing=1.0)

    panel(fig.add_subplot(gs[1, 2]), "Numbers ↔ matrices", [
        r"conjugate $\bar z$   ↔   conjugate transpose $A^H$",
        r"real, $\bar z = z$   ↔   Hermitian, $A^H = A$",
        r"unit circle, $\bar z z = 1$   ↔   unitary, $U^HU = I$",
        r"$e^{i\theta}$, $\theta$ real   ↔   $e^{iH}$, $H$ Hermitian",
        "",
        r"$U$ = [[1, i], [i, 1]]/$\sqrt{2}$ $= e^{i(\pi/4)\sigma_x}$:",
        r"eigenvalues $e^{\pm i\pi/4}$ on eigenvectors (1, 1), (1, −1)",
        "",
        r"Proof of real $\lambda$: $\lambda\langle v,v\rangle = \langle v, Av\rangle$",
        r"$= \langle Av, v\rangle = \bar\lambda\langle v, v\rangle$",
    ], spacing=0.083, fontsize=12)
    save_04(fig, name="04_hermitian_and_unitary_matrices.png")


# ================================================================ module 04, notebook 05
def dft_matrix(n):
    indices = np.arange(n)
    return np.exp(-2j * np.pi / n) ** np.outer(indices, indices) / np.sqrt(n)


def sheet_04_05_the_dft_matrix():
    F4 = dft_matrix(4)
    assert np.allclose(2 * F4, [[1, 1, 1, 1], [1, -1j, -1, 1j], [1, -1, 1, -1], [1, 1j, -1, -1j]])
    for n in (2, 4, 7, 64):
        F = dft_matrix(n)
        assert np.allclose(F.conj().T @ F, np.eye(n)) and np.allclose(F, F.T)
    x4 = np.array([1.0, 2.0, 3.0, 4.0])
    X4 = F4 @ x4
    assert np.allclose(X4, [5, -1 + 1j, -1, -1 - 1j]) and np.allclose(np.fft.fft(x4), 2 * X4)
    assert np.isclose(np.sum(np.abs(X4) ** 2), 30)
    F6 = dft_matrix(6)
    assert np.allclose((F6 @ F6 @ np.arange(6.0)).real, [0, 5, 4, 3, 2, 1])
    printed_counts = {1: [1, 0, 0, 0], 2: [1, 1, 0, 0], 3: [1, 1, 1, 0], 4: [2, 1, 1, 0], 5: [2, 1, 1, 1],
                      6: [2, 2, 1, 1], 7: [2, 2, 2, 1], 8: [3, 2, 2, 1], 9: [3, 2, 2, 2], 10: [3, 3, 2, 2],
                      11: [3, 3, 3, 2], 12: [4, 3, 3, 2]}
    for n, counts in printed_counts.items():
        eig = np.linalg.eigvals(dft_matrix(n))
        assert [int(np.sum(np.abs(eig - t) < 1e-8)) for t in (1, -1, -1j, 1j)] == counts

    fig = new_figure("The DFT is a unitary change of basis built from roots of unity",
                     r"$F_{jk} = \omega^{jk}/\sqrt{n}$, $\omega = e^{-2\pi i/n}$;   $F^HF = I$ because "
                     r"$\Sigma_j\, q^j = 0$ for $q^n = 1 \neq q$;   $X = Fx$, $x = F^HX$, $\|X\| = \|x\|$, $F^4 = I$")
    gs = grid(fig, top=0.84, wspace=0.32)

    # 5.1 columns walk through roots of unity
    ax = fig.add_subplot(gs[0, 0])
    n = 8
    j = np.arange(n)
    circle = unit_circle()
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=1.5)
    for k, color, size in [(1, BLUE, 9), (3, ORANGE, 5)]:
        entries = np.exp(-2j * np.pi * j * k / n)
        loop = np.r_[entries, entries[:1]]
        ax.plot(loop.real, loop.imag, "-", color=color, alpha=0.55, linewidth=1.3)
        ax.plot(entries.real, entries.imag, "o", color=color, markersize=size, zorder=3,
                label=f"column k = {k}: step −{360 * k // n}°")
    for index, entry in enumerate(np.exp(-2j * np.pi * j / n)):
        ax.text(1.22 * entry.real, 1.22 * entry.imag, f"j={index}", ha="center", va="center", fontsize=8.5,
                color=BLUE)
    argand(ax, (-1.55, 1.55), (-1.45, 2.05), r"Entries $\omega^{jk}$ of $\sqrt{8}\,F$, $j = 0..7$")
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.2, columnspacing=0.8, handletextpad=0.3)
    ax.set_yticks(np.arange(-1, 1.01, 0.5))

    # 5.2 the matrix as a picture
    ax = fig.add_subplot(gs[0, 1])
    n = 32
    idx = np.arange(n)
    scaled_F = np.exp(-2j * np.pi * np.outer(idx, idx) / n)
    image = ax.imshow(scaled_F.real, cmap=DIVERGING, vmin=-1, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    ax.set_title(r"Re$(\sqrt{n}F) = \cos(2\pi jk/n)$, n = 32")
    ax.set_xlabel("column k (frequency)")
    ax.set_ylabel("row j (time)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 5.3 aliasing of f_7
    ax = fig.add_subplot(gs[0, 2])
    n = 8
    t_fine = np.linspace(0, 1, 400)
    t_samples = np.arange(n) / n
    samples = np.exp(2j * np.pi * 7 * t_samples)
    assert np.allclose(samples, np.exp(-2j * np.pi * t_samples))
    ax.plot(t_fine, np.cos(2 * np.pi * 7 * t_fine), color=MUTED, linewidth=1, label="cos(2π·7t): 7 cycles")
    ax.plot(t_fine, np.cos(-2 * np.pi * t_fine), "--", color=ORANGE, linewidth=2, label="cos(2π·(−1)t)")
    ax.vlines(t_samples, 0, samples.real, color=BLUE, linewidth=2)
    ax.plot(t_samples, samples.real, "o", color=BLUE, markersize=7, label=r"Re of $\sqrt{8}\,f_7$ samples")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(-1.25, 1.6)
    ax.grid(True)
    ax.set_xlabel("t = j / n")
    ax.set_title(r"Aliasing: frequency 7 = frequency $-1$ (n = 8)")
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4)

    # 5.4 a signal and its Fourier coordinates
    ax = fig.add_subplot(gs[1, 0])
    n = 64
    jj = np.arange(n)
    x = 0.5 + 2 * np.cos(2 * np.pi * 5 * jj / n) + np.sin(2 * np.pi * 12 * jj / n)
    X = np.fft.fft(x, norm="ortho")
    large = np.flatnonzero(np.abs(X) > 1e-9)
    assert large.tolist() == [0, 5, 12, 52, 59]
    assert np.allclose(np.round(X[large], 4), [4, 8, -4j, 4j, 8])
    energy_x, energy_X = np.sum(x ** 2), np.sum(np.abs(X) ** 2)
    assert np.isclose(energy_x, 176) and np.isclose(energy_X, 176)
    ax.vlines(np.arange(n), 0, np.abs(X), color=BLUE, linewidth=1.6)
    ax.plot(large, np.abs(X[large]), "o", color=ORANGE, markersize=7, zorder=3)
    for k in large:
        ax.annotate(f"k={k}\n{cfmt(np.round(X[k], 4))}", (k, abs(X[k])), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5)
    ax.set_ylim(0, 10.8)
    ax.set_xlim(-2, 66)
    ax.grid(axis="y")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$|X_k|$")
    ax.set_title(r"$x_j = 0.5 + 2\cos(2\pi 5j/64) + \sin(2\pi 12j/64)$: $|X_k|$")
    ax.text(32, 6.0, f"Parseval: ‖x‖² = {energy_x:.0f} = ‖X‖²", ha="center", fontsize=10, color=INK2)

    # 5.5 eigenvalues of F
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=1.5)
    n = 16
    eig = np.linalg.eigvals(dft_matrix(n))
    counts16 = []
    for target, color in zip((1, -1, 1j, -1j), (BLUE, ORANGE, AQUA, PURPLE)):
        matches = eig[np.abs(eig - target) < 1e-8]
        counts16.append(len(matches))
        radii = 1 + 0.08 * np.arange(len(matches))
        ax.plot(radii * np.real(target), radii * np.imag(target), "o", color=color, markersize=7)
        ax.annotate(f"{len(matches)}×", (1.62 * np.real(target), 1.62 * np.imag(target)), ha="center",
                    va="center", fontsize=12, color=color, weight="bold")
    assert sum(counts16) == 16 and counts16 == [5, 4, 3, 4]
    argand(ax, (-1.95, 1.95), (-1.95, 1.95), r"Eigenvalues of $F$ (n = 16) are only $\pm 1, \pm i$")
    ax.text(0, -0.25, r"$F^4 = I$ ⟹ $\lambda^4 = 1$", ha="center", fontsize=10.5, color=INK2)
    ax.text(0, -0.5, "(stacked outward by multiplicity)", ha="center", fontsize=8.5, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "Example: x = (1, 2, 3, 4), n = 4", [
        r"$2F_4$ rows: (1, 1, 1, 1), (1, −i, −1, i),",
        "                 (1, −1, 1, −1), (1, i, −1, −i)",
        r"$X = F_4x = (5,\ -1+i,\ -1,\ -1-i)$,   $X_3 = \overline{X_1}$",
        "`np.fft.fft(x) -> [10, -2+2j, -2, -2-2j] = 2X`",
        r"Parseval: $1+4+9+16 = 30 = 25+2+1+2$",
        "",
        r"$X_k = \langle f_k, x\rangle$: coordinates in the Fourier basis",
        r"$F^2$ reverses: $(0,1,2,3,4,5) \mapsto (0,5,4,3,2,1)$",
        r"Shift $S$: $FSF^H = \mathrm{diag}(e^{-2\pi ik/n})$",
        "(a delay only rotates each coefficient's phase)",
    ], spacing=0.086, fontsize=12)
    save_04(fig, name="05_the_dft_matrix.png")


# ================================================================ module 05, notebook 01
def domain_colouring_unit(values, modulus_scale=None):
    """Notebook 05/01's colouring: hue = arg f, brightness from |f| / modulus_scale (or constant)."""
    hue = (np.angle(values) / (2 * np.pi)) % 1.0
    if modulus_scale is None:
        brightness = np.ones_like(hue)
    else:
        brightness = np.clip(np.abs(values) / modulus_scale, 0, 1) ** 0.5
    return hsv_to_rgb(np.dstack([hue, np.full_like(hue, 0.9), brightness]))


def sheet_05_01_limits_and_continuity():
    # 4.1 epsilon-delta check with rng(0)
    rng = np.random.default_rng(0)
    z0, limit_value = 1 + 1j, 2j
    worst_values = []
    for epsilon in [1.0, 0.1, 0.001]:
        delta = min(1.0, epsilon / 4)
        radii = delta * np.sqrt(rng.uniform(0, 1, 100_000))
        angles = rng.uniform(-np.pi, np.pi, 100_000)
        z = z0 + radii * np.exp(1j * angles)
        worst_values.append(np.max(np.abs(z ** 2 - limit_value)))
    assert [f"{v:.6f}" for v in worst_values] == ["0.769478", "0.071325", "0.000707"]

    # 4.2 conj(z)/z along paths
    t = np.array([1e-1, 1e-3, 1e-6, 1e-9])
    spiral = t * np.exp(1j / t)
    assert np.allclose(np.round(np.conj(spiral) / spiral, 3), [0.408 - 0.913j, -0.367 - 0.93j, 0.755 + 0.656j,
                                                               0.404 - 0.915j])
    assert np.allclose(np.conj(t * np.exp(1j * np.pi / 4)) / (t * np.exp(1j * np.pi / 4)), -1j)
    # 4.3 sympy: limit along y = m x
    x, m = sp.symbols("x m", real=True)
    zl = x + sp.I * m * x
    on_line = sp.simplify(sp.limit(sp.simplify(sp.conjugate(zl) / zl), x, 0))
    assert sp.simplify(on_line - (1 - sp.I * m) ** 2 / (1 + m ** 2)) == 0
    # 4.5 Arg jumps; Example 6 limit at infinity
    assert f"{np.angle(-1 + 1j / 1000):+.6f}" == "+3.140593" and f"{np.angle(-1 - 1j / 1000):+.6f}" == "-3.140593"
    w = sp.symbols("w")
    assert sp.limit(((2 / w + 1) / (1 / w - 3)).simplify(), w, 0) == 2

    fig = new_figure("A complex limit must be the same along every path into the point",
                     r"$\mathrm{lim}_{z \to z_0}\, f(z) = L$:  for every $\varepsilon > 0$ some $\delta > 0$ gives "
                     r"$0 < |z - z_0| < \delta \Rightarrow |f(z) - L| < \varepsilon$.   "
                     r"$\bar z / z = e^{-2i\theta}$ on each ray: no limit at 0")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 epsilon-delta: two discs in one plane
    ax = fig.add_subplot(gs[0, 0])
    epsilon = 0.4
    delta = min(1.0, epsilon / 4)
    ring = np.exp(1j * np.linspace(-np.pi, np.pi, 200))
    disc = z0 + delta * ring
    image = disc ** 2
    assert np.max(np.abs(image - 2j)) < epsilon
    eps_disc = 2j + epsilon * ring
    ax.fill(eps_disc.real, eps_disc.imag, color=ORANGE_LIGHT, alpha=0.7, label=r"$\varepsilon$-disc, $\varepsilon = 0.4$")
    ax.fill(image.real, image.imag, color=BLUE, alpha=0.75, label=r"image of the $\delta$-disc under $z^2$")
    ax.fill(disc.real, disc.imag, color=BLUE_LIGHT, alpha=0.9, label=r"$\delta$-disc, $\delta = \min(1, \varepsilon/4) = 0.1$")
    ax.plot(disc.real, disc.imag, color=BLUE, linewidth=1)
    ax.plot([1, 0], [1, 2], "o", color=INK, markersize=4)
    ax.annotate("", xy=(0.25, 1.85), xytext=(0.9, 1.12),
                arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.4, connectionstyle="arc3,rad=0.3"))
    ax.text(0.73, 1.62, r"$z \mapsto z^2$", fontsize=11, color=INK2)
    ax.text(1.05, 0.88, r"$z_0 = 1+i$", fontsize=10.5)
    ax.text(0, 1.55, r"$L = 2i$", fontsize=10.5, ha="center", va="top")
    argand(ax, (-0.6, 1.6), (0.6, 2.9), r"$\mathrm{lim}_{z \to 1+i}\, z^2 = 2i$: the $\delta$-disc lands inside")
    boxed_legend(ax, loc="upper right", fontsize=8.4)

    # 5.2 domain colouring of conj(z)/z
    ax = fig.add_subplot(gs[0, 1])
    g = np.linspace(-1, 1, 601)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    Z[np.abs(Z) == 0] = 1e-12
    ax.imshow(domain_colouring_unit(np.conj(Z) / Z), extent=[-1, 1, -1, 1], origin="lower")
    for radius in [0.5, 0.25, 0.1]:
        c = radius * unit_circle(200)
        ax.plot(c.real, c.imag, color="white", linewidth=0.9, linestyle="--")
    ax.set_aspect("equal")
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(r"$\bar z / z$: every colour in every disc around 0")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 5.3 values along paths
    ax = fig.add_subplot(gs[0, 2])
    tt = np.geomspace(1, 1e-3, 400)
    paths = [(r"ray $\theta = 0$", tt, BLUE), (r"ray $\theta = \pi/4$", tt * np.exp(1j * np.pi / 4), ORANGE),
             (r"ray $\theta = \pi/2$", 1j * tt, AQUA), (r"ray $\theta = 3\pi/4$", tt * np.exp(3j * np.pi / 4), PURPLE)]
    circle = unit_circle()
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=5, zorder=0)
    spiral_path = tt * np.exp(1j * np.log(tt))
    ws = np.conj(spiral_path) / spiral_path
    ax.plot(ws.real, ws.imag, color=INK2, linewidth=1, label=r"spiral $t\,e^{i\ln t}$: never settles", zorder=1)
    for label, zp, color in paths:
        wv = np.conj(zp) / zp
        assert np.allclose(wv, wv[0])
        ax.plot(wv.real[-1], wv.imag[-1], "o", markersize=13, color=color,
                label=f"{label} → {cfmt(np.round(wv[-1], 6))}", zorder=3)
    argand(ax, (-1.45, 1.45), (-1.45, 2.3), r"Values of $\bar z / z$ as $z \to 0$ along paths")
    ax.set_yticks(np.arange(-1, 1.01, 0.5))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, columnspacing=0.8)

    # 5.4 Arg jumps
    ax = fig.add_subplot(gs[1, 0])
    g = np.linspace(-2, 2, 401)
    X, Y = np.meshgrid(g, g)
    image = ax.imshow(np.angle(X + 1j * Y), extent=[-2, 2, -2, 2], origin="lower", cmap=DIVERGING)
    ax.plot([-2, 0], [0, 0], color=INK, linewidth=3, label=r"jump from $\pi$ to $-\pi$")
    ax.plot([-1, -1], [0.1, -0.1], "o", color=INK, markersize=5)
    ax.annotate(r"Arg$(-1 + i/n) \to \pi$", (-1, 0.1), xytext=(-1.9, 0.75), fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.annotate(r"Arg$(-1 - i/n) \to -\pi$", (-1, -0.1), xytext=(-1.9, -0.85), fontsize=9.5, color="white",
                arrowprops=dict(arrowstyle="-", color="white"))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Arg z (radians)")
    cbar.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Arg z: continuous except on the negative axis")
    boxed_legend(ax, loc="lower right", fontsize=8.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 5.5 the Riemann sphere, cut along the real axis
    ax = fig.add_subplot(gs[1, 1])
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED, linewidth=1.4)
    ax.axhline(0, color=INK2, linewidth=1)
    for xv, color in [(0.3, BLUE), (1.0, AQUA), (3.0, ORANGE), (10.0, PURPLE), (-0.6, BLUE)]:
        X_s, Z_s = 2 * xv / (xv ** 2 + 1), (xv ** 2 - 1) / (xv ** 2 + 1)
        ax.plot([0, xv], [1, 0], color=color, linewidth=1, alpha=0.8)
        ax.plot(xv, 0, "o", color=color, markersize=6)
        ax.plot(X_s, Z_s, "s", color=color, markersize=7, markeredgecolor=INK)
        if xv < 5:
            ax.text(xv, -0.12, f"z = {xv:g}".replace("-", "−"), ha="center", va="top", fontsize=9, color=color)
    ax.text(2.6, 0.82, "→ z = 10", ha="center", va="bottom", fontsize=9, color=PURPLE)
    ax.plot(0, 1, "o", color=INK, markersize=8)
    ax.text(0.1, 1.12, r"$N = \infty$", fontsize=11)
    ax.plot(0, -1, "o", color=INK, markersize=8)
    ax.text(0.1, -1.2, r"$S = 0$", fontsize=11)
    ax.set_xlim(-1.5, 3.6)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("Re z (the plane)")
    ax.set_ylabel("Z (height on sphere)")
    ax.set_title(r"Riemann sphere (slice): as $|z| \to \infty$, climb to $N$")

    panel(fig.add_subplot(gs[1, 2]), "The worked numbers", [
        r"$|z^2 - 2i| = |z - z_0|\,|z + z_0| < 4\,|z - z_0|$  if $|z - z_0| < 1$",
        r"so $\delta = \min(1, \varepsilon/4)$; 100,000 points (seed 0):",
        "`eps = 1      max |z^2 - 2i| = 0.769478`",
        "`eps = 0.1    max |z^2 - 2i| = 0.071325`",
        "`eps = 0.001  max |z^2 - 2i| = 0.000707`",
        "",
        r"$\bar z/z$ on $y = mx$: $(1 - im)^2/(1 + m^2)$ (sympy)",
        r"$m = 0 \to 1$,  $m = 1 \to -i$,  imaginary axis $\to -1$",
        "",
        r"At $\infty$: $z = 1/w$ turns $\frac{2z+1}{z-3}$ into $\frac{2+w}{1-3w} \to 2$",
    ], spacing=0.084, fontsize=12)
    save_05(fig, name="01_limits_and_continuity.png")


# ================================================================ module 05, notebook 02
def cauchy_riemann(f_of_z, x, y, z):
    expression = sp.expand_complex(f_of_z.subs(z, x + sp.I * y))
    u = sp.simplify(sp.re(expression))
    v = sp.simplify(sp.im(expression))
    return (u, v, sp.simplify(sp.diff(u, x) - sp.diff(v, y)), sp.simplify(sp.diff(u, y) + sp.diff(v, x)))


def sheet_05_02_the_complex_derivative_and_cauchy_riemann():
    x, y = sp.symbols("x y", real=True)
    z = sp.symbols("z")
    residuals = {name: cauchy_riemann(f, x, y, z)[2:] for name, f in
                 {"z^2": z ** 2, "z^3": z ** 3, "exp(z)": sp.exp(z), "1/z": 1 / z,
                  "conj(z)": sp.conjugate(z), "|z|^2": z * sp.conjugate(z)}.items()}
    assert [str(r) for r in residuals["conj(z)"]] == ["2", "0"]
    assert [str(r) for r in residuals["|z|^2"]] == ["2*x", "2*y"]
    assert all(r == (0, 0) for name, r in residuals.items() if name not in ("conj(z)", "|z|^2"))

    # 4.3 directional difference quotients at z0 = 1 + i
    z0 = 1 + 1j
    printed = {"z^2": ("+2.010000+2.000000i  +2.007071+2.007071i  +2.000000+2.010000i  +1.990000+2.000000i",
                       "+2.000001+2.000000i  +2.000001+2.000001i  +2.000000+2.000001i  +1.999999+2.000000i"),
               "conj(z)": ("+1.000000+0.000000i  +0.000000-1.000000i  -1.000000-0.000000i  +1.000000+0.000000i",
                           "+1.000000+0.000000i  +0.000000-1.000000i  -1.000000-0.000000i  +1.000000+0.000000i")}
    functions = {"z^2": lambda q: q ** 2, "conj(z)": np.conj}
    for name, f in functions.items():
        for t, line in zip([1e-2, 1e-6], printed[name]):
            row = [(f(z0 + t * np.exp(1j * th)) - f(z0)) / (t * np.exp(1j * th)) for th in [0, np.pi / 4, np.pi / 2, np.pi]]
            assert "  ".join(f"{q.real:+.6f}{q.imag:+.6f}i" for q in row) == line
    # 4.4 A and B for |z|^2
    A_w, B_w = 1 - 1j, 1 + 1j

    fig = new_figure("Complex-differentiable means the Jacobian is a rotation-scaling",
                     r"$f'(z_0) = \mathrm{lim}_{h \to 0} \frac{f(z_0 + h) - f(z_0)}{h}$ from every direction.   "
                     r"Horizontal = vertical step  ⟹  $u_x = v_y$,  $u_y = -v_x$,  $f' = u_x + i\,v_x$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 quotients from 72 directions
    ax = fig.add_subplot(gs[0, 0])
    t = 1e-4
    theta = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    h = t * np.exp(1j * theta)
    cases = [(r"$z^2$ → point $2z_0 = 2+2i$", lambda q: q ** 2, BLUE),
             (r"$e^z$ → point $e^{1+i}$", np.exp, AQUA),
             (r"$x^2 + iy^2$ → point 2", lambda q: q.real ** 2 + 1j * q.imag ** 2, INK),
             (r"$\bar z$ → circle $e^{-2i\theta}$", np.conj, ORANGE),
             (r"$|z|^2$ → circle $A + Be^{-2i\theta}$", lambda q: np.abs(q) ** 2, PURPLE)]
    for label, f, color in cases:
        quotients = (f(z0 + h) - f(z0)) / h
        ax.plot(quotients.real, quotients.imag, "o", markersize=4 if "circle" in label else 9, color=color,
                label=label, zorder=3)
    q_abs = (np.abs(z0 + h) ** 2 - abs(z0) ** 2) / h
    assert np.allclose(q_abs, A_w + B_w * np.exp(-2j * theta), atol=1e-3)
    argand(ax, (-1.4, 6.6), (-2.7, 2.8), r"Difference quotients from 72 directions at $z_0 = 1+i$")
    boxed_legend(ax, loc="center right", fontsize=8.3, title="point = differentiable", title_fontsize=8.5)

    # 5.3 tiny squares
    sub = gs[0, 1].subgridspec(2, 2, hspace=0.55, wspace=0.45)
    zc = 1 + 0.5j
    side = 0.12
    corners = zc + side * np.array([-0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j, -0.5 + 0.5j])
    edge = np.linspace(0, 1, 50)
    outline = np.concatenate([corners[k] + edge * (corners[(k + 1) % 4] - corners[k]) for k in range(4)])
    corner_colours = [ORANGE, AQUA, BLUE, PURPLE]
    square_cases = [("square at 1 + 0.5i", lambda q: q), (r"$z^2$: rotated square", lambda q: q ** 2),
                    (r"$\bar z$: flipped", np.conj), (r"$x^2 + iy^2$: rectangle", lambda q: q.real ** 2 + 1j * q.imag ** 2)]
    for k, (title, f) in enumerate(square_cases):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        img = f(outline)
        ax.fill(img.real, img.imag, color=BLUE_LIGHT, alpha=0.7)
        ax.plot(img.real, img.imag, color=INK, linewidth=1)
        for corner, colour in zip(corners, corner_colours):
            mapped = f(np.array([corner]))[0]
            ax.plot(mapped.real, mapped.imag, "o", color=colour, markersize=6)
        centre = f(np.array([zc]))[0]
        ax.set_xlim(centre.real - 0.22, centre.real + 0.22)
        ax.set_ylim(centre.imag - 0.22, centre.imag + 0.22)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        ax.grid(False)
    fig.text(0.515, 0.475, "corners counter-clockwise: orange, green, blue, purple", ha="center", fontsize=9,
             color=INK2)

    panel(fig.add_subplot(gs[0, 2]), "Two directions, one derivative", [
        r"real step $h = t$:        $f' = u_x + i\,v_x$",
        r"imaginary step $h = it$:  $f' = (u_y + i\,v_y)/i = v_y - i\,u_y$",
        r"equal  ⟹  $u_x = v_y$,  $u_y = -v_x$",
        "",
        "Jacobian [[u_x, u_y], [v_x, v_y]] = [[a, −b], [b, a]]",
        r"= multiply by $f' = a + ib$;  $\det J = |f'|^2 \geq 0$",
        "",
        r"$z^2$ at $1+i$, $t = 10^{-6}$, $\theta = 0, \pi/4, \pi/2, \pi$:",
        "`2.000001+2i  2.000001+2.000001i ...`",
        r"$\bar z$: $\bar h / h$ = 1, −i, −1, 1 for every $t$",
    ], spacing=0.085, fontsize=12)

    # 5.2 convergence for e^z
    ax = fig.add_subplot(gs[1, 0])
    steps = np.geomspace(1e-1, 1e-7, 40)
    for (label, th), color in zip([(r"$\theta = 0$", 0.0), (r"$\theta = \pi/3$", np.pi / 3),
                                   (r"$\theta = \pi/2$", np.pi / 2), (r"$\theta = 5\pi/4$", 5 * np.pi / 4)],
                                  [BLUE, ORANGE, AQUA, PURPLE]):
        hh = steps * np.exp(1j * th)
        ax.loglog(steps, np.abs((np.exp(z0 + hh) - np.exp(z0)) / hh - np.exp(z0)), color=color, linewidth=1.8,
                  label=label)
    ax.loglog(steps, steps, "--", color=INK, linewidth=1, label="slope 1")
    ax.invert_xaxis()
    ax.grid(True, which="major")
    ax.set_xlabel("step size t = |h|")
    ax.set_ylabel(r"error $|Q - f'(z_0)|$")
    ax.set_title(r"$e^z$ at $1+i$: error shrinks like $t$ from every side")
    boxed_legend(ax, loc="upper right", fontsize=8.6)

    # 4.2 where CR holds
    ax = fig.add_subplot(gs[1, 1])
    line = np.linspace(-2, 2, 50)
    ax.plot(line, line, color=ORANGE, linewidth=3.5, label=r"$x^2 + iy^2$: $y = x$", zorder=2)
    ax.plot(line, -line, color=AQUA, linewidth=2, linestyle="--", label=r"$x^3 + iy^3$: $y = \pm x$", zorder=3)
    ax.plot(line, line, color=AQUA, linewidth=2, linestyle="--", zorder=3)
    ax.plot(0, 0, "o", color=PURPLE, markersize=12, label=r"$|z|^2$: only $z = 0$", zorder=4)
    sol = sp.solve([sp.Eq(sp.diff(x ** 3, x), sp.diff(y ** 3, y))], [x, y], dict=True)
    assert len(sol) == 2
    argand(ax, (-2, 2), (-2, 2.9), "Where the CR equations hold (not open sets!)")
    ax.set_yticks(np.arange(-2, 2.1, 1))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, columnspacing=0.8)

    panel(fig.add_subplot(gs[1, 2]), "CR residuals (sympy)", [
        "`f         u_x - v_y   u_y + v_x`",
        "`z^2       0           0`",
        "`z^3       0           0`",
        "`exp(z)    0           0`",
        "`1/z       0           0`",
        "`conj(z)   2           0`",
        "`|z|^2     2*x         2*y`",
        "",
        r"Passing: $f' = u_x + i\,v_x$ = $2z$, $3z^2$, $e^z$, $-1/z^2$",
        r"$|z|^2$: $A + Be^{-2i\theta}$ with $A = 1-i$, $B = 1+i$ at $1+i$",
    ], spacing=0.082, fontsize=12)
    save_05(fig, name="02_the_complex_derivative_and_cauchy_riemann.png")


# ================================================================ module 05, notebook 03
def domain_colouring_soft(values):
    """Notebook 05/03's colouring: hue = arg f, brightness rising with |f|."""
    hue = (np.angle(values) / (2 * np.pi)) % 1.0
    brightness = 1 - 1 / (1 + np.abs(values) ** 0.6)
    return hsv_to_rgb(np.dstack([hue, np.full_like(hue, 0.9), 0.15 + 0.85 * brightness]))


def sheet_05_03_holomorphic_functions():
    x, y = sp.symbols("x y", real=True)
    z, zb = sp.symbols("z zb")

    def wirtinger(f_xy):
        g = sp.simplify(sp.expand(f_xy.subs({x: (z + zb) / 2, y: (z - zb) / (2 * sp.I)}, simultaneous=True)))
        return g, sp.simplify(sp.diff(g, z)), sp.simplify(sp.diff(g, zb))

    table = {"z^3 + 2z": sp.expand((x + sp.I * y) ** 3 + 2 * (x + sp.I * y)), "conj(z)": x - sp.I * y,
             "|z|^2": x ** 2 + y ** 2, "Re z": x, "Im z": y,
             "x^2-y^2+2ixy+3x+3iy": x ** 2 - y ** 2 + 2 * sp.I * x * y + 3 * x + 3 * sp.I * y}
    results = {name: [str(e) for e in wirtinger(f)] for name, f in table.items()}
    assert results["z^3 + 2z"] == ["z*(z**2 + 2)", "3*z**2 + 2", "0"]
    assert results["conj(z)"][2] == "1" and results["|z|^2"] == ["z*zb", "zb", "z"]
    assert results["Re z"][2] == "1/2" and results["Im z"][2] == "I/2"
    assert results["x^2-y^2+2ixy+3x+3iy"] == ["z*(z + 3)", "2*z + 3", "0"]

    def numerical_d_dzbar(f, q, step=1e-6):
        d_dx = (f(q + step) - f(q - step)) / (2 * step)
        d_dy = (f(q + 1j * step) - f(q - 1j * step)) / (2 * step)
        return 0.5 * (d_dx + 1j * d_dy)

    pts = np.array([0.3 + 0.2j, -1 + 2j, 2 - 0.5j])
    assert np.allclose(np.round(numerical_d_dzbar(lambda q: np.abs(q) ** 2, pts), 6), pts)
    assert np.allclose(np.round(numerical_d_dzbar(np.exp, pts), 6), 0)
    vertical = (np.log(-1 + 1e-6j) - np.log(-1 - 1e-6j)) / (2j * 1e-6)
    assert f"{vertical:.4g}" == "3.142e+06+0j"

    fig = new_figure("Holomorphic: complex-differentiable on a whole open set",
                     r"Wirtinger $\frac{\partial}{\partial \bar z} = \frac{1}{2}(\partial_x + i\,\partial_y)$:  "
                     r"$\frac{\partial f}{\partial \bar z} = \frac{1}{2}[(u_x - v_y) + i(v_x + u_y)]$ packs both CR "
                     r"equations;  holomorphic ⟺ $\frac{\partial f}{\partial \bar z} = 0$ on an open set, then "
                     r"$f' = \frac{\partial f}{\partial z}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 maps of |df/dzbar|
    sub = gs[0, 0:2].subgridspec(1, 4, wspace=0.35)
    g = np.linspace(-2, 2, 301)
    X, Y = np.meshgrid(g, g)
    maps = [(r"$z^2 + e^z$: 0 everywhere", np.zeros_like(X)),
            (r"$|z|^2$: 0 only at 0", np.abs(X + 1j * Y)),
            (r"$x^2 + iy^2$: 0 on $y = x$", np.abs(X - Y)),
            (r"Re $z$: never 0", np.full_like(X, 0.5))]
    for k, (title, magnitude) in enumerate(maps):
        ax = fig.add_subplot(sub[0, k])
        image = ax.imshow(magnitude, extent=[-2, 2, -2, 2], origin="lower", cmap=SEQ.reversed(), vmin=0, vmax=3)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Re")
        if k == 0:
            ax.set_ylabel("Im")
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(image, ax=fig.axes[-4:], fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label(r"$|\partial f / \partial \bar z|$ (dark = CR hold)")
    cbar.outline.set_visible(False)
    fig.text(0.34, 0.53, r"Holomorphic needs a whole dark region; a dark point or line is not enough.",
             ha="center", fontsize=10.5, color=INK2)

    panel(fig.add_subplot(gs[0, 2]), "The practical test (sympy)", [
        r"Write $f$ with $z$, $\bar z$; differentiate in $\bar z$:",
        "`f                  in z, zb      d/dzbar`",
        "`z^3 + 2z           z*(z**2 + 2)  0`",
        "`conj(z)            zb            1`",
        "`|z|^2              z*zb          z`",
        "`Re z               z/2 + zb/2    1/2`",
        "`Im z               I*(zb - z)/2  I/2`",
        "`x^2-y^2+2ixy+3x+3iy z*(z + 3)    0`",
        "",
        r"Rigid: Re $f$, Im $f$ or $|f|$ constant ⟹ $f$ constant.",
        r"Log $z$: vertical quotient at $-1$ ≈ $\pi/10^{-6}$ (the cut).",
    ], spacing=0.078, fontsize=12)

    # 5.2 domain colouring: orientation
    g = np.linspace(-2, 2, 601)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    loop = 1 + 0.5 * unit_circle(100)
    for k, (values, title) in enumerate([(Z ** 2 - 1, r"$z^2 - 1$: colours run counter-clockwise"),
                                         (np.conj(Z) ** 2 - 1, r"$\bar z^2 - 1$: colours run backwards")]):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(domain_colouring_soft(values), extent=[-2, 2, -2, 2], origin="lower")
        ax.plot(loop.real, loop.imag, color="white", linewidth=1.4)
        ax.annotate("", xy=(1.0, 0.5), xytext=(1.05, 0.497), arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
        ax.axhline(0, color=INK, linewidth=0.5)
        ax.axvline(0, color=INK, linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 5.3 the ladder: x|x| versus z^2
    ax = fig.add_subplot(gs[1, 2])
    xs = np.linspace(-2, 2, 401)
    ax.plot(xs, xs * np.abs(xs), color=BLUE, linewidth=2, label=r"$x|x|$")
    ax.plot(xs, 2 * np.abs(xs), color=ORANGE, linewidth=2, label=r"$(x|x|)' = 2|x|$: corner")
    ax.plot(xs[xs < 0], -2 * np.ones(np.sum(xs < 0)), color=AQUA, linewidth=2.4,
            label=r"$(x|x|)'' = 2\,\mathrm{sign}(x)$: jump")
    ax.plot(xs[xs > 0], 2 * np.ones(np.sum(xs > 0)) + 0.0, color=AQUA, linewidth=2.4)
    ax.plot(xs, xs ** 2, "--", color=INK2, linewidth=1.2, label=r"$z^2$ on ℝ: $2z$, $2$, $0$, … all smooth")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-4.2, 6.5)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_title(r"Real $x|x|$: differentiable once, not twice")
    boxed_legend(ax, loc="upper center", fontsize=8.4)
    save_05(fig, name="03_holomorphic_functions.png")


# ================================================================ module 05, notebook 04
def grid_image(ax, f, x_values, y_values, samples=400):
    x_range = (min(x_values), max(x_values))
    y_range = (min(y_values), max(y_values))
    for a in x_values:
        image = f(a + 1j * np.linspace(*y_range, samples))
        ax.plot(image.real, image.imag, color=BLUE, linewidth=1)
    for b in y_values:
        image = f(np.linspace(*x_range, samples) + 1j * b)
        ax.plot(image.real, image.imag, color=ORANGE, linewidth=1)


def sheet_05_04_conformal_maps():
    # 4.1 tangent vectors
    z0 = 1 + 1j
    v1, v2 = np.exp(0.2j), np.exp(1.4j)
    push = lambda f, d, eps=1e-7: (f(z0 + eps * d) - f(z0 - eps * d)) / (2 * eps)
    w1, w2 = push(lambda q: q ** 2, v1), push(lambda q: q ** 2, v2)
    assert np.allclose(np.round(w1, 6), 1.562794 + 2.357472j)
    assert round(abs(w1), 6) == 2.828427 and round(np.angle(w1 / v1), 6) == 0.785398
    assert round(np.angle(w2 / w1), 6) == 1.2

    # 4.2 random angle test (seed 1)
    rng = np.random.default_rng(1)
    maps = {"z^2": lambda q: q ** 2, "exp(z)": np.exp, "z + 1/z": lambda q: q + 1 / q, "conj(z)": np.conj,
            "x^2 + i y^2": lambda q: q.real ** 2 + 1j * q.imag ** 2}
    points = rng.uniform(0.3, 2, 200) * np.exp(1j * rng.uniform(-np.pi, np.pi, 200))
    d1 = np.exp(1j * rng.uniform(-np.pi, np.pi, 200))
    d2 = np.exp(1j * rng.uniform(-np.pi, np.pi, 200))
    eps = 1e-6
    changes = {}
    for name, f in maps.items():
        a1 = (f(points + eps * d1) - f(points - eps * d1)) / (2 * eps)
        a2 = (f(points + eps * d2) - f(points - eps * d2)) / (2 * eps)
        before, after = np.angle(d2 / d1), np.angle(a2 / a1)
        changes[name] = (np.max(np.abs(np.abs(after) - np.abs(before))), np.max(np.abs(after - before)))
    assert all(changes[n][1] < 1e-8 for n in ("z^2", "exp(z)", "z + 1/z")) and changes["conj(z)"][0] < 1e-8
    assert f"{changes['conj(z)'][1]:.2e}" == "6.27e+00" and f"{changes['x^2 + i y^2'][0]:.2e}" == "2.30e+00"

    # 4.4 Joukowski
    radius = 1.5
    th = np.linspace(0, 2 * np.pi, 9)
    ell = radius * np.exp(1j * th) + 1 / (radius * np.exp(1j * th))
    assert round(ell.real.max(), 4) == 2.1667 and round(ell.imag.max(), 4) == 0.8333

    fig = new_figure("Holomorphic maps with f′ ≠ 0 preserve angles: tiny squares stay squares",
                     r"$\frac{d}{dt}f(\gamma(t)) = f'(z_0)\,\gamma'(0)$: every direction is rotated by "
                     r"$\arg f'(z_0)$ and scaled by $|f'(z_0)|$;   where $f(z) - f(z_0) \approx c(z - z_0)^m$, "
                     r"angles are multiplied by $m$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    grid_image(ax, lambda q: q ** 2, np.linspace(0.2, 2, 10), np.linspace(0.2, 2, 10))
    ax.plot([], [], color=BLUE, label="images of x = a")
    ax.plot([], [], color=ORANGE, label="images of y = b")
    argand(ax, (-4, 4), (0, 9.4), r"$w = z^2$ on $0.2 \leq x, y \leq 2$: orthogonal parabolas")
    ax.set_yticks(np.arange(0, 8.1, 2))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.6)

    ax = fig.add_subplot(gs[0, 1])
    grid_image(ax, np.exp, np.linspace(-1.5, 1.2, 10), np.linspace(0, np.pi, 10))
    ax.plot([], [], color=BLUE, label=r"$x = a$ → circle $|w| = e^a$")
    ax.plot([], [], color=ORANGE, label=r"$y = b$ → ray $\arg w = b$")
    argand(ax, (-3.6, 3.6), (-0.3, 4.6), r"$w = e^z$ on the strip $0 \leq y \leq \pi$")
    ax.set_yticks(np.arange(0, 3.1, 1))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, columnspacing=0.8)

    # 5.2 zooming in
    sub = gs[0, 2].subgridspec(1, 2, wspace=0.35)
    zc = 1 + 0.4j
    half = 0.05
    offsets = np.linspace(-half, half, 6)
    param = np.linspace(-half, half, 100)
    for k, title in enumerate([r"$(z^2 - z_0^2)/f'(z_0)$" + "\nsame grid back",
                               r"$x^2 + iy^2$, $\div\, 2x_0$" + "\nflat rectangles"]):
        ax = fig.add_subplot(sub[0, k])
        for c in offsets:
            for line, colour in [(zc + c + 1j * param, BLUE), (zc + param + 1j * c, ORANGE)]:
                if k == 0:
                    image = (line ** 2 - zc ** 2) / (2 * zc)
                else:
                    image = (line.real ** 2 + 1j * line.imag ** 2 - (zc.real ** 2 + 1j * zc.imag ** 2)) / (2 * zc.real)
                ax.plot(image.real, image.imag, color=colour, linewidth=1.2)
        ax.set_xlim(-0.07, 0.07)
        ax.set_ylim(-0.07, 0.07)
        ax.set_aspect("equal")
        ax.set_xticks([-0.05, 0, 0.05])
        ax.set_yticks([-0.05, 0, 0.05])
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Re (offset)", fontsize=9.5)
        if k == 0:
            ax.set_ylabel("Im (offset)", fontsize=9.5)
    fig.text(0.86, 0.505, r"tiny grid around $z_0 = 1 + 0.4i$; Jacobian of $x^2 + iy^2$ is diag(2, 0.8)",
             ha="center", fontsize=9, color=INK2)

    # 5.3 angles double at the critical point
    ax = fig.add_subplot(gs[1, 0])
    tt = np.linspace(0, 1.2, 100)
    for angle_deg, colour in zip([0, 30, 60, 90], [BLUE, ORANGE, AQUA, PURPLE]):
        ray = tt * np.exp(1j * np.radians(angle_deg))
        ax.plot((ray ** 2).real, (ray ** 2).imag, color=colour, linewidth=2.2,
                label=f"{angle_deg}° → {2 * angle_deg}°")
        short = 1 + 0.35 * tt * np.exp(1j * np.radians(angle_deg))
        ax.plot((short ** 2).real, (short ** 2).imag, color=colour, linewidth=1.2, linestyle="--")
    arc = 0.3 * np.exp(1j * np.linspace(0, np.pi, 50))
    ax.plot(arc.real, arc.imag, color=INK)
    ax.text(-0.2, 0.38, "180°", fontsize=10)
    ax.text(1.35, 1.0, "rays from 1 (dashed):\nangles kept, 30° apart", fontsize=9, color=INK2)
    argand(ax, (-1.6, 2.6), (-0.3, 2.75), r"Rays under $z^2$: doubled at 0, kept at 1")
    ax.set_yticks(np.arange(0, 2.1, 0.5))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, title="ray from 0 → its image", title_fontsize=8.4)
    for power in (2, 3):
        img1, img2 = (1e-3) ** power, (1e-3 * np.exp(1j * np.pi / 6)) ** power
        assert round(np.degrees(np.angle(img2 / img1)), 1) == 30.0 * power

    # 5.4 Joukowski airfoil
    ax = fig.add_subplot(gs[1, 1])
    centre = -0.1 + 0.12j
    r_air = abs(1 - centre)
    theta = np.linspace(0, 2 * np.pi, 600)
    for scale in [1.0, 1.15, 1.35, 1.6, 2.0, 2.6]:
        circ = centre + scale * r_air * np.exp(1j * theta)
        image = circ + 1 / circ
        style = dict(color=INK, linewidth=2.6) if scale == 1.0 else dict(color=BLUE, linewidth=0.9)
        ax.plot(image.real, image.imag, **style)
    radial = np.linspace(1.0, 2.6, 100) * r_air
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        spoke = centre + radial * np.exp(1j * angle)
        image = spoke + 1 / spoke
        ax.plot(image.real, image.imag, color=ORANGE, linewidth=0.9)
    ax.plot([2], [0], "x", color=ORANGE, markersize=11, markeredgewidth=2.5, zorder=5)
    ax.annotate(r"cusp at $J(1) = 2$", (2, 0), xytext=(2.1, -1.2), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2))
    argand(ax, (-3.6, 3.6), (-2.6, 3.3), r"$J(z) = z + 1/z$ on circles about $-0.1+0.12i$")
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.text(0, 2.85, "the circle through z = 1 becomes an airfoil (black)", ha="center", fontsize=9.3, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "The numbers", [
        "z² at 1 + i: f′ = 2 + 2i, so every tiny arrow is",
        "rotated 45°, stretched 2.828 (areas × 8);",
        "two tangents 1.2 rad apart stay 1.2 rad apart.",
        "",
        "Max angle change, 200 random points (seed 1):",
        "`z^2, e^z, z + 1/z   < 1e-9  (rounding)`",
        "`conj(z)   size kept, sign flipped (6.27)`",
        "`x^2 + iy^2   size changes up to 2.30`",
        "",
        r"$J' = 1 - 1/z^2 = 0$ at $\pm 1$;  $|z| = 1.5$ → ellipse",
        "with semi-axes 2.1667 and 0.8333, foci ±2.",
    ], spacing=0.08, fontsize=12)
    save_05(fig, name="04_conformal_maps.png")


# ================================================================ module 05, notebook 05
def sheet_05_05_harmonic_functions():
    x, y = sp.symbols("x y", real=True)
    z = sp.symbols("z")

    def harmonic_conjugate(u):
        assert sp.simplify(sp.diff(u, x, 2) + sp.diff(u, y, 2)) == 0
        v_partial = sp.integrate(sp.diff(u, x), y)
        phi_prime = sp.simplify(-sp.diff(u, y) - sp.diff(v_partial, x))
        assert not phi_prime.has(y)
        return sp.simplify(v_partial + sp.integrate(phi_prime, x))

    conjugates = {}
    for u in [x ** 2 - y ** 2, sp.exp(x) * sp.cos(y), x ** 3 - 3 * x * y ** 2 + 2 * y, x / (x ** 2 + y ** 2)]:
        v = harmonic_conjugate(u)
        conjugates[str(u)] = (str(v), str(sp.simplify((u + sp.I * v).subs({x: z, y: 0}))))
    assert conjugates["x**2 - y**2"] == ("2*x*y", "z**2")
    assert conjugates["x**3 - 3*x*y**2 + 2*y"] == ("3*x**2*y - 2*x - y**3", "z*(z**2 - 2*I)")
    assert conjugates["x/(x**2 + y**2)"] == ("-y/(x**2 + y**2)", "1/z")

    # 4.4 mean values
    z0 = 0.3 + 0.2j
    theta = np.linspace(0, 2 * np.pi, 2001)
    fns = {"Re z^2": (lambda q: (q ** 2).real, "+0.05000000"), "Re e^z": (lambda q: np.exp(q).real, "+1.32295150"),
           "Im(z^3 - 2iz)": (lambda q: (q ** 3 - 2j * q).imag, "-0.55400000"),
           "ln|z - 2|": (lambda q: np.log(np.abs(q - 2)), "+0.53750121")}
    for name, (u, printed) in fns.items():
        assert f"{u(z0):+.8f}" == printed
        for r in [0.1, 0.5, 1.0]:
            assert f"{trapezoid(u(z0 + r * np.exp(1j * theta)), theta) / (2 * np.pi):+.8f}" == printed
    bowl = [trapezoid(np.abs(z0 + r * np.exp(1j * theta)) ** 2, theta) / (2 * np.pi) for r in [0.1, 0.5, 1.0]]
    assert [f"{b:+.8f}" for b in bowl] == ["+0.14000000", "+0.38000000", "+1.13000000"]

    # 4.5 Jacobi relaxation
    n = 41
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    exact = X ** 2 - Y ** 2
    u_grid = np.zeros_like(exact)
    u_grid[0, :], u_grid[-1, :], u_grid[:, 0], u_grid[:, -1] = exact[0, :], exact[-1, :], exact[:, 0], exact[:, -1]
    errors = []
    for _ in range(4000):
        u_grid[1:-1, 1:-1] = 0.25 * (u_grid[:-2, 1:-1] + u_grid[2:, 1:-1] + u_grid[1:-1, :-2] + u_grid[1:-1, 2:])
        errors.append(np.max(np.abs(u_grid - exact)))
    errors = np.array(errors)
    assert [f"{errors[k - 1]:.2e}" for k in (10, 100, 1000)] == ["5.62e-01", "1.25e-01", "1.13e-07"]
    assert errors[-1] < 1e-14

    fig = new_figure("Re f and Im f of a holomorphic f are harmonic, and their level curves cross at 90°",
                     r"$\Delta u = u_{xx} + u_{yy} = v_{yx} - v_{xy} = 0$;   $\nabla u \cdot \nabla v = 0$, "
                     r"$|\nabla u| = |\nabla v| = |f'|$;   complex potential $\Omega = \varphi + i\psi$, "
                     r"velocity $= \overline{\Omega'(z)}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # 5.1 orthogonal level curves of z^2
    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(-2, 2, 401)
    X, Y = np.meshgrid(g, g)
    W = (X + 1j * Y) ** 2
    ax.contour(X, Y, W.real, levels=16, colors=BLUE, linewidths=1.1, linestyles="solid")
    ax.contour(X, Y, W.imag, levels=16, colors=ORANGE, linewidths=1.1, linestyles="solid")
    ax.plot([], [], color=BLUE, label=r"$u = x^2 - y^2$ = const")
    ax.plot([], [], color=ORANGE, label=r"$v = 2xy$ = const")
    argand(ax, (-2, 2), (-2, 2.9), r"$f = z^2$: hyperbolas meet at right angles", grid_on=False)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.6)

    # 5.3 flow past a cylinder
    ax = fig.add_subplot(gs[0, 1])
    g = np.linspace(-3, 3, 501)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    Z_safe = np.where(np.abs(Z) < 1, np.nan, Z)
    with np.errstate(invalid="ignore"):
        potential = Z_safe + 1 / Z_safe
        speed = np.abs(np.conj(1 - 1 / Z_safe ** 2))
    on_cylinder = np.exp(1j * np.linspace(0, 2 * np.pi, 9))
    assert np.allclose(np.abs(1 - 1 / on_cylinder ** 2), 2 * np.abs(np.sin(np.angle(on_cylinder))))
    image = ax.imshow(speed, extent=[-3, 3, -3, 3], origin="lower", cmap=SEQ, vmin=0, vmax=2)
    ax.contour(X, Y, potential.imag, levels=np.linspace(-3, 3, 25), colors=INK, linewidths=0.9, linestyles="solid")
    ax.contour(X, Y, potential.real, levels=np.linspace(-6, 6, 25), colors=ORANGE, linewidths=0.8, linestyles="dashed")
    cyl = unit_circle(200)
    ax.fill(cyl.real, cyl.imag, color=GRID, zorder=3)
    ax.plot([-1, 1], [0, 0], "o", color=ORANGE, markersize=7, markeredgecolor=INK, zorder=4)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"speed $|\Omega'(z)|$")
    cbar.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(r"$\Omega = z + 1/z$: streamlines (black) around a cylinder")
    for spine in ax.spines.values():
        spine.set_visible(False)

    panel(fig.add_subplot(gs[0, 2]), "Harmonic conjugate: solve CR for v", [
        r"$v_y = u_x$,  $v_x = -u_y$;  works exactly when $\Delta u = 0$",
        r"$u = x^3 - 3xy^2 + 2y$:  $v = 3x^2y - y^3 + \phi(x)$,",
        r"$v_x = 6xy + \phi' = -u_y = 6xy - 2$  ⟹  $\phi = -2x$",
        r"$u + iv = z^3 - 2iz$",
        "`x^2 - y^2    -> v = 2xy,           f = z^2`",
        "`e^x cos y    -> v = e^x sin y,     f = e^z`",
        "`x/(x^2+y^2)  -> v = -y/(x^2+y^2),  f = 1/z`",
        "",
        r"Circle means, $z_0 = 0.3+0.2i$, $r$ = 0.1, 0.5, 1:",
        "`Re e^z   1.32295150 for every r  (= u(z0))`",
        "`x^2+y^2  0.14, 0.38, 1.13  vs u(z0) = 0.13`",
    ], spacing=0.077, fontsize=12)

    # 5.4 source and sink
    ax = fig.add_subplot(gs[1, 0])
    g = np.linspace(-2.5, 2.5, 400)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    values = np.log(Z + 1) - np.log(Z - 1)
    velocity = np.conj(1 / (Z + 1) - 1 / (Z - 1))
    for shift in np.linspace(0, np.pi, 8, endpoint=False):
        ax.contour(X, Y, np.sin(values.real - shift), levels=[0], colors=BLUE, linewidths=0.9, linestyles="solid")
        ax.contour(X, Y, np.sin(values.imag - shift), levels=[0], colors=INK, linewidths=1, linestyles="solid")
    step = 25
    unit_v = velocity / (np.abs(velocity) + 1e-12)
    ax.quiver(X[::step, ::step], Y[::step, ::step], unit_v.real[::step, ::step], unit_v.imag[::step, ::step],
              color=ORANGE, scale=25, width=0.005)
    ax.plot([-1, 1], [0, 0], "o", color=ORANGE, markersize=8, markeredgecolor=INK, zorder=5)
    ax.plot([], [], color=BLUE, label=r"equipotentials $\varphi$ = const")
    ax.plot([], [], color=INK, label=r"streamlines $\psi$ = const")
    argand(ax, (-2.5, 2.5), (-2.5, 3.4), r"Source at $-1$, sink at $+1$: Log$(z+1)$ − Log$(z-1)$", grid_on=False)
    ax.set_yticks(np.arange(-2, 2.1, 1))
    boxed_legend(ax, loc="upper center", ncol=2, fontsize=8.4, columnspacing=0.8)

    # 5.5 mean value on a circle
    ax = fig.add_subplot(gs[1, 1])
    th = np.linspace(0, 2 * np.pi, 600)
    vals = np.exp(z0 + np.exp(1j * th)).real
    centre_value = np.exp(z0).real
    average = trapezoid(vals, th) / (2 * np.pi)
    assert f"{average:.4f}" == f"{centre_value:.4f}" == "1.3230"
    ax.plot(th, vals, color=BLUE, linewidth=2, label=r"$u(z_0 + e^{i\theta})$, $u = $ Re $e^z$")
    ax.axhline(centre_value, color=INK, linestyle="--", linewidth=1.3, label=rf"$u(z_0) = {centre_value:.4f}$")
    ax.fill_between(th, vals, centre_value, where=vals >= centre_value, color=ORANGE_LIGHT, alpha=0.9)
    ax.fill_between(th, vals, centre_value, where=vals < centre_value, color=BLUE_LIGHT, alpha=0.9)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", "π/2", "π", "3π/2", "2π"])
    ax.set_ylim(-0.3, 4.6)
    ax.grid(True)
    ax.set_xlabel(r"$\theta$ on the circle $|z - z_0| = 1$, $z_0 = 0.3 + 0.2i$")
    ax.set_ylabel("u")
    ax.set_title(f"Mean-value property: average = {average:.4f} = u(z₀)")
    boxed_legend(ax, loc="upper center", fontsize=8.6)

    # 4.5 relaxation
    ax = fig.add_subplot(gs[1, 2])
    sweeps = np.arange(1, 4001)
    ax.semilogy(sweeps, errors, color=BLUE, linewidth=2)
    for k, offset in ((10, (10, 2)), (100, (10, -10)), (1000, (8, 6)), (4000, (-62, 8))):
        ax.plot(k, errors[k - 1], "o", color=ORANGE, markersize=7, zorder=3)
        ax.annotate(f"{k} sweeps: {errors[k - 1]:.2e}" if k == 10 else f"{errors[k - 1]:.2e}", (k, errors[k - 1]),
                    textcoords="offset points", xytext=offset, fontsize=9.5)
    ax.grid(True)
    ax.set_xlim(0, 4300)
    ax.set_xlabel("Jacobi sweeps (each value ← average of 4 neighbours)")
    ax.set_ylabel(r"max $|u - (x^2 - y^2)|$")
    ax.set_title(r"Relaxation recovers $x^2 - y^2$ from its boundary")
    save_05(fig, name="05_harmonic_functions.png")


if __name__ == "__main__":
    sheet_04_01_complex_numbers_as_matrices()
    sheet_04_02_complex_eigenvalues_of_real_matrices()
    sheet_04_03_complex_vectors_and_inner_products()
    sheet_04_04_hermitian_and_unitary_matrices()
    sheet_04_05_the_dft_matrix()
    sheet_05_01_limits_and_continuity()
    sheet_05_02_the_complex_derivative_and_cauchy_riemann()
    sheet_05_03_holomorphic_functions()
    sheet_05_04_conformal_maps()
    sheet_05_05_harmonic_functions()
