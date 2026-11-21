#!/usr/bin/env python3
"""Concept sheets for modules 02 (polynomials and roots) and 03 (complex functions).

One page per notebook that summarizes its central idea. Every number is computed here from the
same polynomials, maps, grids and random seeds the notebook uses, and is checked against the
notebook's printed outputs with asserts.
Images are written to modules/02_polynomials_and_roots/img/ and modules/03_complex_functions/img/.

Run from anywhere:  python scripts/make_modules_02_03_concept_images.py
"""
import cmath
import math
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb, to_rgb
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID, SEQ,
                       new_figure, grid, panel)
from viz_style import save as _save

MODULE_02 = REPO / "modules" / "02_polynomials_and_roots"
MODULE_03 = REPO / "modules" / "03_complex_functions"
save_02 = partial(_save, folder=MODULE_02 / "img")
save_03 = partial(_save, folder=MODULE_03 / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
DEEP = LinearSegmentedColormap.from_list("deep_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


# ================================================================ helpers
def argand(ax, xlim, ylim, title, xlabel="Re", ylabel="Im", show_grid=True):
    """Argand diagram: equal aspect, both axes through the origin, axes labelled Re/Im."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    if show_grid:
        ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def arrow(ax, start, end, color, width=2.2, zorder=4, label=None):
    start, end = complex(start), complex(end)
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, mutation_scale=15,
                                shrinkA=0, shrinkB=0))
    if label is not None:
        ax.plot([], [], color=color, linewidth=width, label=label)


def legend(ax, loc="best", fontsize=9, **kw):
    ax.legend(loc=loc, fontsize=fontsize, frameon=True, facecolor=SURFACE, edgecolor="none",
              framealpha=0.92, **kw)


def plain_image_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def domain_colouring(f, re_range, im_range, resolution=500):
    """The notebooks' scheme: hue = arg f, m = (2/pi) arctan|f|, S = min(1, 2 - 2m), V = min(1, 2m)."""
    ratio = (im_range[1] - im_range[0]) / (re_range[1] - re_range[0])
    x = np.linspace(*re_range, resolution)
    y = np.linspace(*im_range, max(2, int(round(resolution * ratio))))
    X, Y = np.meshgrid(x, y)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = f(X + 1j * Y)
        hue = (np.angle(W) / (2 * np.pi)) % 1.0
        m = (2 / np.pi) * np.arctan(np.abs(W))
    hue = np.nan_to_num(hue)
    m = np.nan_to_num(m, nan=1.0)
    hsv = np.stack([hue, np.clip(2 - 2 * m, 0, 1), np.clip(2 * m, 0, 1)], axis=-1)
    return hsv_to_rgb(hsv)


def show_domain_colouring(ax, f, re_range, im_range, title, resolution=500):
    ax.imshow(domain_colouring(f, re_range, im_range, resolution), origin="lower",
              extent=[*re_range, *im_range], aspect="equal", interpolation="bilinear")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    plain_image_axes(ax)


def pi_ticks(ax, axis, multiples, labels):
    ticks = np.pi * np.asarray(multiples, float)
    if axis == "x":
        ax.set_xticks(ticks, labels)
    else:
        ax.set_yticks(ticks, labels)


# ================================================================ module 02, notebook 01
def sheet_02_01_quadratics_and_conjugate_roots():
    # 4.1: square roots from x^2 = (|w| + u)/2, y^2 = (|w| - u)/2, sign(y) = sign(v)
    def complex_sqrt(w):
        u, v = w.real, w.imag
        x = math.sqrt((abs(w) + u) / 2)
        y = math.sqrt((abs(w) - u) / 2)
        return complex(x, -y if v < 0 else y)

    for w in [3 + 4j, -8 - 6j, -9 + 0j, 2j, -5 + 12j, 1 - 1j]:
        assert abs(complex_sqrt(w) - cmath.sqrt(w)) < 1e-12
    assert f"{complex_sqrt(1 - 1j):.6f}" == "1.098684-0.455090j"
    w = -5 + 12j
    delta = cmath.sqrt(w)
    assert abs(delta - (2 + 3j)) < 1e-12 and abs(abs(delta) - math.sqrt(13)) < 1e-12

    # 4.2: Example 3 and the random quadratic with seed 0
    def solve_quadratic(a, b, c):
        d = cmath.sqrt(b * b - 4 * a * c)
        return (-b + d) / (2 * a), (-b - d) / (2 * a)

    z1, z2 = solve_quadratic(1, -(3 + 1j), 4 + 3j)
    assert abs(z1 - (2 - 1j)) < 1e-12 and abs(z2 - (1 + 2j)) < 1e-12
    assert abs((z1 + z2) - (3 + 1j)) < 1e-12 and abs(z1 * z2 - (4 + 3j)) < 1e-12
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=3) + 1j * rng.normal(size=3)
    residuals = [abs(a * r**2 + b * r + c) for r in solve_quadratic(a, b, c)]
    assert max(residuals) < 1e-15
    half_delta = (z1 - z2) / 2
    assert abs(half_delta - (1 - 3j) / 2) < 1e-12

    # 4.4: Example 4 factored from one root
    z = sp.symbols("z")
    p = z**4 - 2 * z**3 + 6 * z**2 - 8 * z + 8
    assert sp.expand(p.subs(z, 1 + sp.I)) == 0
    assert sp.expand((z**2 - 2 * z + 2) * (z**2 + 4) - p) == 0
    assert str(sp.factor(p)) == "(z**2 + 4)*(z**2 - 2*z + 2)"

    # Example 5: Vieta without solving
    e1, e2 = 2, 5
    assert e1**2 - 2 * e2 == -6
    assert abs((1 + 2j) ** 2 + (1 - 2j) ** 2 + 6) < 1e-12

    # 5.2: three degree-9 polynomials, real then complex coefficients, seed 7 in that order
    rng = np.random.default_rng(7)
    real_root_sets, complex_root_sets = [], []
    for _ in range(3):
        real_coefficients = rng.normal(size=10)
        complex_coefficients = rng.normal(size=10) + 1j * rng.normal(size=10)
        real_root_sets.append(np.roots(real_coefficients))
        complex_root_sets.append(np.roots(complex_coefficients))
    for roots in real_root_sets:
        mirrored = np.conj(roots)
        assert all(np.min(np.abs(roots - m)) < 1e-8 for m in mirrored)
        assert np.sum(np.abs(roots.imag) < 1e-9) % 2 == 1
    asymmetric = [max(np.min(np.abs(roots - m)) for m in np.conj(roots)) for roots in complex_root_sets]
    assert min(asymmetric) > 1e-3

    # 5.4: roots of z^2 - 2z + c for c from -3 to 5
    c_values = np.linspace(-3, 5, 41)
    locus = np.array([np.roots([1, -2, cv]).astype(complex) for cv in c_values])
    assert np.allclose(locus.sum(axis=1), 2)

    fig = new_figure("Completing the square works in ℂ; real coefficients mirror the roots",
                     r"$z = \frac{-b \pm \delta}{2a}$ with $\delta^2 = b^2 - 4ac$ for any complex $a, b, c$.   "
                     r"Real coefficients: $p(\bar z) = \overline{p(z)}$, so non-real roots come in pairs $w, \bar w$")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.sqrt(13) * np.cos(t), np.sqrt(13) * np.sin(t), color=MUTED, linestyle="--", linewidth=1,
            label=r"$|z| = \sqrt{13}$")
    for radius, angle, color in [(2.4, cmath.phase(w), BLUE), (1.5, cmath.phase(delta), ORANGE)]:
        s = np.linspace(0, angle, 100)
        ax.plot(radius * np.cos(s), radius * np.sin(s), color=color, linewidth=1.3)
    arrow(ax, 0, w, BLUE, label=r"$w = -5 + 12i$,  $|w| = 13$")
    arrow(ax, 0, delta, ORANGE, label=r"$\delta = 2 + 3i$ (principal)")
    arrow(ax, 0, -delta, AQUA, label=r"$-\delta = -2 - 3i$")
    ax.text(2.4, 1.0, r"$\frac{1}{2}\mathrm{Arg}\, w$", color=ORANGE, fontsize=11)
    argand(ax, (-7, 15), (-5, 14), "Square roots halve the angle")
    legend(ax, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    total = z1 + z2
    midpoint = total / 2
    ax.plot([z1.real, total.real, z2.real], [z1.imag, total.imag, z2.imag], color=MUTED, linestyle=":",
            linewidth=1.2)
    ax.plot([z1.real, z2.real], [z1.imag, z2.imag], color=ORANGE, linestyle="--", linewidth=1.2)
    arrow(ax, 0, z1, ORANGE, label=r"$z_1 = 2 - i$,  $z_2 = 1 + 2i$")
    arrow(ax, 0, z2, ORANGE)
    arrow(ax, 0, total, BLUE, label=r"$z_1 + z_2 = 3 + i = -b/a$")
    ax.scatter([midpoint.real], [midpoint.imag], color=INK, s=30, zorder=5)
    ax.annotate(r"midpoint $-b/2a$", (midpoint.real, midpoint.imag), xytext=(2.6, -1.6), fontsize=10,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    argand(ax, (-1, 5.5), (-2.2, 3), r"$z^2 - (3+i)z + (4+3i)$: not conjugates")
    legend(ax, loc="upper left", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "Solve, then factor", [
        "Ex. 3: Δ = (3+i)² − 4(4+3i) = −8 − 6i,",
        "δ = 1 − 3i  →  z = 2 − i and 1 + 2i.",
        "Vieta: sum 3 + i = −b/a, product 4 + 3i = c/a.",
        "",
        "Ex. 4: p = z⁴ − 2z³ + 6z² − 8z + 8, root 1 + i.",
        "Real coefficients, so 1 − i is a root too:",
        "(z − w)(z − w̄) = z² − 2 Re(w) z + |w|²",
        "`sp.div(p, z**2 - 2*z + 2) -> z**2 + 4`",
        "p = (z² − 2z + 2)(z² + 4), roots 1 ± i, ± 2i.",
        "",
        "Ex. 5: z² − 2z + 5 gives z₁² + z₂² = 4 − 10 = −6.",
    ], fontsize=12, spacing=0.078)

    colors = [BLUE, ORANGE, AQUA]
    for column, (root_sets, title, filled) in enumerate([
            (real_root_sets, "Degree 9, real coefficients (seed 7)", True),
            (complex_root_sets, "Degree 9, complex coefficients", False)]):
        ax = fig.add_subplot(gs[1, column])
        if filled:
            ax.axhspan(-0.04, 0.04, color=GRID, zorder=0)
        for trial, (roots, color) in enumerate(zip(root_sets, colors)):
            if filled:
                for r in roots[roots.imag > 1e-9]:
                    ax.plot([r.real, r.real], [r.imag, -r.imag], color=color, linewidth=0.7, alpha=0.45)
            ax.scatter(roots.real, roots.imag, s=34, color=color, zorder=3, label=f"polynomial {trial + 1}")
        argand(ax, (-2.2, 2.2), (-2.2, 2.2),
               title + (": mirror pairs" if filled else ": no symmetry"))
        legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    scatter = None
    for cv, roots in zip(c_values, locus):
        scatter = ax.scatter(roots.real, roots.imag, c=[cv, cv], cmap=DEEP, vmin=-3, vmax=5, s=18, zorder=3)
    ax.scatter([1], [0], marker="x", color=ORANGE, s=90, linewidths=2.5, zorder=4, label="collision at c = 1")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("c", fontsize=11)
    cbar.outline.set_visible(False)
    argand(ax, (-2, 4), (-2.5, 2.5), r"Roots of $z^2 - 2z + c$, $c = -3 \dots 5$")
    legend(ax, loc="upper left", fontsize=8.5)
    ax.text(1.5, -1.6, r"$1 \pm i\sqrt{c - 1}$", fontsize=10, color=INK2)
    save_02(fig, name="01_quadratics_and_conjugate_roots.png")


# ================================================================ module 02, notebook 02
def winding_number(coefficients, radius, samples=4000):
    theta = np.linspace(0, 2 * np.pi, samples + 1)
    values = np.polyval(coefficients, radius * np.exp(1j * theta))
    return np.angle(values[1:] / values[:-1]).sum() / (2 * np.pi)


def sheet_02_02_fundamental_theorem_of_algebra():
    z = sp.symbols("z")
    multiplicities = sp.roots(z**3 - 3 * z + 2, z)
    assert multiplicities == {1: 2, -2: 1}
    assert str(sp.factor(z**4 + 4)) == "(z**2 - 2*z + 2)*(z**2 + 2*z + 2)"

    # 4.2: degree-7 complex polynomial, seed 3
    rng = np.random.default_rng(3)
    coefficients7 = rng.normal(size=8) + 1j * rng.normal(size=8)
    roots7 = np.roots(coefficients7)
    assert len(roots7) == 7 and np.allclose(coefficients7[0] * np.poly(roots7), coefficients7)

    # 4.3: the test quartic and its winding numbers
    roots = np.array([0.5, 2.5, -1 + 1.5j, -1 - 1.5j])
    coefficients = np.poly(roots).real
    assert np.allclose(coefficients, [1, -1, -1.5, -7.25, 4.0625])
    moduli = np.abs(roots)
    assert np.allclose(np.round(moduli, 4), [0.5, 2.5, 1.8028, 1.8028])
    table = {0.2: 0, 0.4: 0, 1.0: 1, 1.7: 1, 2.0: 3, 2.4: 3, 3.0: 4, 10.0: 4}
    for radius, expected in table.items():
        assert abs(winding_number(coefficients, radius) - expected) < 1e-6
        assert np.sum(moduli < radius) == expected

    # 4.4: Cauchy bound, seed 11
    rng = np.random.default_rng(11)
    lines = []
    for _ in range(4):
        degree = rng.integers(3, 9)
        cs = rng.normal(size=degree + 1) * 3 + 1j * rng.normal(size=degree + 1)
        bound = 1 + np.max(np.abs(cs[1:] / cs[0]))
        largest = np.max(np.abs(np.roots(cs)))
        lines.append(f"{degree} {largest:.3f} {bound:.3f}")
    assert lines == ["3 1.180 1.904", "3 1.117 1.851", "5 1.918 3.379", "6 1.140 1.823"]

    # Exercise 6: the quintic z^5 + 3z + 1
    quintic = [1, 0, 0, 0, 3, 1]
    assert [round(winding_number(quintic, r)) for r in (1.0, 1.3, 2.0)] == [1, 3, 5]

    # 5.3: W(R) as a step function
    root_moduli = np.unique(np.round(moduli, 10))
    radii = np.linspace(0.01, 3.5, 351)
    radii = radii[np.min(np.abs(radii[:, None] - root_moduli[None, :]), axis=1) > 1e-3]
    windings = np.array([round(winding_number(coefficients, r, 3000)) for r in radii])
    assert windings[0] == 0 and windings[-1] == 4 and set(np.diff(windings)) <= {0, 1, 2}

    critical = abs(roots[2])
    theta = np.linspace(0, 2 * np.pi, 4000)
    assert round(winding_number(coefficients, critical - 0.12)) == 1
    assert round(winding_number(coefficients, critical + 0.12)) == 3
    assert round(winding_number(coefficients, 0.3)) == 0 and round(winding_number(coefficients, 1.2)) == 1

    fig = new_figure("Degree n means exactly n roots: the image of a big circle winds n times",
                     r"FTA: every non-constant $p$ has a root, so $p(z) = a_n(z - z_1)\cdots(z - z_n)$.   "
                     r"Winding number of $p(|z| = R)$ around 0 = number of roots in $|z| < R$")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-3.2, 3.5, 400)
    y = np.linspace(-3.2, 3.2, 400)
    X, Y = np.meshgrid(x, y)
    log_modulus = np.log10(np.abs(np.polyval(coefficients, X + 1j * Y)) + 1e-12)
    filled = ax.contourf(X, Y, log_modulus, levels=np.linspace(-1.5, 2.6, 24), cmap=SEQ.reversed(),
                         extend="both")
    ax.contour(X, Y, log_modulus, levels=np.linspace(-1.5, 2.6, 24), colors="white", linewidths=0.3)
    cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2])
    cbar.set_label(r"$\log_{10}|p(z)|$")
    cbar.outline.set_visible(False)
    for r in root_moduli:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="white", linestyle="--", linewidth=1)
    ax.scatter(roots.real, roots.imag, color=ORANGE, s=45, zorder=4, edgecolor="white", label="roots")
    argand(ax, (-3.2, 3.5), (-3.2, 3.2), r"$|p|$ has no minima except the roots", show_grid=False)
    legend(ax, loc="upper left", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.step(radii, windings, where="post", color=BLUE, linewidth=2.4, label="winding number W(R)")
    for r in root_moduli:
        ax.axvline(r, color=ORANGE, linestyle="--", linewidth=1)
    ax.plot([], [], color=ORANGE, linestyle="--", label="moduli of the roots")
    for r, label, height in [(0.5, "+1", 0.5), (math.sqrt(13) / 2, "+2 (pair)", 2.0), (2.5, "+1", 3.5)]:
        ax.text(r + 0.06, height, label, color=ORANGE, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_xlabel("radius R")
    ax.set_ylabel("W(R)")
    ax.grid(True)
    ax.set_title("W climbs from 0 to 4 only at root moduli")
    legend(ax, loc="upper left", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "The test quartic of §4.3", [
        "p = (z − ½)(z − 5/2)(z² + 2z + 13/4)",
        "  = z⁴ − z³ − 1.5z² − 7.25z + 4.0625",
        "root moduli 0.5, 1.8028 (pair), 2.5",
        "",
        "`R:  0.2 0.4 1.0 1.7 2.0 2.4 3.0 10`",
        "`W:   0   0   1   1   3   3   4   4`",
        "",
        "Small R: p ≈ p(0) ≠ 0, so W = 0.",
        "Large R: p ≈ aₙzⁿ, so W = n.",
        "W can only change where the image curve",
        "passes through 0: at a root.",
    ], fontsize=12, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    for radius, color, wn in [(0.3, AQUA, 0), (1.2, BLUE, 1)]:
        image = np.polyval(coefficients, radius * np.exp(1j * theta))
        ax.plot(image.real, image.imag, color=color, linewidth=1.8, label=f"R = {radius}: W = {wn}")
        k = 150
        arrow(ax, image[k], image[k + 6], color, width=1.8)
    ax.scatter([0], [0], marker="x", color=ORANGE, s=70, linewidths=2.5, zorder=5)
    ax.scatter([coefficients[-1]], [0], color=INK, s=18, zorder=5)
    ax.text(coefficients[-1] + 0.3, -1.6, "p(0) = 4.0625", fontsize=9.5, color=INK2)
    argand(ax, (-8, 16), (-12.5, 12.5), r"Images of $|z| = 0.3$ and $|z| = 1.2$")
    legend(ax, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    for radius, color, label in [(critical - 0.12, BLUE, f"R = {critical - 0.12:.3f}: W = 1"),
                                 (critical + 0.12, ORANGE, f"R = {critical + 0.12:.3f}: W = 3")]:
        image = np.polyval(coefficients, radius * np.exp(1j * theta))
        ax.plot(image.real, image.imag, color=color, linewidth=1.8, label=label)
    ax.scatter([0], [0], marker="x", color=INK, s=70, linewidths=2.5, zorder=5)
    argand(ax, (-6, 6), (-6, 6), r"Sweeping over the pair $-1 \pm 1.5i$")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.sqrt(2) * np.cos(theta), np.sqrt(2) * np.sin(theta), color=MUTED, linestyle="--", linewidth=1,
            label=r"$|z| = \sqrt{2}$")
    for factor, color, label in [([1, -2, 2], BLUE, r"$z^2 - 2z + 2$"), ([1, 2, 2], ORANGE, r"$z^2 + 2z + 2$")]:
        pair = np.roots(factor)
        ax.plot(pair.real, pair.imag, color=color, linestyle=":", linewidth=1.5)
        ax.scatter(pair.real, pair.imag, s=80, color=color, zorder=4, label=label)
        for r in pair:
            sign_re = "−" if r.real < 0 else ""
            sign_im = "−" if r.imag < 0 else "+"
            ax.text(r.real * 1.38, r.imag * 1.33, f"{sign_re}1 {sign_im} i", ha="center", va="center", fontsize=10)
    argand(ax, (-2.3, 2.3), (-3.1, 2.3), r"$z^4 + 4$: two real quadratic factors")
    legend(ax, loc="lower center", fontsize=8.5, ncol=3, handletextpad=0.3, columnspacing=0.8)
    save_02(fig, name="02_fundamental_theorem_of_algebra.png")


# ================================================================ module 02, notebook 03
def newton_basins(coefficients, x_range, y_range, size, iterations=40, tolerance=1e-6):
    derivative = np.polyder(coefficients)
    roots = np.roots(coefficients)
    roots = roots[np.argsort(np.mod(np.angle(roots), 2 * np.pi))]
    x = np.linspace(*x_range, size)
    y = np.linspace(*y_range, size)
    Z = x[None, :] + 1j * y[:, None]
    steps_needed = np.full(Z.shape, iterations)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for k in range(iterations):
            Z = Z - np.polyval(coefficients, Z) / np.polyval(derivative, Z)
            close = np.min(np.abs(Z[..., None] - roots), axis=-1) < tolerance
            steps_needed = np.where(close & (steps_needed == iterations), k + 1, steps_needed)
        distances = np.nan_to_num(np.abs(Z[..., None] - roots), nan=np.inf)
    label = np.where(np.min(distances, axis=-1) < tolerance, np.argmin(distances, axis=-1), -1)
    return roots, label, steps_needed


def basin_rgb(label, palette, steps=None):
    image = np.tile(np.array(to_rgb(INK)), label.shape + (1,))
    shade = np.ones(label.shape) if steps is None else 1.0 - 0.6 * np.clip(steps / 25, 0, 1)
    for index, color in enumerate(palette):
        mask = label == index
        image[mask] = np.array(to_rgb(color)) * shade[mask, None]
    return image


def sheet_02_03_newtons_method_in_the_complex_plane():
    roots_of_unity = np.exp(2j * np.pi * np.arange(3) / 3)

    # 4.1: Newton from -1 + i for z^3 - 1
    z = -1 + 1j
    path = [z]
    for _ in range(50):
        step = (z**3 - 1) / (3 * z**2)
        z = z - step
        path.append(z)
        if abs(step) < 1e-14:
            break
    path = np.array(path)
    errors_simple = np.array([np.min(np.abs(roots_of_unity - p)) for p in path])
    assert " ".join(f"{e:.2e}" for e in errors_simple[:6]) == "5.18e-01 1.70e-01 2.64e-02 7.13e-04 5.08e-07 2.58e-13"
    assert np.isclose(path[-1], roots_of_unity[1])

    # 4.2: a real start for z^2 + 1 never converges
    z = complex(0.5)
    for _ in range(60):
        z = z - (z**2 + 1) / (2 * z)
    assert f"{z:.6f}" == "-0.051650+0.000000j"

    # 4.3: double root, standard and modified Newton
    p = lambda z: (z - 1) ** 2 * (z + 1)
    dp = lambda z: 2 * (z - 1) * (z + 1) + (z - 1) ** 2
    error_runs = {}
    for multiplier in (1, 2):
        z = 2 + 0.5j
        errors = []
        for _ in range(8):
            if abs(p(z)) < 1e-300:
                break
            z = z - multiplier * p(z) / dp(z)
            errors.append(abs(z - 1))
        error_runs[multiplier] = np.array(errors)
    assert " ".join(f"{e:.1e}" for e in error_runs[1]) == "6.4e-01 3.6e-01 1.9e-01 9.8e-02 5.0e-02 2.5e-02 1.3e-02 6.4e-03"
    assert " ".join(f"{e:.1e}" for e in error_runs[2]) == "1.7e-01 6.9e-03 1.2e-05 3.6e-11 3.1e-22 0.0e+00"

    # 4.4: basins on [-2, 2]^2 at 400 x 400
    roots, label400, steps400 = newton_basins([1, 0, 0, -1], (-2, 2), (-2, 2), size=400)
    fractions = [f"{np.mean(label400 == i):.4f}" for i in range(3)]
    assert fractions == ["0.3529", "0.3236", "0.3236"]
    assert np.mean(label400 == -1) == 1.25e-05 and np.median(steps400) == 6.0

    # 5.4: two starts 1e-12 apart with different destinations
    def destination(z0, iterations=80):
        zz = np.asarray(z0, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(iterations):
                zz = zz - (zz**3 - 1) / (3 * zz**2)
            distances = np.nan_to_num(np.abs(zz[..., None] - roots_of_unity), nan=np.inf)
        return np.where(np.min(distances, axis=-1) < 1e-8, np.argmin(distances, axis=-1), -1)

    segment = np.linspace(-1.2, 0.2, 4000) + 0.15j
    labels = destination(segment)
    change = np.nonzero(labels[1:] != labels[:-1])[0][0]
    left, right = segment[change], segment[change + 1]
    label_left = destination(left)
    for _ in range(60):
        middle = (left + right) / 2
        if destination(middle) == label_left:
            left = middle
        else:
            right = middle
        if abs(right - left) < 1e-12:
            break
    assert f"{abs(right - left):.1e}" == "6.5e-13"
    assert f"{left:.12f}" == "-1.180116196550+0.150000000000j"
    assert int(destination(left)) == 1 and int(destination(right)) == 2

    # 5.5: z^3 - 2z + 2 with its attracting 2-cycle
    cycle_coefficients = [1, 0, -2, 2]
    cycle_roots = np.roots(cycle_coefficients)
    xs = np.linspace(-2.5, 2.5, 600)
    Z = xs[None, :] + 1j * xs[:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(60):
            Z = Z - np.polyval(cycle_coefficients, Z) / np.polyval(np.polyder(cycle_coefficients), Z)
        distances = np.nan_to_num(np.abs(Z[..., None] - cycle_roots), nan=np.inf)
    cycle_label = np.where(np.min(distances, axis=-1) < 1e-6, np.argmin(distances, axis=-1), -1)
    trapped = float(np.mean(cycle_label == -1))
    assert round(trapped, 4) == 0.0075
    N = lambda z: z - (z**3 - 2 * z + 2) / (3 * z**2 - 2)
    assert N(0) == 1 and N(1) == 0

    palette = [ORANGE, BLUE, AQUA]
    fig = new_figure("Newton's method finds complex roots, but the basins have fractal borders",
                     r"$z_{k+1} = z_k - p(z_k)/p'(z_k)$: error squared near a simple root, "
                     r"times $1 - 1/m$ near a root of multiplicity $m$.   Colour = the root a start converges to")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.42, left=0.065)

    ax = fig.add_subplot(gs[0, 0])
    roots_plot, label_plot, steps_plot = newton_basins([1, 0, 0, -1], (-2, 2), (-2, 2), size=600)
    ax.imshow(basin_rgb(label_plot, palette, steps_plot), extent=(-2, 2, -2, 2), origin="lower",
              interpolation="nearest")
    ax.scatter(roots_plot.real, roots_plot.imag, color="white", edgecolor=INK, s=60, zorder=3)
    ax.add_patch(Rectangle((-0.9, -0.3), 0.6, 0.6, fill=False, edgecolor="white", linewidth=1.5))
    argand(ax, (-2, 2), (-2, 2), r"Basins of $z^3 - 1$ (darker = slower)", show_grid=False)
    plain_image_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    starts = [1.8 + 1.2j, -1 + 1j, -1.5 - 0.3j, 0.3 - 1.5j, -0.6 + 0.05j, 0.25 + 0.45j, -0.2 - 0.9j]
    for z0 in starts:
        z = complex(z0)
        orbit = [z]
        for _ in range(25):
            z = z - (z**3 - 1) / (3 * z**2)
            orbit.append(z)
        orbit = np.array(orbit)
        dest = np.argmin(np.abs(roots_of_unity - orbit[-1]))
        color = palette[[0, 1, 2][dest]]
        ax.plot(orbit.real, orbit.imag, "o-", color=color, markersize=3, linewidth=1)
        ax.scatter([z0.real], [z0.imag], color=color, s=50, marker="s", zorder=3)
    ax.scatter(roots_of_unity.real, roots_of_unity.imag, color=INK, s=140, marker="*", zorder=4,
               label=r"roots $1, \omega, \bar\omega$")
    argand(ax, (-2.2, 2.2), (-2.2, 2.2), "Orbits: squares = starts")
    legend(ax, loc="lower left", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "What the notebook measures", [
        "From −1 + i, distance to ω:",
        "`0.52  0.17  0.026  7e-4  5e-7  3e-13`",
        "digits double: N'(r) = 0 at a simple root.",
        "",
        "Basins of z³ − 1 on [−2, 2]², 400 × 400:",
        "`0.3529  0.3236  0.3236`",
        "of the square each; median 6 steps.",
        "",
        "Starts 6.5e-13 apart near −1.1801 + 0.15i",
        "converge to ω and to ω̄: sensitive dependence.",
        "",
        "Real starts stay real: z₀ = 0.5 never finds ± i.",
    ], fontsize=12, spacing=0.074)

    ax = fig.add_subplot(gs[1, 0])
    k_simple = np.arange(1, 6)
    ax.semilogy(k_simple, errors_simple[1:6], "o-", color=AQUA, linewidth=2,
                label=r"$z^3 - 1$, simple root")
    ax.semilogy(np.arange(1, 9), error_runs[1], "o-", color=ORANGE, linewidth=2,
                label="double root, standard: ratio → 1/2")
    modified = error_runs[2][error_runs[2] > 0]
    ax.semilogy(np.arange(1, len(modified) + 1), modified, "o-", color=BLUE, linewidth=2,
                label="double root, modified (m = 2)")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"error $|z_k - r|$")
    ax.set_ylim(1e-23, 1e6)
    ax.grid(True, which="major")
    ax.set_title("Quadratic vs linear convergence")
    legend(ax, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    _, zoom_label, _ = newton_basins([1, 0, 0, -1], (-0.9, -0.3), (-0.3, 0.3), size=500, iterations=60)
    ax.imshow(basin_rgb(zoom_label, palette), extent=(-0.9, -0.3, -0.3, 0.3), origin="lower",
              interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Zoom (white box): all three colours meet")
    plain_image_axes(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(basin_rgb(cycle_label, palette), extent=(-2.5, 2.5, -2.5, 2.5), origin="lower",
              interpolation="nearest")
    ax.scatter(cycle_roots.real, cycle_roots.imag, color="white", edgecolor=INK, s=55, zorder=3, label="roots")
    ax.scatter([0, 1], [0, 0], color="#f5d000", edgecolor=INK, s=50, marker="D", zorder=3,
               label=r"2-cycle $0 \leftrightarrow 1$")
    argand(ax, (-2.5, 2.5), (-2.5, 2.5), rf"$z^3 - 2z + 2$: {trapped:.2%} trapped (black)", show_grid=False)
    plain_image_axes(ax)
    legend(ax, loc="lower left", fontsize=8.5)
    save_02(fig, name="03_newtons_method_in_the_complex_plane.png")


# ================================================================ module 02, notebook 04
def sheet_02_04_partial_fractions_over_c():
    numerator = [1, 3]
    denominator = np.polymul([1, -1], [1, 0, 4])
    poles = np.roots(denominator)
    coefficients = np.polyval(numerator, poles) / np.polyval(np.polyder(denominator), poles)
    by_pole = {complex(np.round(pk, 9)): A for pk, A in zip(poles, coefficients)}
    assert np.isclose(by_pole[1 + 0j], 0.8)
    assert np.isclose(by_pole[2j], -0.4 - 0.05j) and np.isclose(by_pole[-2j], -0.4 + 0.05j)
    A_2i = (3 + 2j) / ((2j - 1) * 4j)
    assert np.isclose(A_2i, -0.4 - 0.05j)

    rng = np.random.default_rng(0)
    test_points = rng.normal(size=5) + 1j * rng.normal(size=5)
    direct = np.polyval(numerator, test_points) / np.polyval(denominator, test_points)
    pieces = sum(A / (test_points - zk) for zk, A in zip(poles, coefficients))
    assert np.max(np.abs(direct - pieces)) < 1e-14

    # 4.4: recombine the conjugate pair into a real term
    B = 2 * A_2i.real
    C = -2 * (A_2i * np.conj(2j)).real
    assert np.isclose(B, -0.8) and np.isclose(C, 0.2)

    x = sp.symbols("x")
    assert sp.simplify(sp.apart((x + 3) / ((x - 1) * (x**2 + 4)), x)
                       - (-(4 * x - 1) / (5 * (x**2 + 4)) + sp.Rational(4, 5) / (x - 1))) == 0

    # 4.5: integral of 1/(x^3 + 1) on [0, 1] through complex logarithms
    cubic_poles = np.roots([1, 0, 0, 1])
    cubic_coefficients = 1 / (3 * cubic_poles**2)
    via_logs = np.sum(cubic_coefficients * (np.log(1 - cubic_poles) - np.log(0 - cubic_poles)))
    exact = np.log(2) / 3 + np.pi / (3 * np.sqrt(3))
    assert abs(via_logs.imag) < 1e-12 and np.isclose(via_logs.real, exact)
    assert f"{exact:.15f}" == "0.835648848264721"

    fig = new_figure("Over ℂ every rational function splits into simple poles",
                     r"$\frac{P(z)}{Q(z)} = \sum_k \frac{A_k}{z - z_k}$ with the cover-up formula "
                     r"$A_k = \frac{P(z_k)}{Q'(z_k)}$ (the residue);   conjugate pairs recombine into "
                     r"$\frac{Bx + C}{x^2 + px + q}$")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.42, right=0.955)

    ax = fig.add_subplot(gs[0, 0])
    scale = 2.0
    ax.scatter(poles.real, poles.imag, marker="x", color=INK, s=110, linewidths=2.2, zorder=4, label="poles")
    offsets = {1 + 0j: (0.15, 0.3), 2j: (0.2, 0.35), -2j: (0.2, -0.75)}
    for pk, A in zip(poles, coefficients):
        arrow(ax, pk, pk + scale * A, ORANGE, width=2.2)
        dx, dy = offsets[complex(np.round(pk, 9))]
        ax.text(pk.real + dx, pk.imag + dy, f"A = {A.real:.2f} {A.imag:+.2f}i", color=ORANGE, fontsize=10)
    ax.plot([], [], color=ORANGE, linewidth=2.2, label=r"coefficient $A_k$ (× 2)")
    argand(ax, (-2.5, 3.2), (-3, 3), r"$R = \frac{z + 3}{(z - 1)(z^2 + 4)}$: $A_{\bar w} = \overline{A_w}$")
    legend(ax, loc="center left", fontsize=8.5)

    xs = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(xs, xs)
    Z = X + 1j * Y
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.polyval(numerator, Z) / np.polyval(denominator, Z)
        remainder = R - A_2i / (Z - 2j)
    for column, (values, title) in enumerate([(R, r"$\log_{10}|R(z)|$: three spikes"),
                                              (remainder, r"$\log_{10}|R - A_{2i}/(z - 2i)|$: spike at $2i$ gone")]):
        ax = fig.add_subplot(gs[0, 1 + column])
        image = ax.imshow(np.log10(np.abs(values) + 1e-12), extent=(-3, 3, -3, 3), origin="lower",
                          cmap=SEQ, vmin=-1.5, vmax=1.5)
        ax.scatter([1, 0, 0], [0, 2, -2], marker="x", color=ORANGE, s=50, linewidths=2)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)
        argand(ax, (-3, 3), (-3, 3), title, show_grid=False)
        plain_image_axes(ax)

    ax = fig.add_subplot(gs[1, 0])
    xr = np.linspace(-6, 6, 2000)
    xr = np.where(np.abs(xr - 1) < 0.02, np.nan, xr)
    ax.plot(xr, (xr + 3) / ((xr - 1) * (xr**2 + 4)), color=INK, linewidth=2.6, label=r"$R(x)$")
    ax.plot(xr, 0.8 / (xr - 1), color=BLUE, linestyle="--", linewidth=1.6, label=r"$\frac{4/5}{x - 1}$")
    ax.plot(xr, (-0.8 * xr + 0.2) / (xr**2 + 4), color=ORANGE, linestyle="--", linewidth=1.6,
            label=r"$\frac{-4x/5 + 1/5}{x^2 + 4}$ (pair $\pm 2i$)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(1, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title("On the real line: the real pieces add up")
    legend(ax, loc="lower left", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    xs01 = np.linspace(0, 1, 50)
    ax.plot([-2.5, 0], [0, 0], color=ORANGE, linewidth=4, alpha=0.6, label="branch cut of Log")
    for pk, color in zip(cubic_poles, [BLUE, AQUA, INK2]):
        trace = xs01 - pk
        ax.plot(trace.real, trace.imag, color=color, linewidth=3,
                label=rf"$x - z_k$, $z_k = {pk.real:.2f} {pk.imag:+.2f}i$")
        arrow(ax, trace[-8], trace[-1], color, width=2)
    argand(ax, (-2.5, 2.5), (-2.9, 1.5), r"$\int_0^1 \frac{dx}{x^3+1}$: paths $x - z_k$ avoid the cut")
    legend(ax, loc="lower left", fontsize=8.5)

    panel(fig.add_subplot(gs[1, 2]), "Example 2, both ways", [
        "Cover up (z − 2i), evaluate at 2i:",
        "A₂ᵢ = (3 + 2i)/((2i − 1)·4i) = −2/5 − i/20",
        "A₁ = 4/5,  A₋₂ᵢ = conj(A₂ᵢ) = −2/5 + i/20",
        "",
        "B = 2 Re A = −4/5,  C = −2 Re(A w̄) = 1/5:",
        "R(x) = (4/5)/(x − 1) + (−4x/5 + 1/5)/(x² + 4)",
        "",
        "∫₀¹ dx/(x³ + 1) = Σ Aₖ [Log(1 − zₖ) − Log(−zₖ)]",
        "`0.835648848264722 + 0j`",
        "= ln 2 / 3 + π/(3√3)",
    ], fontsize=12, spacing=0.08)
    save_02(fig, name="04_partial_fractions_over_c.png")


# ================================================================ module 03, notebook 01
def sheet_03_01_complex_functions_as_maps():
    assert (1 + 1j) ** 2 == 2j and np.isclose(1 / (1 + 1j), 0.5 - 0.5j)
    zz = 1 + 1j
    assert np.isclose(zz**2 + zz, 1 + 3j)
    assert np.isclose((1 + 0.5j) ** 2, 0.75 + 1j)
    yv = np.linspace(-5, 5, 1001)
    line = 1.0 + 1j * yv
    w = line**2
    assert np.max(np.abs(w.real - (1 - w.imag**2 / 4))) < 1e-12
    w = 1 / line
    assert np.max(np.abs(np.abs(w - 0.5) - 0.5)) < 1e-12
    corners = np.array([0, 1, 1 + 1j, 1j])
    assert np.allclose((1 + 1j) * corners + 2, [2, 3 + 1j, 2 + 2j, 1 + 1j])

    def f(z):
        return (z - 1) * (z + 1) ** 2 / (z - 1j)

    assert np.allclose(np.poly([1, -1, -1]), [1, 1, -1, -1])
    assert [f"{abs(f(z0 + 1e-6)):.3e}" for z0 in (1, -1, 1j)] == ["2.828e-06", "1.414e-12", "2.828e+06"]

    fig = new_figure("A complex function is a map from the z-plane to the w-plane",
                     r"$f(x + iy) = u(x, y) + i\,v(x, y)$:   $z^2 = (x^2 - y^2) + i\,2xy$ squares $|z|$ and doubles "
                     r"$\arg z$;   $1/z$ inverts $|z|$ and negates $\arg z$")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.42)

    grid_values = np.linspace(-2, 2, 9)
    grid_values = grid_values[np.abs(grid_values) > 1e-9]
    t = np.linspace(-2, 2, 400)
    marker = 1 + 0.5j

    ax = fig.add_subplot(gs[0, 0])
    for a in grid_values:
        ax.plot(a + 0 * t, t, color=BLUE, linewidth=1)
        ax.plot(t, a + 0 * t, color=ORANGE, linewidth=1)
    ax.plot(marker.real, marker.imag, "o", color=INK, markersize=7, zorder=5)
    ax.text(marker.real + 0.1, marker.imag + 0.12, "1 + 0.5i", fontsize=10)
    ax.plot([], [], color=BLUE, label="x = a")
    ax.plot([], [], color=ORANGE, label="y = b")
    argand(ax, (-2.3, 2.3), (-2.3, 2.3), "z-plane: a square grid")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    for a in grid_values:
        ax.plot(((a + 1j * t) ** 2).real, ((a + 1j * t) ** 2).imag, color=BLUE, linewidth=1)
        ax.plot(((t + 1j * a) ** 2).real, ((t + 1j * a) ** 2).imag, color=ORANGE, linewidth=1)
    image = marker**2
    ax.plot(image.real, image.imag, "o", color=INK, markersize=7, zorder=5)
    ax.text(image.real + 0.35, image.imag - 0.2, "0.75 + i", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    argand(ax, (-4.5, 4.5), (-8.2, 8.2), r"$w = z^2$: confocal parabolas")
    ax.text(-4.3, -7.6, r"$x = a \mapsto u = a^2 - v^2/4a^2$", fontsize=9.5, color=BLUE, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))

    ax = fig.add_subplot(gs[0, 2])
    tt = np.linspace(-40, 40, 8001)
    for a in [-2, -1, -0.5, 0.5, 1, 2]:
        ax.plot((1 / (a + 1j * tt)).real, (1 / (a + 1j * tt)).imag, color=BLUE, linewidth=1)
        ax.plot((1 / (tt + 1j * a)).real, (1 / (tt + 1j * a)).imag, color=ORANGE, linewidth=1)
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 300))
    ax.plot(unit.real, unit.imag, color=INK2, linestyle="--", linewidth=1, label="unit circle")
    ax.plot(0.5, -0.5, "o", color=INK, markersize=7, zorder=5)
    ax.text(0.45, -1.3, r"$1/(1+i) = (1-i)/2$", fontsize=9.5, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    argand(ax, (-2.5, 2.5), (-2.5, 2.5), r"$w = 1/z$: lines → circles through 0")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    show_domain_colouring(ax, lambda z: z, (-2, 2), (-2, 2), "Domain colouring key: f(z) = z", resolution=400)
    ax.plot(unit.real, unit.imag, color=INK, linestyle=":", linewidth=1)
    for text, point in [("1", 1.35), ("i", 1.35j), ("−1", -1.35), ("−i", -1.35j)]:
        ax.text(point.real, point.imag, text, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.85))

    ax = fig.add_subplot(gs[1, 1])
    show_domain_colouring(ax, f, (-2.5, 2.5), (-2.5, 2.5), r"$f = (z-1)(z+1)^2/(z-i)$", resolution=450)
    for text, point, offset in [("simple zero", 1, (0.25, -0.9)), ("double zero", -1, (-0.6, -0.95)),
                                ("simple pole", 1j, (0.3, 0.6))]:
        ax.annotate(text, (point.real, point.imag), xytext=(point.real + offset[0], point.imag + offset[1]),
                    fontsize=9.5, bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.88),
                    arrowprops=dict(arrowstyle="->", color=INK))

    panel(fig.add_subplot(gs[1, 2]), "Reading a domain colouring", [
        "hue = arg f,  dark = small |f|,  white = large |f|",
        "",
        "zero of order n: black dot, colours cycle",
        "n times counter-clockwise; pole: white dot,",
        "colours cycle the other way.",
        "",
        "Here: one cycle at 1, two at −1,",
        "one reversed at i; far away f ≈ z²,",
        "so 1 + 2 − 1 = 2 cycles around the edge.",
        "",
        "`|f| next to -1: 1.414e-12 (double zero)`",
    ], fontsize=12, spacing=0.078)
    save_03(fig, name="01_complex_functions_as_maps.png")


# ================================================================ module 03, notebook 02
def sheet_03_02_the_complex_exponential():
    checks = {1 + 1j * math.pi / 2: "0.000000+2.718282j", 2 - 3j: "-7.315110-1.042744j",
              -10 + 7j: "0.000034+0.000030j"}
    for z, text in checks.items():
        value = math.exp(z.real) * complex(math.cos(z.imag), math.sin(z.imag))
        assert abs(value - cmath.exp(z)) < 1e-12 and f"{cmath.exp(z):.6f}" == text
    z = 0.3 + 0.8j
    assert all(f"{cmath.exp(z + 2j * np.pi * k):.12f}" == "0.940455687910+0.968329437469j" for k in range(-2, 3))
    for w, real_part in [(-1, 0.0), (1 + 1j, 0.3466), (-2, 0.6931)]:
        principal = cmath.log(w)
        assert round(principal.real, 4) == real_part
        assert all(abs(cmath.exp(principal + 2j * math.pi * k) - w) < 1e-12 for k in range(-2, 3))
    zs = 1j * math.pi
    partial = 0
    errors = {}
    for n in range(21):
        partial += zs**n / math.factorial(n)
        errors[n] = abs(partial - cmath.exp(zs))
    assert f"{errors[10]:.2e}" == "7.16e-03" and f"{errors[20]:.2e}" == "5.34e-10"

    fig = new_figure("The real part of z sets the size of eᶻ, the imaginary part its angle",
                     r"$e^{x+iy} = e^x(\cos y + i \sin y)$:   $|e^z| = e^x$, $\arg e^z = y$.   "
                     r"Never zero, and periodic: $e^{z + 2\pi i} = e^z$, so $e^z = w$ has the solutions "
                     r"$\ln|w| + i(\arg w + 2\pi k)$")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.42)

    x_values = np.linspace(-1, 1, 5)
    y_values = np.linspace(-np.pi, np.pi, 13)
    tv = np.linspace(-np.pi, np.pi, 400)
    th = np.linspace(-1, 1, 200)
    marker = 1 + 1j * np.pi / 2

    ax = fig.add_subplot(gs[0, 0])
    for a in x_values:
        ax.plot(a + 0 * tv, tv, color=BLUE, linewidth=1.2)
    for b in y_values:
        ax.plot(th, b + 0 * th, color=ORANGE, linewidth=1.2)
    ax.plot(marker.real, marker.imag, "o", color=INK, markersize=7, zorder=5)
    ax.annotate(r"$1 + i\pi/2$", (marker.real, marker.imag), xytext=(-0.5, 2.45), fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=INK))
    argand(ax, (-1.6, 1.6), (-3.5, 3.5), r"z-plane: $|x| \leq 1$, $|y| \leq \pi$")
    pi_ticks(ax, "y", [-1, -0.5, 0, 0.5, 1], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax = fig.add_subplot(gs[0, 1])
    for a in x_values:
        image = np.exp(a + 1j * tv)
        ax.plot(image.real, image.imag, color=BLUE, linewidth=1.2)
    for b in y_values:
        image = np.exp(th + 1j * b)
        ax.plot(image.real, image.imag, color=ORANGE, linewidth=1.2)
    ax.plot(0, np.e, "o", color=INK, markersize=7, zorder=5)
    ax.annotate(r"$e^{1 + i\pi/2} = e \cdot i$", (0, np.e), xytext=(1.2, 2.75), fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=INK))
    argand(ax, (-3, 3), (-3, 3), r"$w = e^z$: lines → circles $|w| = e^a$ and rays")

    panel(fig.add_subplot(gs[0, 2]), "Worked numbers", [
        "e^(2 − 3i) = e²(cos 3 − i sin 3)",
        "`         = -7.315110-1.042744j`",
        "|e^(−10 + 7i)| = e⁻¹⁰ ≈ 4.5e-5, never 0",
        "",
        "e^z = −1  ⇔  z = iπ(2k + 1)",
        "e^z = 1 + i  ⇔  z = ½ ln 2 + i(π/4 + 2πk)",
        "`               = 0.3466 + 0.7854i + 2πik`",
        "",
        "Series Σ (iπ)ⁿ/n! → −1:",
        "error 7.16e-03 at N = 10, 5.34e-10 at N = 20",
    ], fontsize=12, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    show_domain_colouring(ax, np.exp, (-3, 3), (-2 * np.pi, 2 * np.pi), r"$e^z$: bands of period $2\pi$, no zeros",
                          resolution=300)
    for k in (-2, -1, 0, 1, 2):
        ax.axhline(k * np.pi, color="white", linestyle=":", linewidth=0.8)
    pi_ticks(ax, "y", [-2, -1, 0, 1, 2], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])

    ax = fig.add_subplot(gs[1, 1])

    def rectangle_boundary(x0, x1, y0, y1, n=100):
        return np.concatenate([np.linspace(x0, x1, n) + 1j * y0, x1 + 1j * np.linspace(y0, y1, n),
                               np.linspace(x1, x0, n) + 1j * y1, x0 + 1j * np.linspace(y1, y0, n)])

    pieces = [(-2 * np.pi, BLUE, 6.0, r"$y \in [-2\pi, -3\pi/2]$"), (0, AQUA, 3.5, r"$y \in [0, \pi/2]$"),
              (2 * np.pi, INK2, 1.2, r"$y \in [2\pi, 5\pi/2]$"), (np.pi, ORANGE, 2.0, r"$y \in [\pi, 3\pi/2]$")]
    images = []
    for shift, color, width, label in pieces:
        image = np.exp(rectangle_boundary(0, 1, shift, shift + np.pi / 2))
        images.append(image)
        ax.fill(image.real, image.imag, color=color, alpha=0.18)
        ax.plot(image.real, image.imag, color=color, linewidth=width, label=label)
    assert np.allclose(images[0], images[1]) and np.allclose(images[1], images[2])
    argand(ax, (-3.2, 3.2), (-3.2, 3.2), r"Rectangles $0 \leq x \leq 1$ shifted by $2\pi$: same image")
    legend(ax, loc="lower right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    terms = np.array([zs**n / math.factorial(n) for n in range(16)])
    partial_sums = np.concatenate([[0], np.cumsum(terms)])
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 300))
    ax.plot(unit.real, unit.imag, color=GRID, linewidth=1.2)
    ax.plot(partial_sums.real, partial_sums.imag, "o-", color=BLUE, markersize=4, linewidth=1.4,
            label=r"partial sums $S_N(i\pi)$")
    for n in range(1, 5):
        ax.annotate(f"$S_{n - 1}$", (partial_sums[n].real, partial_sums[n].imag), xytext=(5, 5),
                    textcoords="offset points", fontsize=10)
    ax.plot(-1, 0, "*", color=ORANGE, markersize=16, zorder=5, label=r"$e^{i\pi} = -1$")
    argand(ax, (-5, 4.5), (-3, 4), r"Each term turns a further $90°$")
    legend(ax, loc="lower right", fontsize=8.5)
    save_03(fig, name="02_the_complex_exponential.png")


# ================================================================ module 03, notebook 03
def sheet_03_03_the_complex_logarithm_and_branch_cuts():
    assert f"{cmath.log(1 - 1j):.6f}" == "0.346574-0.785398j"
    assert f"{cmath.log(-3):.6f}" == "1.098612+3.141593j"
    jumps = [f"{cmath.log(complex(-4, e)).imag - cmath.log(complex(-4, -e)).imag:.6f}" for e in (1e-1, 1e-3, 1e-6)]
    assert jumps == ["6.233196", "6.282685", "6.283185"]
    above = cmath.log(complex(-4, 0.001))
    assert f"{above.real:.4f}" == "1.3863" and f"{above.imag:.4f}" == "3.1413"
    assert cmath.log(complex(-1, 0.0)).imag == math.pi and cmath.log(complex(-1, -0.0)).imag == -math.pi
    assert cmath.log(-1 - 0j).imag == math.pi

    rng = np.random.default_rng(seed=0)
    n = 100_000
    z = rng.uniform(0.2, 3, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    w = rng.uniform(0.2, 3, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    k = (np.log(z * w) - np.log(z) - np.log(w)) / (2j * np.pi)
    assert np.allclose(k.imag, 0) and np.allclose(k.real, np.round(k.real))
    values, counts = np.unique(np.round(k.real).astype(int), return_counts=True)
    shares = counts / n
    assert values.tolist() == [-1, 0, 1] and [f"{s:.3f}" for s in shares] == ["0.126", "0.748", "0.126"]
    z0 = -1 + 1j
    assert np.isclose(np.log(z0 * z0), np.log(2) - 0.5j * np.pi)
    assert np.isclose(np.log(z0) + np.log(z0), np.log(2) + 1.5j * np.pi)

    fig = new_figure("Every z ≠ 0 has infinitely many logarithms; Log picks one and must jump",
                     r"$\log z = \ln|z| + i(\mathrm{Arg}\, z + 2\pi k)$.   Principal value "
                     r"$\mathrm{Log}\, z = \ln|z| + i\,\mathrm{Arg}\, z$, $-\pi < \mathrm{Arg}\, z \leq \pi$, "
                     r"jumps by $2\pi i$ across the cut $(-\infty, 0]$")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(xs, xs)
    image = ax.imshow(np.angle(X + 1j * Y), origin="lower", extent=[-2, 2, -2, 2], cmap=SEQ,
                      vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    cbar.set_label(r"Arg z = Im Log z")
    cbar.outline.set_visible(False)
    ax.plot([-2, 0], [0, 0], color=ORANGE, linewidth=3.5, label="branch cut")
    argand(ax, (-2, 2), (-2, 2), r"Arg z jumps by $2\pi$ across the cut", show_grid=False)
    plain_image_axes(ax)
    legend(ax, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 4 * np.pi, 1000)
    circle = np.exp(1j * t)
    ax.plot(t, np.angle(circle), color=BLUE, linewidth=2, label="Im Log z (principal): saw-tooth")
    ax.plot(t, np.unwrap(np.angle(circle)), "--", color=ORANGE, linewidth=2,
            label=r"continuous choice: $+2\pi$ per lap")
    for crossing in [np.pi, 3 * np.pi]:
        ax.axvline(crossing, color=MUTED, linestyle=":", linewidth=1)
    pi_ticks(ax, "x", range(5), ["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    pi_ticks(ax, "y", [-1, 0, 1, 2, 3, 4], [r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])
    ax.set_xlabel(r"$t$  ($z = e^{it}$, twice around the unit circle)")
    ax.set_ylabel("Im of the logarithm")
    ax.grid(True)
    ax.set_title("No continuous log around 0")
    legend(ax, loc="upper left", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "Values and the jump", [
        "Log(1 − i) = ½ ln 2 − iπ/4",
        "`cmath.log(1-1j) -> 0.346574-0.785398j`",
        "Log(e^(3πi)) = Log(−1) = πi ≠ 3πi",
        "",
        "Across the cut at −4:",
        "Log(−4 ± 0.001i) ≈ 1.3863 ± 3.1413i",
        "two points 0.002 apart, values ≈ 2π apart",
        "",
        "On the cut the sign of a zero decides:",
        "`log(complex(-1, 0.0))  ->  3.141593j`",
        "`log(complex(-1, -0.0)) -> -3.141593j`",
    ], fontsize=12, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    radii = np.linspace(0.02, 3, 300)
    angles = np.linspace(-np.pi + 1e-3, np.pi, 600)
    for r in [0.25, 0.5, 1, 2, 3]:
        image = np.log(r * np.exp(1j * angles))
        ax.plot(image.real, image.imag, color=BLUE, linewidth=1)
    for theta in np.linspace(-3 * np.pi / 4, np.pi / 2, 6):
        image = np.log(radii * np.exp(1j * theta))
        ax.plot(image.real, image.imag, color=MUTED, linewidth=1)
    for theta, color, label in [(np.pi - 0.05, ORANGE, "ray just above the cut"),
                                (-np.pi + 0.05, AQUA, "ray just below the cut")]:
        image = np.log(radii * np.exp(1j * theta))
        ax.plot(image.real, image.imag, color=color, linewidth=3, label=label, zorder=4)
    for level in (-np.pi, np.pi):
        ax.axhline(level, color=INK2, linestyle="--", linewidth=1, zorder=1)
    ax.plot([], [], color=BLUE, label=r"circles $|z| = r$ → Re $w = \ln r$")
    argand(ax, (-4.2, 1.5), (-4.0, 6.6), r"$w = \mathrm{Log}\, z$ opens the cut plane into a strip")
    pi_ticks(ax, "y", [-1, 0, 1], [r"$-\pi$", "0", r"$\pi$"])
    legend(ax, loc="upper left", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.axhspan(-np.pi, np.pi, color=BLUE_LIGHT, alpha=0.45, zorder=0,
               label=r"principal strip $-\pi < \mathrm{Im} \leq \pi$")
    for zz, color in [(-1, ORANGE), (1j, AQUA), (2 + 2j, BLUE), (0.3, INK2)]:
        col = np.array([cmath.log(zz) + 2j * math.pi * kk for kk in range(-1, 2)])
        ax.plot(col.real, col.imag, "o", markerfacecolor="none", color=color, markersize=9, markeredgewidth=1.6)
        ax.plot(cmath.log(zz).real, cmath.log(zz).imag, "o", color=color, markersize=9, label=f"log({zz})")
    argand(ax, (-3, 3), (-10, 10), r"All values: columns $2\pi$ apart")
    pi_ticks(ax, "y", [-3, -2, -1, 0, 1, 2, 3], [r"$-3\pi$", r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$", r"$3\pi$"])
    legend(ax, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)

    ax = fig.add_subplot(gs[1, 2])
    labels = [f"k = {v:+d}" for v in values]
    ax.bar(np.arange(3), shares, width=0.6, color=[ORANGE, BLUE, ORANGE], zorder=2)
    for i, s in enumerate(shares):
        ax.text(i, s + 0.02, f"{s:.3f}", ha="center", fontsize=11)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("share of 100,000 random pairs")
    ax.grid(axis="y", zorder=0)
    ax.set_title(r"Log(zw) = Log z + Log w + $2\pi i k$")
    ax.text(1, 0.86, "fails for a quarter of all pairs (seed 0)", ha="center", fontsize=9.5, color=INK2)
    save_03(fig, name="03_the_complex_logarithm_and_branch_cuts.png")


# ================================================================ module 03, notebook 04
def sheet_03_04_complex_powers():
    assert 1j**1j == math.exp(-math.pi / 2) and f"{1j ** 1j:.5f}" == "0.20788+0.00000j"
    assert np.isclose((1 + 1j) ** 1j, 0.4288290062943679 + 0.1548717524642468j)
    assert np.isclose(2**1j, 0.7692389013639721 + 0.6389612763136348j)
    assert np.isclose((-8) ** (1 / 3), 1 + 1j * math.sqrt(3))
    assert np.isclose((-8) ** (2 / 3), -2 + 2j * math.sqrt(3))

    def all_powers(z, w, ks):
        return [cmath.exp(w * (cmath.log(z) + 2j * math.pi * k)) for k in ks]

    def distinct(values, tolerance=1e-9):
        unique = []
        for v in values:
            if all(abs(v - u) > tolerance * max(1, abs(u)) for u in unique):
                unique.append(v)
        return unique

    counts = [len(distinct(all_powers(-8, w, range(-20, 21)))) for w in (3, 1 / 3, 2 / 3, 4 / 6, 5 / 4, math.sqrt(2))]
    assert counts == [1, 3, 3, 3, 4, 41]

    i_to_i = {k: math.exp(-math.pi / 2 - 2 * math.pi * k) for k in range(-2, 4)}
    assert [f"{i_to_i[k]:.6g}" for k in range(-2, 3)] == ["59609.7", "111.318", "0.20788", "0.000388203", "7.24947e-07"]

    rng = np.random.default_rng(seed=3)
    z = rng.normal(size=50_000) + 1j * rng.normal(size=50_000)
    w = rng.normal(size=50_000) + 1j * rng.normal(size=50_000)
    a, b = 0.7 + 0.2j, 1.3 - 0.5j
    laws = [("zᵃ zᵇ = z^(a+b)", np.isclose(z**a * z**b, z ** (a + b))),
            ("(zᵃ)³ = z^(3a)", np.isclose((z**a) ** 3, z ** (3 * a))),
            ("(zᵃ)ᵇ = z^(ab), a = 0.7+0.2i", np.isclose((z**a) ** b, z ** (a * b))),
            ("(z³)^(1/2) = z^(3/2)", np.isclose((z**3) ** 0.5, z**1.5)),
            ("√(zw) = √z √w", np.isclose(np.sqrt(z * w), np.sqrt(z) * np.sqrt(w)))]
    rates = [f"{holds.mean():.1%}" for _, holds in laws]
    assert rates == ["100.0%", "100.0%", "100.0%", "33.3%", "75.1%"]

    fig = new_figure("zʷ = e^(w log z): one value, q values or infinitely many",
                     r"Values $e^{w\,\mathrm{Log}\, z}\, e^{2\pi i k w}$: integer $w$ → one,  $w = p/q$ → $q$,  "
                     r"otherwise infinitely many.   Principal value $e^{w\,\mathrm{Log}\, z}$ is what Python's ** returns")
    gs = grid(fig, top=0.84, wspace=0.32, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 300)
    for exponent, radius, color, name in [(1 / 3, 2, BLUE, r"$(-8)^{1/3}$"), (2 / 3, 4, AQUA, r"$(-8)^{2/3}$")]:
        values = np.array(all_powers(-8, exponent, range(3)))
        assert np.allclose(np.abs(values), radius)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color=GRID, linewidth=1)
        polygon = np.append(values, values[0])
        ax.plot(polygon.real, polygon.imag, "--", color=color, linewidth=1)
        for v in values:
            arrow(ax, 0, v, color, width=1.8)
        principal = complex(-8) ** exponent
        ax.plot(principal.real, principal.imag, "*", color=ORANGE, markersize=15, zorder=6)
        ax.plot([], [], color=color, linewidth=1.8, label=f"{name}: 3 values")
    ax.plot([], [], "*", color=ORANGE, markersize=12, label="principal (Python **)")
    ax.text(-2.2, -0.75, "−2", fontsize=10, ha="right")
    ax.text(4.15, 0.35, "4", fontsize=10)
    argand(ax, (-5, 5), (-7.2, 5), r"Cube roots of $-8$: principal is $1 + i\sqrt{3}$")
    legend(ax, loc="lower right", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    k = np.arange(200)
    dense = np.exp(2j * np.pi * k * (np.sqrt(2) - 1))
    eighth = np.exp(2j * np.pi * k * 3 / 8)
    assert len(distinct(list(eighth))) == 8
    ax.plot(dense.real, dense.imag, "o", markersize=3, color=BLUE, label=r"$w = \sqrt{2} - 1$: dense")
    ax.plot(eighth.real, eighth.imag, "o", markersize=10, markerfacecolor="none", markeredgewidth=2,
            color=ORANGE, label=r"$w = 3/8$: 8 values")
    ax.plot(1, 0, "*", color=INK, markersize=14, zorder=5, label="principal $1^w = 1$")
    argand(ax, (-1.5, 1.5), (-1.5, 1.5), r"All values of $1^w = e^{2\pi i k w}$, $k < 200$")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 2])
    wc = 0.3 + 0.05j
    ks = np.arange(-10, 11)
    points = np.exp(2j * np.pi * ks * wc)
    ts = np.linspace(-10, 10, 2000)
    spiral = np.exp(2j * np.pi * ts * wc)
    ax.plot(spiral.real, spiral.imag, color=GRID, linewidth=1.5, label=r"spiral $e^{2\pi i t w}$")
    ax.plot(points.real, points.imag, "o", color=BLUE, markersize=5, label=r"values, $k = -10 \dots 10$")
    ax.plot(1, 0, "*", color=ORANGE, markersize=15, zorder=5, label="principal value")
    argand(ax, (-2.5, 2.5), (-2.5, 2.5), r"$w = 0.3 + 0.05i$: a logarithmic spiral")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    show_domain_colouring(ax, np.sqrt, (-2, 2), (-2, 2), r"Principal $\sqrt{z}$: sign flips across the cut",
                          resolution=420)
    ax.plot([-2, 0], [0, 0], color="white", linewidth=1.2, linestyle="--")
    ax.text(-1.95, 0.25, r"$\sqrt{-4 + 0.001i} \approx 2i$", fontsize=9.5,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(-1.95, -0.55, r"$\sqrt{-4 - 0.001i} \approx -2i$", fontsize=9.5,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.85))
    assert np.isclose(cmath.sqrt(-4 + 0.001j), 2j, atol=1e-3) and np.isclose(cmath.sqrt(-4 - 0.001j), -2j, atol=1e-3)

    ax = fig.add_subplot(gs[1, 1])
    kk = np.arange(-2, 4)
    vals = np.array([i_to_i[int(v)] for v in kk])
    ax.semilogy(kk, vals, "o-", color=BLUE, markersize=8, linewidth=1.2)
    ax.semilogy([0], [i_to_i[0]], "*", color=ORANGE, markersize=18, zorder=5,
                label=r"principal $e^{-\pi/2} \approx 0.2079$")
    for v, value in zip(kk, vals):
        ax.annotate(f"{value:.3g}", (v, value), xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("k")
    ax.set_ylabel("value (all real and positive)")
    ax.set_xlim(-2.6, 3.9)
    ax.grid(True, which="major")
    ax.set_title(r"All values of $i^i = e^{-\pi/2 - 2\pi k}$")
    legend(ax, loc="upper right", fontsize=8.5)

    lines = ["With principal values, on 50,000 random z, w:", ""]
    for (name, _), rate in zip(laws, rates):
        lines.append(f"{name}:  {rate}")
    lines += ["",
              "(z³)^(1/2) = z^(3/2) needs 3 Arg z ∈ (−π, π]:",
              "exactly one third of all directions.",
              "√(−1)·√(−1) = −1 but √((−1)(−1)) = 1."]
    panel(fig.add_subplot(gs[1, 2]), "Which exponent laws survive?", lines, fontsize=12, spacing=0.078)
    save_03(fig, name="04_complex_powers.png")


# ================================================================ module 03, notebook 05
def sheet_03_05_complex_trig_and_hyperbolic_functions():
    assert np.isclose(cmath.cos(1j), math.cosh(1)) and np.isclose(cmath.sin(1j), 1j * math.sinh(1))
    assert f"{cmath.sin(3 + 4j):.6f}" == "3.853738-27.016813j"
    assert f"{abs(cmath.sin(3 + 4j)):.6f}" == "27.290282"
    assert f"{cmath.sin(1 + 1j):.6f}" == "1.298458+0.634964j"
    table = {y: (abs(cmath.sin(1j * y)), abs(cmath.cos(1j * y))) for y in [0, 1, 2, 5, 10, 20]}
    assert f"{table[5][0]:.5g}" == "74.203" and f"{table[10][1]:.5g}" == "11013"
    root = math.log(2 + math.sqrt(3))
    assert f"{root:.4f}" == "1.3170"
    for sign in (+1, -1):
        u = 2 + sign * cmath.sqrt(3)
        for kk in range(-1, 2):
            zz = -1j * cmath.log(u) + 2 * math.pi * kk
            assert abs(cmath.cos(zz) - 2) < 1e-12
    sin_solutions = [math.pi / 2 + 2 * math.pi * kk + s * 1j * root for kk in (-1, 0, 1) for s in (1, -1)]
    assert all(abs(cmath.sin(zz) - 2) < 1e-12 for zz in sin_solutions)

    xs = np.linspace(-7, 7, 1401)
    ys = np.linspace(-3, 3, 601)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    for name, values, count in [("sin", np.sin(Z), 55), ("cos", np.cos(Z), 40)]:
        small = np.abs(values) < 0.02
        assert small.sum() == count and np.abs(Z[small].imag).max() <= 0.0100001

    fig = new_figure("Complex sine and cosine are unbounded: off the real axis they grow like ½e^|y|",
                     r"$\sin(x + iy) = \sin x \cosh y + i \cos x \sinh y$,   $|\sin z|^2 = \sin^2 x + \sinh^2 y$,   "
                     r"$\cos(iy) = \cosh y$.   Zeros stay real; $\cos z = 2$ at $2\pi k \pm 1.317i$")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.42)

    ax = fig.add_subplot(gs[0, 0:2])
    x = np.linspace(-2 * np.pi, 2 * np.pi, 600)
    y = np.linspace(-2, 2, 240)
    Xs, Ys = np.meshgrid(x, y)
    modulus = np.abs(np.sin(Xs + 1j * Ys))
    filled = ax.contourf(Xs, Ys, modulus, levels=np.linspace(0, 3.8, 20), cmap=SEQ, extend="max")
    ax.contour(Xs, Ys, modulus, levels=[1], colors=ORANGE, linewidths=1.6)
    ax.plot([], [], color=ORANGE, linewidth=1.6, label=r"level curve $|\sin z| = 1$")
    cbar = fig.colorbar(filled, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$|\sin z|$")
    cbar.outline.set_visible(False)
    zeros = np.pi * np.arange(-2, 3)
    ax.plot(zeros, np.zeros_like(zeros), "o", color="white", markeredgecolor=INK, markersize=8,
            label=r"zeros $k\pi$ (the only ones)")
    solutions = [s for s in sin_solutions if abs(s.real) <= 2 * np.pi]
    ax.plot([s.real for s in solutions], [s.imag for s in solutions], "D", color=AQUA, markeredgecolor=INK,
            markersize=7, label=r"$\sin z = 2$ at $\pi/2 \pm 1.317i$, period $2\pi$")
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    pi_ticks(ax, "x", [-2, -1, 0, 1, 2], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.set_title(r"The landscape $|\sin z|$: bounded by 1 only near the real axis")
    plain_image_axes(ax)
    legend(ax, loc="upper left", fontsize=8.5, bbox_to_anchor=(1.09, 1.0))

    panel(fig.add_subplot(gs[0, 2]), "Worked numbers", [
        "cos i = cosh 1 ≈ 1.5431, sin i = 1.1752i",
        "|sin(3 + 4i)| = √(sin²3 + sinh²4) ≈ 27.29",
        "",
        "`y    |sin(iy)|   e^y / 2`",
        "`5       74.203    74.207`",
        "`10       11013     11013`",
        "",
        "cos z = 2: e^(iz) = 2 ± √3,",
        "z = 2πk ∓ i ln(2 + √3) ≈ 2πk ∓ 1.3170i",
    ], fontsize=11, spacing=0.058, top=0.66)

    ax = fig.add_subplot(gs[1, 0])
    xl = np.linspace(-2 * np.pi, 2 * np.pi, 800)
    for yy, color in zip([0, 0.5, 1, 2], [INK2, AQUA, BLUE, ORANGE]):
        ax.plot(xl, np.abs(np.sin(xl + 1j * yy)), color=color, linewidth=1.8,
                label=f"y = {yy}: {np.sinh(abs(yy)):.2f} to {np.cosh(yy):.2f}")
    pi_ticks(ax, "x", [-2, -1, 0, 1, 2], [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
    ax.set_xlabel("x = Re z")
    ax.set_ylabel(r"$|\sin(x + iy)|$")
    ax.set_ylim(0, 4.6)
    ax.grid(True)
    ax.set_title(r"Between $\sinh|y|$ and $\cosh y$")
    legend(ax, loc="upper left", fontsize=8.5, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    tv = np.linspace(-1.5, 1.5, 300)
    th = np.linspace(-np.pi / 2, np.pi / 2, 300)
    for a in np.linspace(-np.pi / 2, np.pi / 2, 9):
        image = np.sin(a + 1j * tv)
        ax.plot(image.real, image.imag, color=BLUE, linewidth=1)
    for bb in np.linspace(-1.5, 1.5, 7):
        image = np.sin(th + 1j * bb)
        ax.plot(image.real, image.imag, color=ORANGE, linewidth=1)
    ax.plot([-1, 1], [0, 0], "o", color=INK, markersize=6, label=r"foci $\pm 1$")
    m = np.sin(0.5 + 1j)
    ax.plot(m.real, m.imag, "s", color=AQUA, markeredgecolor=INK, markersize=8, label=r"$\sin(0.5 + i)$")
    ax.plot([], [], color=ORANGE, label="y = b → ellipses")
    ax.plot([], [], color=BLUE, label="x = a → hyperbolas")
    argand(ax, (-2.6, 2.6), (-3.7, 2.6), r"$w = \sin z$ on $|x| \leq \pi/2$, $|y| \leq 1.5$")
    legend(ax, loc="lower center", fontsize=8, ncol=2, columnspacing=0.8)

    ax = fig.add_subplot(gs[1, 2])
    tt = np.linspace(-3, 3, 400)
    assert np.allclose(np.cos(1j * tt).imag, 0) and np.allclose(np.cos(1j * tt), np.cosh(tt))
    ax.plot(tt, np.cos(tt + 0j).real, color=BLUE, linewidth=2, label=r"$\cos t$ (real axis)")
    ax.plot(tt, np.cos(1j * tt).real, color=ORANGE, linewidth=2, label=r"$\cos(it)$ (imaginary axis)")
    ax.plot(tt, np.cosh(tt), ":", color=INK, linewidth=2.5, label=r"$\cosh t$")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("value (real in both cases)")
    ax.grid(True)
    ax.set_title(r"One function, two directions: $\cos(it) = \cosh t$")
    legend(ax, loc="upper center", fontsize=8.5)
    save_03(fig, name="05_complex_trig_and_hyperbolic_functions.png")


# ================================================================ module 03, notebook 06
def sheet_03_06_mobius_transformations():
    def apply(M, z):
        return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])

    T = np.array([[1, -1], [1, 1]], dtype=complex)
    assert apply(T, 0) == -1 and apply(T, 1) == 0 and np.isclose(apply(T, 1j), 1j)

    rng = np.random.default_rng(seed=7)
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    N = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    z = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    assert np.allclose(apply(M, apply(N, z)), apply(M @ N, z))
    assert not np.allclose(apply(N, apply(M, z)), apply(M @ N, z))

    Mc = np.array([[1, 2j], [1 - 1j, 3]])
    circle = (0.5 + 0.2j) + 1.3 * np.exp(1j * np.linspace(0, 2 * np.pi, 400, endpoint=False))
    image = apply(Mc, circle)
    p1, p2, p3 = image[0], image[133], image[266]
    A = np.array([[2 * (p2 - p1).real, 2 * (p2 - p1).imag], [2 * (p3 - p1).real, 2 * (p3 - p1).imag]])
    rhs = np.array([abs(p2) ** 2 - abs(p1) ** 2, abs(p3) ** 2 - abs(p1) ** 2])
    centre = complex(*np.linalg.solve(A, rhs))
    radius = abs(p1 - centre)
    assert f"{np.round(centre, 4)}" == "(-0.0481+0.6346j)" and round(radius, 4) == 0.2795
    assert np.max(np.abs(np.abs(image - centre) - radius)) < 1e-14

    def cross_ratio(z, z1, z2, z3):
        return (z - z1) * (z2 - z3) / ((z - z3) * (z2 - z1))

    assert np.isclose(cross_ratio(1j, 1, -1, -1j), 0.5)
    assert np.isclose(cross_ratio(2j, 1, -1, -1j), 0.5 - 1j / 6)

    cayley = np.array([[1, -1j], [1, 1j]])
    inverse = np.array([[1j, 1j], [-1, 1]])
    rng = np.random.default_rng(seed=2)
    upper = rng.normal(size=10_000) + 1j * np.abs(rng.normal(size=10_000))
    real_line = rng.normal(size=1000) * 10
    disc = np.sqrt(rng.uniform(0, 1, 10_000)) * np.exp(2j * np.pi * rng.uniform(size=10_000))
    assert np.all(np.abs(apply(cayley, upper)) < 1)
    assert np.allclose(np.abs(apply(cayley, real_line)), 1)
    assert np.all(apply(inverse, disc).imag > 0)
    assert np.allclose(inverse @ cayley, 2j * np.eye(2))

    fixed = sp.solve(sp.Eq((2 * sp.Symbol("z") + 1) / (sp.Symbol("z") + 2), sp.Symbol("z")))
    assert sorted(fixed) == [-1, 1]
    T4 = lambda z: (z + 1j) / (1j * z + 1)
    assert np.isclose(T4(0), 1j) and np.isclose(T4(1), 1) and np.isclose(T4(1e9), -1j, atol=1e-8)

    def cayley_map(z):
        return (z - 1j) / (z + 1j)

    fig = new_figure("Möbius maps send circles and lines to circles and lines",
                     r"$T(z) = \frac{az + b}{cz + d}$, $ad - bc \neq 0$:   composition = product of the matrices [[a, b], [c, d]];   "
                     r"three points fix $T$.   "
                     r"Cayley $C(z) = \frac{z - i}{z + i}$: half-plane → disc")
    gs = grid(fig, top=0.84, wspace=0.3, hspace=0.42)

    t = np.linspace(-60, 60, 20001)
    s = np.linspace(1e-3, 200, 40001)
    heights = [0.25, 0.5, 1, 2, 4]
    columns = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]

    ax = fig.add_subplot(gs[0, 0])
    for b in heights:
        ax.plot(t, b + 0 * t, color=ORANGE, linewidth=1)
    for a in columns:
        ax.plot(a + 0 * s, s, color=BLUE, linewidth=1)
    ax.plot(t, 0 * t, color=INK, linewidth=2.5, label="real axis")
    ax.plot(0, 1, "*", color=AQUA, markeredgecolor=INK, markersize=15, zorder=5, label="$i$")
    argand(ax, (-3.5, 3.5), (-1.5, 4.6), "Upper half-plane grid")
    legend(ax, loc="lower center", fontsize=8.5, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    for b in heights:
        image = cayley_map(t + 1j * b)
        ax.plot(image.real, image.imag, color=ORANGE, linewidth=1)
    for a in columns:
        image = cayley_map(a + 1j * s)
        ax.plot(image.real, image.imag, color=BLUE, linewidth=1)
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    ax.plot(unit.real, unit.imag, color=INK, linewidth=2.5, label="image of the real axis")
    ax.plot(0, 0, "*", color=AQUA, markeredgecolor=INK, markersize=15, zorder=5, label="$C(i) = 0$")
    ax.plot(1, 0, "s", color=INK, label=r"$C(\infty) = 1$")
    argand(ax, (-1.3, 1.3), (-1.85, 1.3), r"$C(z) = (z - i)/(z + i)$: the unit disc")
    legend(ax, loc="lower center", fontsize=8, ncol=3, columnspacing=0.6, handletextpad=0.3)

    panel(fig.add_subplot(gs[0, 2]), "Matrices do the bookkeeping", [
        "T ↔ [[a, b], [c, d]];  T_M ∘ T_N = T_MN",
        "(order matters: T_N ∘ T_M ≠ T_MN, seed 7)",
        "",
        "Ex. 2: (z + 1)/(z − 1) after 1/z:",
        "`[[1, 1], [1, -1]] @ [[0, 1], [1, 0]]`",
        "`  = [[1, 1], [-1, 1]]  ->  (z+1)/(1-z)`",
        "",
        "Cayley:  C⁻¹ · C = 2i · identity,",
        "so C⁻¹(w) = i(1 + w)/(1 − w).",
        "Circle |z − (0.5 + 0.2i)| = 1.3 under",
        "[[1, 2i], [1−i, 3]]: all 400 image points",
        "lie on one circle of radius 0.2795.",
    ], fontsize=12, spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    disc_map = lambda z: (z - 0.5) / (1 - 0.5 * z)
    theta = np.linspace(0, 2 * np.pi, 400)
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        image = disc_map(r * np.exp(1j * theta))
        ax.plot(image.real, image.imag, color=BLUE, linewidth=2.4 if r == 1.0 else 1)
    rr = np.linspace(0, 1, 200)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        image = disc_map(rr * np.exp(1j * angle))
        ax.plot(image.real, image.imag, color=AQUA, linewidth=1)
    ax.plot(-0.5, 0, "o", color=INK, markersize=7, zorder=5, label="$T(0) = -0.5$")
    ax.plot(0, 0, "*", color=ORANGE, markersize=14, zorder=5, label="$T(0.5) = 0$")
    ax.plot([], [], color=BLUE, label="images of |z| = 0.2 … 1")
    ax.plot([], [], color=AQUA, label="images of rays")
    argand(ax, (-1.15, 1.15), (-1.15, 1.6), r"$T(z) = \frac{z - 0.5}{1 - 0.5z}$ on a polar grid")
    legend(ax, loc="upper center", fontsize=8, ncol=2, columnspacing=0.8)

    ax = fig.add_subplot(gs[1, 1])
    Tf = lambda z: (2 * z + 1) / (z + 2)
    for centre_height in [-1.5, -0.6, -0.25, 0.25, 0.6, 1.5]:
        c = 1j * centre_height
        rad = abs(1 - c)
        ring = c + rad * np.exp(1j * theta)
        ax.plot(ring.real, ring.imag, color=GRID, linewidth=1, zorder=0)
    starts = -1 + 0.12 * np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
    for start, colour in zip(starts, DEEP(np.linspace(0, 1, len(starts)))):
        orbit = [start]
        for _ in range(12):
            orbit.append(Tf(orbit[-1]))
        orbit = np.array(orbit)
        assert abs(orbit[-1] - 1) < 1e-4
        ax.plot(orbit.real, orbit.imag, "-", color=colour, linewidth=0.6, alpha=0.5)
        ax.plot(orbit.real, orbit.imag, "o", color=colour, markersize=3.5)
    ax.plot(1, 0, "*", color=ORANGE, markersize=16, zorder=5, label="attracting 1, |T'(1)| = 1/3")
    ax.plot(-1, 0, "X", color=INK, markersize=11, zorder=5, label="repelling −1, |T'(−1)| = 3")
    argand(ax, (-2, 2), (-1.7, 1.7), r"Iterating $T(z) = \frac{2z + 1}{z + 2}$")
    legend(ax, loc="lower left", fontsize=8)

    panel(fig.add_subplot(gs[1, 2]), "Three points and the cross-ratio", [
        "(z, z₁, z₂, z₃) = (z − z₁)(z₂ − z₃) / ((z − z₃)(z₂ − z₁))",
        "sends z₁, z₂, z₃ to 0, 1, ∞ and is Möbius-invariant.",
        "",
        "Four points are concyclic ⇔ cross-ratio is real:",
        "(i, 1, −1, −i) = 1/2  → on the unit circle",
        "(2i, 1, −1, −i) = 0.5 − 0.1667i  → not",
        "",
        "Ex. 4: 0 ↦ i, 1 ↦ 1, ∞ ↦ −i gives",
        "T(z) = (z + i)/(iz + 1): real axis → unit circle.",
        "Fixed points solve cz² + (d − a)z − b = 0:",
        "at most two unless T is the identity.",
    ], fontsize=12, spacing=0.078)
    save_03(fig, name="06_mobius_transformations.png")


if __name__ == "__main__":
    sheet_02_01_quadratics_and_conjugate_roots()
    sheet_02_02_fundamental_theorem_of_algebra()
    sheet_02_03_newtons_method_in_the_complex_plane()
    sheet_02_04_partial_fractions_over_c()
    sheet_03_01_complex_functions_as_maps()
    sheet_03_02_the_complex_exponential()
    sheet_03_03_the_complex_logarithm_and_branch_cuts()
    sheet_03_04_complex_powers()
    sheet_03_05_complex_trig_and_hyperbolic_functions()
    sheet_03_06_mobius_transformations()
