#!/usr/bin/env python3
"""Concept sheets for modules 00 (foundations) and 01 (polar form and Euler's formula).

One page per notebook that summarizes its central idea, algebraically and in the complex plane.
Every number is computed here from the same complex numbers, polynomials, step counts and random
seeds the notebook uses, and is checked against the notebook's printed outputs with asserts.
Images are written to modules/00_foundations/img/ and modules/01_polar_form_and_euler/img/.

Run from anywhere:  python scripts/make_modules_00_01_concept_images.py
"""
import cmath
import math
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, panel)
from viz_style import save as _save

FOUNDATIONS = REPO / "modules" / "00_foundations"
POLAR = REPO / "modules" / "01_polar_form_and_euler"
save_00 = partial(_save, folder=FOUNDATIONS / "img")
save_01 = partial(_save, folder=POLAR / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
UNIT = np.exp(1j * np.linspace(0, 2 * np.pi, 400))


# ================================================================ helpers
def argand(ax, xlim, ylim, title):
    """Argand frame: equal aspect, both axes through 0, axes labelled Re and Im."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.grid(True)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)


def arrow(ax, start, end, color, width=2.2, zorder=4, label=None):
    start, end = complex(start), complex(end)
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, mutation_scale=15,
                                shrinkA=0, shrinkB=0))
    if label is not None:
        ax.plot([], [], color=color, linewidth=width, label=label)


def unit_circle(ax, radius=1.0, center=0.0, **kw):
    style = dict(color=GRID, linewidth=1.4, zorder=1)
    style.update(kw)
    curve = center + radius * UNIT
    ax.plot(curve.real, curve.imag, **style)


def fill_shape(ax, shape, color, light, label=None, alpha=0.55):
    closed = np.append(shape, shape[0])
    ax.fill(closed.real, closed.imag, color=light, alpha=alpha, zorder=2, label=label)
    ax.plot(closed.real, closed.imag, color=color, linewidth=1.6, zorder=3)


def legend(ax, **kw):
    style = dict(fontsize=9, framealpha=0.92, facecolor=SURFACE, frameon=True, edgecolor="none")
    style.update(kw)
    ax.legend(**style)


# ================================================================ module 00, notebook 01
def solve_quadratic(a, b, c):
    discriminant = b * b - 4 * a * c
    root_of_discriminant = cmath.sqrt(discriminant)
    return (-b + root_of_discriminant) / (2 * a), (-b - root_of_discriminant) / (2 * a)


def cardano_one_root(p, q):
    inner = cmath.sqrt(q * q / 4 + p ** 3 / 27)
    u = (-q / 2 + inner) ** (1 / 3)
    v = -p / (3 * u)
    return u + v, u, v


def sheet_00_01_why_complex_numbers_exist():
    # 4.3 powers of i and i^2026
    powers = [1j ** n for n in range(4)]
    assert np.allclose(powers, [1, 1j, -1, -1j]) and sp.I ** 2026 == -1
    assert str(cmath.sqrt(-9)) == "3j"
    # 4.4 the quadratic solver on the notebook's four examples
    expected = {(1, -3, 2): (2, 1), (1, -2, 1): (1, 1), (1, 0, 4): (2j, -2j), (1, -2, 5): (1 + 2j, 1 - 2j)}
    for (a, b, c), roots in expected.items():
        assert np.allclose(solve_quadratic(a, b, c), roots)
    # 4.5 Bombelli
    assert sp.expand((2 + sp.I) ** 3) == 2 + 11 * sp.I and sp.expand((2 - sp.I) ** 3) == 2 - 11 * sp.I
    assert (-4) ** 2 / 4 + (-15) ** 3 / 27 == -121.0
    root, u, v = cardano_one_root(p=-15, q=-4)
    assert str(u) == "(2+1.0000000000000002j)" and str(root) == "(4+0j)" and abs(v - (2 - 1j)) < 1e-12
    cubic_roots = np.array([4, -2 + np.sqrt(3), -2 - np.sqrt(3)])
    assert np.allclose(np.polyval([1, 0, -15, -4], cubic_roots), 0)

    fig = new_figure("Complex numbers exist because $i^2 = -1$ makes every quadratic, and Cardano, work",
                     r"$z = a + bi$ with $a, b$ real.   $\Delta < 0$:  $x = \dfrac{-b}{2a} \pm \dfrac{\sqrt{-\Delta}}{2a}\,i$"
                     r",  a conjugate pair.   Bombelli: $(2+i)^3 = 2 + 11i$, so $\sqrt[3]{2+11i} + \sqrt[3]{2-11i} = 4$.")
    gs = grid(fig, top=0.82)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(-3, 4, 400)
    for (label, coeffs), color in zip([(r"$x^2 - 3x + 2$  ($\Delta = 1$)", [1, -3, 2]),
                                       (r"$x^2 + 1$  ($\Delta = -4$)", [1, 0, 1]),
                                       (r"$x^2 - 2x + 5$  ($\Delta = -16$)", [1, -2, 5])],
                                      [BLUE, ORANGE, AQUA]):
        ax.plot(x, np.polyval(coeffs, x), color=color, linewidth=2.2, label=label)
    ax.scatter([1, 2], [0, 0], color=BLUE, s=40, zorder=4)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(-2, 12)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Real roots = crossings; two parabolas miss")
    legend(ax, loc="upper center")

    ax = fig.add_subplot(gs[0, 1])
    sets = [("$x^2 - 3x + 2$: 1, 2", [1, -3, 2], BLUE), (r"$x^2 + 1$: $\pm i$", [1, 0, 1], ORANGE),
            (r"$x^2 - 2x + 5$: $1 \pm 2i$", [1, -2, 5], AQUA), (r"$x^2 + 2x + 10$: $-1 \pm 3i$", [1, 2, 10], INK2)]
    for label, coeffs, color in sets:
        roots = np.roots(coeffs)
        if abs(roots[0].imag) > 0:
            ax.plot(roots.real, roots.imag, color=color, linestyle=":", linewidth=1.3, zorder=2)
        ax.scatter(roots.real, roots.imag, s=70, color=color, zorder=4, label=label)
    assert np.allclose(sorted(np.roots([1, 2, 10]), key=lambda r: r.imag), [-1 - 3j, -1 + 3j])
    argand(ax, (-2, 6.2), (-3.8, 3.8), "Missing roots live off the real line")
    ax.text(6.0, -3.5, "conjugate pairs: mirror\nimages across the Re axis", ha="right", fontsize=9.5, color=INK2)
    legend(ax, loc="upper right", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "Each extension solved an equation", [
        "x + 5 = 2  →  negatives, ℤ",
        "3x = 2  →  fractions, ℚ",
        "x² = 2  →  irrationals, ℝ",
        "x² = −1  →  i, and ℂ = {a + bi}",
        "",
        "i⁰, i¹, i², i³ = 1, i, −1, −i, then repeat:",
        "i²⁰²⁶ = i² = −1  (2026 = 4·506 + 2)",
        "",
        "`math.sqrt(-9)  -> ValueError`",
        "`cmath.sqrt(-9) -> 3j`",
        "Re(3 − 4i) = 3,  Im(3 − 4i) = −4 (real)",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    xc = np.linspace(-4.5, 4.8, 400)
    ax.plot(xc, xc ** 3 - 15 * xc - 4, color=BLUE, linewidth=2.2, label="$x^3 - 15x - 4$")
    ax.scatter(cubic_roots, np.zeros(3), color=ORANGE, s=60, zorder=4,
               label=r"roots $4,\ -2 \pm \sqrt{3}$ (all real)")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylim(-40, 45)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"Bombelli's cubic $x^3 = 15x + 4$: three crossings")
    legend(ax, loc="upper left")

    ax = fig.add_subplot(gs[1, 1])
    arrow(ax, 0, 2 + 1j, BLUE, label=r"$u = 2 + i$,  $u^3 = 2 + 11i$")
    arrow(ax, 0, 2 - 1j, ORANGE, label=r"$v = 2 - i$,  $v^3 = 2 - 11i$")
    arrow(ax, 2 + 1j, 4, ORANGE, width=1.6)
    arrow(ax, 0, 4, AQUA, width=3, zorder=5, label="$u + v = 4$: imaginary parts cancel")
    ax.scatter([4], [0], color=AQUA, s=60, zorder=6)
    argand(ax, (-0.6, 4.8), (-2.2, 2.4), r"Cardano's detour: $x = u + v$ through $\sqrt{-121} = 11i$")
    legend(ax, loc="upper left", fontsize=8.5)

    panel(fig.add_subplot(gs[1, 2]), "Bombelli, 1572", [
        "x³ + px + q = 0 with p = −15, q = −4:",
        "q²/4 + p³/27 = 4 − 125 = −121 < 0",
        "",
        r"$x = \sqrt[3]{2 + 11i} + \sqrt[3]{2 - 11i}$",
        "",
        "(2 + i)³ = 8 + 12i − 6 − i = 2 + 11i",
        "",
        "`u = (2+1.0000000000000002j)`",
        "`u + v = (4+0j)`",
        "",
        "Not a fix for an unsolvable equation:",
        "a necessary detour to a real answer.",
    ], spacing=0.08)
    save_00(fig, name="01_why_complex_numbers_exist.png")


# ================================================================ module 00, notebook 02
class ComplexNumber:
    # the notebook's 4.6 class, reduced to what the check needs
    def __init__(self, re, im=0.0):
        self.re = float(re)
        self.im = float(im)

    def __add__(self, other):
        return ComplexNumber(self.re + other.re, self.im + other.im)

    def __mul__(self, other):
        a, b, c, d = self.re, self.im, other.re, other.im
        return ComplexNumber(a * c - b * d, a * d + b * c)


HOUSE_02 = np.array([1 + 0.5j, 3 + 0.5j, 3 + 2j, 2 + 3j, 1 + 2j])


def sheet_00_02_cartesian_form_and_arithmetic():
    z, w = 3 + 2j, 1 - 4j
    assert z + w == 4 - 2j and z - w == 2 + 6j and z * w == 11 - 10j
    assert (2 - 3j) ** 2 == -5 - 12j and (3 + 2j) * (3 - 2j) == 13 and 1j * z == -2 + 3j
    a, b, c, d = sp.symbols("a b c d", real=True)
    product = sp.expand((a + b * sp.I) * (c + d * sp.I))
    assert sp.re(product) == a * c - b * d and sp.im(product) == a * d + b * c
    assert sum(sp.I ** n for n in range(102)) == 1 + sp.I and sp.I ** 99 == -sp.I and sp.I ** -6 == -1
    rng = np.random.default_rng(seed=0)
    worst_error = 0.0
    for _ in range(10_000):
        a_, b_, c_, d_ = rng.uniform(-10, 10, size=4)
        mine_sum = ComplexNumber(a_, b_) + ComplexNumber(c_, d_)
        mine_product = ComplexNumber(a_, b_) * ComplexNumber(c_, d_)
        worst_error = max(worst_error,
                          abs(complex(mine_sum.re, mine_sum.im) - (complex(a_, b_) + complex(c_, d_))),
                          abs(complex(mine_product.re, mine_product.im) - complex(a_, b_) * complex(c_, d_)))
    assert f"{worst_error:.2e}" == "2.93e-14", worst_error

    fig = new_figure("Complex arithmetic: add the parts, multiply by FOIL with $i^2 = -1$",
                     r"$(a+bi) + (c+di) = (a+c) + (b+d)i$,   $(a+bi)(c+di) = (ac - bd) + (ad + bc)i$.   "
                     r"Geometrically: add = tip-to-tail, multiply by $i$ = quarter turn.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    total = z + w
    arrow(ax, 0, z, BLUE, label="z = 3 + 2i")
    arrow(ax, 0, w, ORANGE, label="w = 1 − 4i")
    arrow(ax, z, total, ORANGE, width=1.4)
    arrow(ax, w, total, BLUE, width=1.4)
    arrow(ax, 0, total, AQUA, width=3, zorder=5, label="z + w = 4 − 2i")
    argand(ax, (-1, 7.6), (-4.6, 3.6), "Addition: tip-to-tail (parallelogram)")
    legend(ax, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    for power, (color, light) in enumerate([(BLUE, BLUE_LIGHT), (ORANGE, ORANGE_LIGHT),
                                            (AQUA, AQUA_LIGHT), (INK2, GRID)]):
        shape = (1j ** power) * HOUSE_02
        fill_shape(ax, shape, color, light)
        centre = shape.mean()
        ax.text(centre.real, centre.imag, f"$i^{power}\\cdot$ house", ha="center", va="center", fontsize=9.5,
                color=INK)
    argand(ax, (-4, 4), (-4, 4), "Times i: (a, b) → (−b, a), a quarter turn")

    panel(fig.add_subplot(gs[0, 2]), "FOIL, then i² = −1", [
        "zw = (3 + 2i)(1 − 4i)",
        "    = 3 − 12i + 2i − 8i²",
        "    = 3 − 10i + 8 = 11 − 10i",
        "(2 − 3i)² = 4 − 12i − 9 = −5 − 12i",
        "(3 + 2i)(3 − 2i) = 9 + 4 = 13 (real!)",
        "",
        "One complex equation = two real ones:",
        "(x + yi)(2 + i) = 1 + 8i ⇒ x = 2, y = 3",
        "",
        "`10,000 random pairs vs complex: 2.93e-14`",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    image = (1 + 1j) * HOUSE_02
    fill_shape(ax, HOUSE_02, BLUE, BLUE_LIGHT, label="house")
    fill_shape(ax, image, ORANGE, ORANGE_LIGHT, label="(1 + i) · house")
    side_before = np.abs(np.diff(np.append(HOUSE_02, HOUSE_02[0])))
    side_after = np.abs(np.diff(np.append(image, image[0])))
    assert np.allclose(side_after / side_before, np.sqrt(2))
    argand(ax, (-3, 5), (-1, 7), "Times 1 + i: rotate 45°, scale by √2 ≈ 1.41")
    legend(ax, loc="upper right")

    ax = fig.add_subplot(gs[1, 1])
    exponents = np.arange(-4, 8)
    values = np.array([complex(1j ** int(n)) for n in exponents])
    unit_circle(ax)
    ax.scatter(values.real, values.imag, s=110, color=BLUE, zorder=4)
    for point, (dx, dy), ha in [(1, (0.12, 0.18), "left"), (1j, (0.12, 0.2), "left"),
                                (-1, (-0.1, 0.18), "right"), (-1j, (0.12, -0.3), "left")]:
        hits = [int(n) for n, p in zip(exponents, values) if abs(p - point) < 1e-12]
        assert len(hits) == 3 and all((n - hits[0]) % 4 == 0 for n in hits)
        ax.text(point.real + dx, point.imag + dy, "n = " + ", ".join(map(str, hits)).replace("-", "−"), fontsize=9.5,
                ha=ha, color=INK)
    argand(ax, (-2.1, 2.1), (-1.8, 1.8), r"$i^n$ for $n = -4, \dots, 7$: four points only")

    panel(fig.add_subplot(gs[1, 2]), "ℂ is a field, but not ordered", [
        "Closure, associative, commutative,",
        "identities, inverses, distributive: all hold",
        "(sympy: every difference simplifies to 0).",
        "So every real identity using +, −, ×, ÷ survives.",
        "",
        "No order: i ≠ 0 forces i² = −1 > 0, and",
        "1 = 1² > 0, so 0 = −1 + 1 > 0. Contradiction.",
        "`1j < 2j  -> TypeError`",
        "",
        "iⁿ = i^(n mod 4), negative n too: i⁻⁶ = −1.",
    ], spacing=0.08)
    save_00(fig, name="02_cartesian_form_and_arithmetic.png")


# ================================================================ module 00, notebook 03
def sheet_00_03_conjugate_modulus_and_division():
    z, w = 3 + 4j, 1 - 2j
    assert z.conjugate() == 3 - 4j and abs(z) == 5.0 and z * z.conjugate() == 25
    assert z / w == -1 + 2j and abs(1 / (3 + 4j) - (0.12 - 0.16j)) < 1e-15
    assert sp.Abs(1 - 2 * sp.I) == sp.sqrt(5) and z ** 2 == -7 + 24j
    assert abs((3 + 4j) ** 3 * (1 - 2j) ** 2) - 625 < 1e-9
    assert (2 + 1j) * (3 + 2j) == 4 + 7j and (2 + 1j) * (3 - 2j) == 8 - 1j
    # 4.4 seed 2
    rng = np.random.default_rng(seed=2)
    zs = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    ws = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    mult_error = np.max(np.abs(np.abs(zs * ws) - np.abs(zs) * np.abs(ws)))
    fraction_equal = np.mean(np.isclose(np.abs(zs + ws), np.abs(zs) + np.abs(ws)))
    assert repr(mult_error) == "2.6645352591003757e-15" and fraction_equal == 0.002
    assert np.all(np.abs(zs + ws) <= np.abs(zs) + np.abs(ws) + 1e-12)

    fig = new_figure(r"Conjugate reflects, modulus measures: $z\bar z = |z|^2$ makes division work",
                     r"$\bar z = a - bi$,   $|z| = \sqrt{a^2 + b^2}$,   $z\bar z = |z|^2$,   "
                     r"$\dfrac{z}{w} = \dfrac{z\,\bar w}{|w|^2}$,   $|zw| = |z|\,|w|$  but only  $|z + w| \leq |z| + |w|$")
    gs = grid(fig, top=0.82)

    ax = fig.add_subplot(gs[0, 0])
    unit_circle(ax, radius=5, color=BLUE_LIGHT)
    arrow(ax, 0, z, BLUE, label="z = 3 + 4i")
    arrow(ax, 0, z.conjugate(), ORANGE, label="z̄ = 3 − 4i")
    ax.plot([3, 3], [4, -4], color=INK2, linestyle="--", linewidth=1.1, label="reflection in the real axis")
    ax.plot([0, 3], [0, 0], color=AQUA, linewidth=3.5, zorder=3, label="Re z = 3")
    ax.plot([3, 3], [0, 4], color=INK, linewidth=2.5, zorder=3, label="Im z = 4")
    ax.text(-3.3, 4.3, "|z| = 5", color=BLUE, fontsize=10.5)
    ax.text(3.3, 4.3, "z = 3 + 4i", color=BLUE, fontsize=10.5)
    ax.text(3.3, -4.7, "z̄ = 3 − 4i", color=ORANGE, fontsize=10.5)
    ax.text(1.5, -0.55, "Re z = 3", color=AQUA, fontsize=10, ha="center")
    ax.text(3.25, 1.8, "Im z = 4", color=INK, fontsize=10)
    ax.text(3.25, -2.2, "mirror in\nthe Re axis", color=INK2, fontsize=9)
    argand(ax, (-6, 7.5), (-6, 6), "Conjugate = mirror image, |z| = distance")

    ax = fig.add_subplot(gs[0, 1])
    unit_circle(ax, color=MUTED, linewidth=1.2)
    points = np.array([2 + 1j, 0.5 + 0.5j, np.exp(1j * 2.2)])
    for p, color in zip(points, [BLUE, ORANGE, AQUA]):
        r = 1 / p
        assert np.isclose(r, p.conjugate() / abs(p) ** 2) and np.isclose(abs(r), 1 / abs(p))
        ax.scatter([p.real], [p.imag], s=60, color=color, zorder=4)
        ax.scatter([r.real], [r.imag], s=60, color=color, marker="s", zorder=4)
        ax.annotate("", xy=(r.real, r.imag), xytext=(p.real, p.imag),
                    arrowprops=dict(arrowstyle="->", color=color, linestyle="--", linewidth=1.3))
        ax.text(p.real + 0.08, p.imag + 0.1, f"|w| = {abs(p):.2f}", color=color, fontsize=9)
        ax.text(r.real + 0.08, r.imag - 0.25, f"|1/w| = {abs(r):.2f}", color=color, fontsize=9)
    ax.plot([], [], "o", color=INK2, label="w")
    ax.plot([], [], "s", color=INK2, label=r"$1/w = \bar w / |w|^2$")
    argand(ax, (-2, 3.2), (-2.2, 2), "Reciprocal: reflect, then invert the length")
    legend(ax, loc="lower left")

    panel(fig.add_subplot(gs[0, 2]), "Divide by multiplying by w̄", [
        "(3 + 4i) / (1 − 2i)",
        "   = (3 + 4i)(1 + 2i) / ((1 − 2i)(1 + 2i))",
        "   = (−5 + 10i) / 5 = −1 + 2i",
        "Check: (−1 + 2i)(1 − 2i) = 3 + 4i",
        "1/(3 + 4i) = (3 − 4i)/25 = 0.12 − 0.16i,",
        "   of length 0.2 = 1/|3 + 4i|",
        "",
        "Conjugation respects +, −, ×, ÷;",
        "Re z = (z + z̄)/2,   Im z = (z − z̄)/(2i)",
        "`abs(z)**2 = 25.0, but z**2 = (-7+24j)`",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    circle = 1.5 * UNIT
    wv = 1 + 1j
    ax.plot(circle.real, circle.imag, color=BLUE, linewidth=2.2, label="|z| = 1.5")
    ax.plot((wv * circle).real, (wv * circle).imag, color=ORANGE, linewidth=2.2,
            label=f"wz: radius 1.5·√2 = {1.5 * abs(wv):.2f}")
    ax.plot((wv + circle).real, (wv + circle).imag, color=AQUA, linewidth=2, linestyle="--",
            label="z + w: shifted, not rescaled")
    assert np.allclose(np.abs(wv * circle), 1.5 * np.sqrt(2))
    ax.scatter([1], [1], color=INK, zorder=5)
    ax.text(1.08, 0.72, "w = 1 + i", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.85))
    argand(ax, (-3.2, 3.2), (-3.2, 3.6), "Moduli multiply: circles stay centred at 0")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    lim = 7.5
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, zorder=1)
    ax.scatter(np.abs(zs) * np.abs(ws), np.abs(zs * ws), s=8, color=BLUE, alpha=0.6, zorder=3,
               label=f"|zw| vs |z||w|: on the line (max error {mult_error:.1e})")
    ax.scatter(np.abs(zs) + np.abs(ws), np.abs(zs + ws), s=8, color=ORANGE, alpha=0.5, zorder=2,
               label=f"|z+w| vs |z|+|w|: below it (equal for {fraction_equal:.1%})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("prediction  |z||w|  or  |z| + |w|")
    ax.set_ylabel("actual size")
    ax.set_title("1000 random pairs (seed 2)")
    legend(ax, loc="upper left", fontsize=8.5, markerscale=2.5)

    panel(fig.add_subplot(gs[1, 2]), "Why size multiplies", [
        r"$|zw|^2 = zw\,\overline{zw} = (z\bar z)(w\bar w) = |z|^2 |w|^2$",
        "",
        "In coordinates (Brahmagupta–Fibonacci):",
        "(ac − bd)² + (ad + bc)² = (a² + b²)(c² + d²)",
        "65 = 5·13:  (2 + i)(3 + 2i) = 4 + 7i → 4² + 7²",
        "                 (2 + i)(3 − 2i) = 8 − i → 8² + 1²",
        "",
        "|(3 + 4i)³(1 − 2i)²| = 5³ · 5 = 625",
        "Sums only obey the triangle inequality.",
    ], spacing=0.085)
    save_00(fig, name="03_conjugate_modulus_and_division.png")


# ================================================================ module 00, notebook 04
FLAG = np.array([1 + 0.5j, 1 + 2.5j, 2.5 + 2j, 1 + 1.5j])


def sheet_00_04_the_complex_plane():
    z, w = 1 + 3j, 4 - 1j
    assert abs(z - w) == 5.0 and (z + w) / 2 == 2.5 + 1j
    rng = np.random.default_rng(seed=0)
    size = 1_000_000
    zs = rng.normal(size=size) + 1j * rng.normal(size=size)
    ws = rng.normal(size=size) + 1j * rng.normal(size=size)
    ratio = np.abs(zs + ws) / (np.abs(zs) + np.abs(ws))
    assert f"{ratio.min():.4f}" == "0.0007" and f"{ratio.max():.6f}" == "1.000000"
    assert np.all(np.abs(np.abs(zs) - np.abs(ws)) <= np.abs(zs - ws) + 1e-12)
    x = np.linspace(-2, 2, 801)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y
    values = np.abs(Z ** 2 + 3 * Z - 1)
    values[np.abs(Z) > 2] = 0
    index = np.unravel_index(np.argmax(values), values.shape)
    assert f"{values[index]:.4f}" == "9.0000" and f"{Z[index]:.3f}" == "2.000+0.000j"
    test_points = np.array([0, 2 - 1j, 4 + 1j, 5 - 1j, 2 + 2j])
    inside = np.abs(test_points - (2 - 1j)) < 3
    on_circle = np.isclose(np.abs(test_points - (2 - 1j)), 3)
    assert inside.tolist() == [True, True, True, False, False]
    assert on_circle.tolist() == [False, False, False, True, True]

    fig = new_figure("The complex plane: addition translates, $|z - w|$ is distance, and $|z + w| \\leq |z| + |w|$",
                     r"$a + bi \leftrightarrow (a, b)$.   $d(z, w) = |z - w|$.   "
                     r"$\left|\,|z| - |w|\,\right| \leq |z \pm w| \leq |z| + |w|$.   "
                     r"$|z - c| = r$: circle;  $|z - a| = |z - b|$: perpendicular bisector.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    zp, wp = 1 + 3j, 4 + 1j
    ax.plot([zp.real, (zp + wp).real, wp.real], [zp.imag, (zp + wp).imag, wp.imag], color=MUTED,
            linestyle="--", linewidth=1.1)
    arrow(ax, 0, zp, BLUE, label="z = 1 + 3i")
    arrow(ax, 0, wp, ORANGE, label="w = 4 + i")
    arrow(ax, 0, zp + wp, AQUA, width=3, label="z + w = 5 + 4i")
    arrow(ax, zp, wp, INK, width=1.6, label="w − z = 3 − 2i (from z to w)")
    argand(ax, (-1, 6.5), (-3.2, 5), "Parallelogram rule")
    legend(ax, loc="lower right", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    maps = [("z", FLAG, BLUE, BLUE_LIGHT), ("z̄: mirror in Re axis", np.conj(FLAG), ORANGE, ORANGE_LIGHT),
            ("−z: through 0", -FLAG, AQUA, AQUA_LIGHT), ("−z̄: mirror in Im axis", -np.conj(FLAG), INK2, GRID)]
    for label, shape, color, light in maps:
        fill_shape(ax, shape, color, light)
        ax.scatter([shape[0].real], [shape[0].imag], color=color, s=28, zorder=4)
        assert np.allclose(np.abs(shape), np.abs(FLAG))
        x0 = shape[0].real
        ax.text(1.75 * np.sign(x0), 3.05 * np.sign(shape[1].imag), label, color=color, fontsize=9.5,
                va="center", ha="center")
    argand(ax, (-3.6, 3.6), (-3.6, 3.6), "Three reflections, all keep |z|")

    ax = fig.add_subplot(gs[0, 2])
    zt, wt = 3 + 1j, 0.5 + 2j
    angle_between = np.linspace(-np.pi, np.pi, 400)
    rotated = abs(wt) * np.exp(1j * (np.angle(zt) + angle_between))
    lengths = np.abs(zt + rotated)
    upper, lower = abs(zt) + abs(wt), abs(abs(zt) - abs(wt))
    assert np.all(lengths <= upper + 1e-12) and np.all(lengths >= lower - 1e-12)
    ax.plot(angle_between, lengths, color=BLUE, linewidth=2.4, label="|z + w| as w turns")
    ax.axhline(upper, color=ORANGE, linestyle="--", label=f"|z| + |w| = {upper:.2f}")
    ax.axhline(lower, color=AQUA, linestyle="--", label=f"||z| − |w|| = {lower:.2f}")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    ax.set_ylim(0, 6.4)
    ax.grid(True)
    ax.set_xlabel("angle between w and z")
    ax.set_ylabel("length")
    ax.set_title("z = 3 + i, |w| fixed: the bounds are sharp")
    legend(ax, loc="upper right", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(-2, 6, 400)
    Xg, Yg = np.meshgrid(xs, xs - 3.5)
    Zg = Xg + 1j * Yg
    ax.contourf(Xg, Yg, (np.abs(Zg - (2 - 1j)) <= 3).astype(float), levels=[0.5, 1.5], colors=[BLUE_LIGHT],
                alpha=0.6)
    unit_circle(ax, radius=3, center=2 - 1j, color=BLUE, linewidth=2)
    colors = [AQUA if i else (ORANGE if o else INK2) for i, o in zip(inside, on_circle)]
    ax.scatter(test_points.real, test_points.imag, color=colors, s=55, zorder=5)
    for p, name in zip(test_points, ["0", "2 − i", "4 + i", "5 − i", "2 + 2i"]):
        ax.text(p.real + 0.15, p.imag + 0.15, name, fontsize=9.5)
    ax.plot([], [], "o", color=AQUA, label="|z − (2 − i)| < 3")
    ax.plot([], [], "o", color=ORANGE, label="on the circle (= 3)")
    argand(ax, (-1.7, 6), (-4.5, 2.9), "Disc |z − (2 − i)| ≤ 3: a mask test")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-3, 3, 400)
    Xg, Yg = np.meshgrid(xs, xs)
    Zg = Xg + 1j * Yg
    ax.contourf(Xg, Yg, (np.abs(Zg - 1) < np.abs(Zg - 1j)).astype(float), levels=[0.5, 1.5],
                colors=[ORANGE_LIGHT], alpha=0.7)
    annulus = (np.abs(Zg + 1.2 + 1.2j) > 0.5) & (np.abs(Zg + 1.2 + 1.2j) < 1.2)
    ax.contourf(Xg, Yg, annulus.astype(float), levels=[0.5, 1.5], colors=[AQUA_LIGHT], alpha=0.8)
    ax.plot([-3, 3], [-3, 3], color=ORANGE, linestyle="--", linewidth=1.6)
    ax.scatter([1, 0], [0, 1], color=INK, s=35, zorder=5)
    ax.text(1.12, -0.35, "1", fontsize=11)
    ax.text(0.12, 1.1, "i", fontsize=11)
    ax.text(1.4, -2.6, "|z − 1| < |z − i|", color=ORANGE, fontsize=10)
    ax.text(-2.9, 1.3, "annulus\n0.5 < |z + 1.2 + 1.2i| < 1.2", color=AQUA, fontsize=9.5)
    argand(ax, (-3, 3), (-3, 3), "Half-plane (bisector y = x) and annulus")

    panel(fig.add_subplot(gs[1, 2]), "Proof and use of the inequality", [
        r"$|z + w|^2 = |z|^2 + 2\,\mathrm{Re}(z\bar w) + |w|^2$",
        r"$\mathrm{Re}(z\bar w) \leq |z|\,|w|$  ⇒  $|z + w| \leq |z| + |w|$",
        "Equality iff w is a non-negative multiple of z.",
        "",
        "1,000,000 random pairs (seed 0): both bounds",
        f"hold, |z+w| / (|z|+|w|) from {ratio.min():.4f} to {ratio.max():.6f}",
        "",
        "|z| ≤ 2: |z² + 3z − 1| ≤ 4 + 6 + 1 = 11,",
        "a safe bound, not the maximum:",
        f"grid search: true max {values[index]:.4f} at z = 2",
    ], spacing=0.085)
    save_00(fig, name="04_the_complex_plane.png")


# ================================================================ module 01, notebook 01
def sheet_01_01_polar_form_modulus_and_argument():
    examples = [1 + 1j, -1 + math.sqrt(3) * 1j, -2 - 2j, 3j, -4, -1j, 2 + 11j]
    polar = [cmath.polar(complex(z)) for z in examples]
    assert [f"{r:.6f}" for r, _ in polar] == ["1.414214", "2.000000", "2.828427", "3.000000",
                                              "4.000000", "1.000000", "11.180340"]
    assert [f"{t / math.pi:.6f}" for _, t in polar] == ["0.250000", "0.666667", "-0.750000", "0.500000",
                                                        "1.000000", "-0.500000", "0.442751"]
    quadrant = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    naive = [math.atan(q.imag / q.real) for q in quadrant]
    correct = [math.atan2(q.imag, q.real) for q in quadrant]
    assert [f"{v:.4f}" for v in naive] == ["0.7854", "-0.7854", "0.7854", "-0.7854"]
    assert [f"{v:.4f}" for v in correct] == ["0.7854", "2.3562", "-2.3562", "-0.7854"]
    assert cmath.phase(complex(-1, 0.0)) == math.pi and cmath.phase(complex(-1, -0.0)) == -math.pi
    assert sp.arg(2 + 11 * sp.I) == sp.atan(sp.Rational(11, 2))

    fig = new_figure(r"Polar form: a distance $r$ and a direction $\theta$ (use atan2, not $\arctan(b/a)$)",
                     r"$z = r(\cos\theta + i\sin\theta)$,   $r = |z| = \sqrt{a^2 + b^2}$,   "
                     r"$\mathrm{Arg}\,z = \mathrm{atan2}(b, a) \in (-\pi, \pi]$,   "
                     r"$\arg z = \{\mathrm{Arg}\,z + 2\pi k\}$")
    gs = grid(fig, top=0.84, wspace=0.3, right=0.94)

    ax = fig.add_subplot(gs[0, 0])
    numbers = [("1 + i", 1 + 1j, BLUE, (0.15, 0.25)), ("−1 + √3 i", -1 + np.sqrt(3) * 1j, ORANGE, (-0.3, 0.35)),
               ("−2 − 2i", -2 - 2j, AQUA, (0.2, -0.45)), ("2 − i", 2 - 1j, INK2, (-0.15, -0.6))]
    for k, (label, zz, color, (dx, dy)) in enumerate(numbers):
        r, theta = np.abs(zz), np.angle(zz)
        arrow(ax, 0, zz, color)
        arc = (0.4 + 0.2 * k) * np.exp(1j * np.linspace(0, theta, 60))
        ax.plot(arc.real, arc.imag, color=color, linewidth=1.6)
        ax.text(zz.real + dx, zz.imag + dy, f"{label}\nr = {r:.3f}, θ = {theta / np.pi:.3f}π".replace("-", "−"), color=color,
                fontsize=9, ha="center", va="center")
    argand(ax, (-3.5, 3.5), (-3.3, 3.3), "Arrow length r, arc = principal Arg")

    extent = [-2, 2, -2, 2]
    g = np.linspace(-2, 2, 401)
    a, b = np.meshgrid(g, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        naive_map = np.arctan(b / a)
    for col, (values, title) in enumerate([(np.arctan2(b, a), "Colour = atan2(b, a): one full sweep"),
                                           (naive_map, "Colour = arctan(b/a): only half")]):
        ax = fig.add_subplot(gs[0, col + 1])
        image = ax.imshow(values, extent=extent, origin="lower", cmap="twilight_shifted",
                          vmin=-np.pi, vmax=np.pi)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(title)
        for q, t, txt in zip(quadrant, correct if col == 0 else naive, ["1+i", "−1+i", "−1−i", "1−i"]):
            ax.scatter([q.real], [q.imag], color=INK, s=18, zorder=4)
            ax.text(q.real, q.imag + 0.22, f"{t:+.2f}".replace("-", "−"), ha="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.8))
        if col == 0:
            ax.plot([-2, 0], [0, 0], color=ORANGE, linewidth=3)
            ax.text(-1.95, -0.35, "cut: jump π → −π", color=ORANGE, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.8))
        else:
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                                ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar.ax.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
            cbar.outline.set_visible(False)

    ax = fig.add_subplot(gs[1, 0:2])
    t = np.linspace(0, 4 * np.pi, 800)
    principal = np.angle(np.exp(1j * t))
    continuous = np.unwrap(principal)
    assert np.allclose(continuous, t)
    ax.plot(t, principal, color=BLUE, linewidth=2.2, label=r"principal $\mathrm{Arg}\,z(t) \in (-\pi, \pi]$")
    ax.plot(t, continuous, color=ORANGE, linestyle="--", linewidth=2,
            label=r"np.unwrap: a continuous choice from the set $\arg z(t)$")
    ax.set_yticks([-np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi], ["−π", "0", "π", "2π", "3π", "4π"])
    ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi], ["0", "π", "2π", "3π", "4π"])
    ax.grid(True)
    ax.set_xlabel(r"t   (z(t) = cos t + i sin t, two laps of the unit circle)")
    ax.set_ylabel("angle")
    ax.set_title("Arg jumps by 2π at the negative real axis; arg does not")
    legend(ax, loc="upper left")

    panel(fig.add_subplot(gs[1, 2]), "atan(b/a) vs atan2(b, a)", [
        "`z      atan(b/a)  atan2(b,a)`",
        "`1+i      0.7854     0.7854  ok`",
        "`-1+i    -0.7854     2.3562  wrong`",
        "`-1-i     0.7854    -2.3562  wrong`",
        "`1-i     -0.7854    -0.7854  ok`",
        "",
        "−1 + √3 i: arctan gives −π/3, Arg = 2π/3.",
        "Arg(−4) = π (not −π); z = 0 has no argument.",
        "Bombelli's 2 + 11i: r = 5√5 ≈ 11.18,",
        "Arg = atan(11/2) ≈ 1.3909 (about 79.7°)",
    ], spacing=0.085)
    save_01(fig, name="01_polar_form_modulus_and_argument.png")


# ================================================================ module 01, notebook 02
HOUSE_12 = np.array([1 + 0j, 2 + 0j, 2 + 1j, 1.5 + 1.6j, 1 + 1j])


def sheet_01_02_multiplication_is_rotation_and_scaling():
    rng = np.random.default_rng(seed=1)
    zs = rng.normal(size=6) + 1j * rng.normal(size=6)
    ws = rng.normal(size=6) + 1j * rng.normal(size=6)
    product = zs * ws
    assert np.allclose(np.abs(product), np.abs(zs) * np.abs(ws))
    sum_of_args = np.angle(zs) + np.angle(ws)
    offsets = np.round((np.angle(product) - sum_of_args) / (2 * np.pi), 12)
    assert offsets.tolist() == [1, 0, -1, 0, 0, 0]
    prod = sp.expand((1 + sp.I) * (sp.sqrt(3) + sp.I))
    assert sp.simplify(sp.Abs(prod)) == 2 * sp.sqrt(2) and sp.simplify(sp.arg(prod)) == 5 * sp.pi / 12
    assert sp.simplify(sp.radsimp(sp.re(prod) / sp.Abs(prod)) - (sp.sqrt(6) - sp.sqrt(2)) / 4) == 0
    quotient = cmath.rect(4, 5 * math.pi / 6) / cmath.rect(2, math.pi / 3)
    assert abs(quotient - 2j) < 1e-12
    u = np.exp(1j * np.pi / 3)
    assert np.isclose(u * (3 + 1j), (3 - np.sqrt(3)) / 2 + 1j * (1 + 3 * np.sqrt(3)) / 2)
    assert np.isclose(1 + np.exp(1j * np.pi / 2) * ((2 + 1j) - 1), 1j)

    fig = new_figure("Multiplying by $w$ rotates the plane by Arg $w$ and scales it by $|w|$",
                     r"$r(\cos\alpha + i\sin\alpha)\cdot s(\cos\beta + i\sin\beta) = rs\,(\cos(\alpha+\beta) + i\sin(\alpha+\beta))$:"
                     r"   moduli multiply, arguments add (the angle-addition formulas).")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    zz, ww = 1 + 1j, np.sqrt(3) + 1j
    pp = zz * ww
    for label, number, color, radius, (dx, dy) in [("z = 1 + i", zz, BLUE, 0.55, (0.05, 0.3)),
                                                   ("w = √3 + i", ww, ORANGE, 0.85, (0.1, -0.45)),
                                                   ("zw", pp, AQUA, 1.15, (0.12, 0.05))]:
        arrow(ax, 0, number, color, width=2.4)
        arc = radius * np.exp(1j * np.linspace(0, np.angle(number), 60))
        ax.plot(arc.real, arc.imag, color=color, linewidth=1.6)
        ax.text(number.real + dx, number.imag + dy,
                f"{label}\n|·| = {abs(number):.3f}, θ = {np.degrees(np.angle(number)):.0f}°",
                color=color, fontsize=9.5)
    argand(ax, (-0.5, 3.6), (-0.5, 3.4), "|zw| = √2 · 2,   arg zw = 45° + 30° = 75°")

    ax = fig.add_subplot(gs[0, 1])
    zt, wt = 1.2 + 0.9j, 0.8 + 1.1j
    tri = np.array([0, 1, wt])
    fill_shape(ax, tri, BLUE, BLUE_LIGHT, label="triangle 0, 1, w")
    fill_shape(ax, zt * tri, ORANGE, ORANGE_LIGHT, label="z × triangle = 0, z, zw")
    for point, label in [(1, "1"), (wt, "w"), (zt, "z"), (zt * wt, "zw")]:
        ax.scatter([point.real], [point.imag], color=INK, s=22, zorder=5)
        ax.text(point.real + 0.06, point.imag + 0.06, label, fontsize=12)
    argand(ax, (-0.8, 2.0), (-0.4, 2.4), "Similar triangles: the product by construction")
    legend(ax, loc="lower right")

    ax = fig.add_subplot(gs[0, 2])
    multipliers = [("w = 1", 1, BLUE, BLUE_LIGHT), ("w = i", 1j, ORANGE, ORANGE_LIGHT),
                   ("w = −1", -1, AQUA, AQUA_LIGHT), ("w = 0.5 − 0.5i", 0.5 - 0.5j, INK2, GRID),
                   ("w = 1 + i", 1 + 1j, INK, "#d6d5d0")]
    for label, wm, color, light in multipliers:
        fill_shape(ax, wm * HOUSE_12, color, light,
                   label=f"{label}: {np.degrees(np.angle(wm)):.0f}°, ×{abs(wm):.3f}".replace("-", "−"))
    ax.scatter([0], [0], color=INK, s=20, zorder=5)
    argand(ax, (-2.5, 6.6), (-3.4, 3.4), "z ↦ wz moves the whole house")
    legend(ax, loc="lower right", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    line = np.linspace(-2 * np.pi, 2 * np.pi, 10)
    for k, style in [(0, "-"), (-1, "--"), (1, "--")]:
        ax.plot(line, line + 2 * np.pi * k, color=INK2 if k == 0 else MUTED, linestyle=style, linewidth=1.2)
    colors = {1.0: ORANGE, 0.0: BLUE, -1.0: AQUA}
    for s_, a_, off in zip(sum_of_args, np.angle(product), offsets):
        ax.scatter([s_], [a_], color=colors[abs(off) and off or 0.0], s=70, zorder=4)
    ax.axhspan(-np.pi, np.pi, color=BLUE_LIGHT, alpha=0.25, zorder=0)
    ax.plot([], [], "o", color=BLUE, label="Arg(zw) = Arg z + Arg w  (4 pairs)")
    ax.plot([], [], "o", color=ORANGE, label="off by +2π")
    ax.plot([], [], "o", color=AQUA, label="off by −2π")
    ticks = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]
    labels = ["−2π", "−π", "0", "π", "2π"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-2 * np.pi, 2 * np.pi)
    ax.set_ylim(-2 * np.pi, 2 * np.pi)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("Arg z + Arg w")
    ax.set_ylabel("Arg(zw)")
    ax.set_title("6 random pairs (seed 1): equal mod 2π")
    legend(ax, loc="upper left", fontsize=8.5)

    ax = fig.add_subplot(gs[1, 1])
    xs, ys = np.meshgrid(np.linspace(0.3, 2, 8), np.linspace(0.2, 1.2, 5))
    pts = (xs + 1j * ys).ravel()
    for power, color, label in [(0, BLUE, "z"), (1, ORANGE, "iz: quarter turn"), (2, AQUA, "i²z = −z: half turn"),
                                (3, INK2, "i³z = −iz")]:
        img = (1j ** power) * pts
        ax.scatter(img.real, img.imag, s=14, color=color, zorder=3, label=label)
    corner = 2 + 1.2j
    for power in range(4):
        ax.annotate("", xy=((1j ** (power + 1) * corner).real, (1j ** (power + 1) * corner).imag),
                    xytext=((1j ** power * corner).real, (1j ** power * corner).imag),
                    arrowprops=dict(arrowstyle="->", color=MUTED, connectionstyle="arc3,rad=0.35"))
    argand(ax, (-2.6, 2.6), (-3.3, 2.6), "Why i² = −1: two quarter turns")
    legend(ax, loc="lower left", fontsize=8, ncol=2, bbox_to_anchor=(-0.02, -0.02))

    panel(fig.add_subplot(gs[1, 2]), "Three readings of one rule", [
        "(1 + i)(√3 + i) = (√3 − 1) + (√3 + 1)i",
        "        = 2√2 (cos 5π/12 + i sin 5π/12)",
        "⇒ cos 75° = (√6 − √2)/4 for free",
        "",
        "Divide: 4∠150° / 2∠60° = 2∠90° = 2i",
        "Rotate about c: z ↦ c + u(z − c), |u| = 1",
        "Matrix: w = c + di acts as [[c, −d], [d, c]]",
        "",
        "Arg(zw) = Arg z + Arg w only mod 2π:",
        "(−1 + i)·(−1) = 1 − i, Arg −π/4, not 7π/4",
    ], spacing=0.08)
    save_01(fig, name="02_multiplication_is_rotation_and_scaling.png")


# ================================================================ module 01, notebook 03
def exp_partial_sum(z, n_terms):
    total = 0
    term = 1 + 0j
    for n in range(n_terms):
        total += term
        term = term * z / (n + 1)
    return total


def integrate_ode(step_function, t_end, n_steps, keep_path=False):
    h = t_end / n_steps
    f = 1 + 0j
    path = [f]
    for _ in range(n_steps):
        f = step_function(f, h)
        path.append(f)
    return np.array(path) if keep_path else f


def forward_euler_step(f, h):
    return f + h * (1j * f)


def midpoint_step(f, h):
    half = f + 0.5 * h * (1j * f)
    return f + h * (1j * half)


def sheet_01_03_eulers_formula():
    theta = 2.0
    target = complex(math.cos(theta), math.sin(theta))
    n_list = [1, 2, 3, 4, 6, 8, 12, 16, 20]
    errors = [abs(exp_partial_sum(1j * theta, n) - target) for n in n_list]
    assert [f"{e:.2e}" for e in errors] == ["1.68e+00", "1.79e+00", "1.24e+00", "6.32e-01", "8.62e-02",
                                           "6.23e-03", "8.47e-06", "3.11e-09", "4.29e-13"]
    all_n = np.arange(1, 25)
    all_errors = np.array([abs(exp_partial_sum(1j * theta, int(n)) - target) for n in all_n])
    t_end = 2 * np.pi
    exact = np.exp(1j * t_end)
    euler_moduli = {n: abs(integrate_ode(forward_euler_step, t_end, n)) for n in (20, 200, 2000)}
    midpoint_moduli = {n: abs(integrate_ode(midpoint_step, t_end, n)) for n in (20, 200, 2000)}
    assert [f"{euler_moduli[n]:.6f}" for n in (20, 200, 2000)] == ["2.563159", "1.103677", "1.009918"]
    assert [f"{midpoint_moduli[n]:.8f}" for n in (20, 200, 2000)] == ["1.02462088", "1.00002435", "1.00000002"]
    assert f"{abs(integrate_ode(midpoint_step, t_end, 20) - exact):.2e}" == "1.04e-01"
    value = cmath.exp(1j * cmath.pi)
    assert value != -1 and abs(value + 1) < 1e-12 and sp.exp(sp.I * sp.pi) + 1 == 0
    assert abs((1j) ** 1j - math.exp(-math.pi / 2)) < 1e-15
    lhs = ((1 + 1j) ** 6) / ((math.sqrt(3) + 1j) ** 2)
    assert abs(lhs - (-math.sqrt(3) - 1j)) < 1e-12

    fig = new_figure(r"Euler's formula: $e^{i\theta} = \cos\theta + i\sin\theta$ walks the unit circle",
                     r"$e^{i\theta} = \sum (i\theta)^n/n!$ splits into the cos and sin series;   "
                     r"$f' = if,\ f(0) = 1$;   $e^{i\pi} + 1 = 0$;   "
                     r"$\cos\theta = \dfrac{e^{i\theta} + e^{-i\theta}}{2}$,  $\sin\theta = \dfrac{e^{i\theta} - e^{-i\theta}}{2i}$")
    gs = grid(fig, top=0.81)

    ax = fig.add_subplot(gs[0, 0])
    n_terms = 12
    terms = [(1j * theta) ** n / math.factorial(n) for n in range(n_terms)]
    partial = np.concatenate([[0], np.cumsum(terms)])
    unit_circle(ax, color=MUTED, linewidth=1.1)
    ramp = [BLUE, ORANGE, AQUA, INK2]
    for n in range(n_terms):
        arrow(ax, partial[n], partial[n + 1], ramp[n % 4], width=2 if n < 6 else 1.4)
        if n < 6:
            ax.text(partial[n + 1].real + 0.06, partial[n + 1].imag + 0.08, f"$S_{n}$", fontsize=10,
                    color=ramp[n % 4])
    ax.scatter([target.real], [target.imag], color=ORANGE, s=80, zorder=6, marker="*")
    ax.annotate(r"$e^{2i} = \cos 2 + i\sin 2$", xy=(target.real, target.imag), xytext=(-2.45, 1.45),
                fontsize=11, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2))
    argand(ax, (-2.6, 1.6), (-0.9, 3.0), r"Partial sums of $\sum (2i)^n/n!$: each term turns 90°")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(all_n, all_errors, color=BLUE, linewidth=2, marker="o", markersize=4, zorder=3)
    ax.scatter(n_list, errors, color=ORANGE, s=40, zorder=4, label="printed in the notebook")
    for n, e in zip(n_list, errors):
        if n in (4, 8, 12, 16, 20):
            ax.text(n + 0.4, e * 1.8, f"{e:.1e}", fontsize=9, color=INK2)
    ax.axvline(theta, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(theta + 0.3, 1e-14, "N = θ", fontsize=9, color=INK2)
    ax.set_xlabel("number of terms N")
    ax.set_ylabel(r"$|S_N - (\cos 2 + i\sin 2)|$")
    ax.set_ylim(1e-16, 20)
    ax.grid(True)
    ax.set_title("Faster than geometric once N > θ")
    legend(ax, loc="upper right")

    panel(fig.add_subplot(gs[0, 2]), "What the formula buys", [
        r"$e^{i\pi} + 1 = 0$: half a turn lands on −1.",
        f"`cmath.exp(1j*pi) + 1 = {(value + 1).imag:.2e}j`",
        "so compare with a tolerance, never ==.",
        "",
        r"$z = re^{i\theta}$: exponent laws do the polar rules",
        "",
        r"$\dfrac{(1+i)^6}{(\sqrt{3}+i)^2} = \dfrac{8e^{3\pi i/2}}{4e^{i\pi/3}} = 2e^{7\pi i/6} = -\sqrt{3} - i$",
        "",
        "",
        r"$i^i = e^{i\cdot i\pi/2} = e^{-\pi/2} \approx 0.2079$, a real number",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    thetas = np.linspace(0, 2 * np.pi, 600)
    shades = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", ORANGE, AQUA]
    for n_t, color in zip([2, 4, 6, 8, 12, 20], shades):
        curve = sum((1j * thetas) ** n / math.factorial(n) for n in range(n_t))
        ax.plot(curve.real, curve.imag, color=color, linewidth=1.8, label=f"N = {n_t}")
    ax.plot(UNIT.real, UNIT.imag, color=INK, linestyle="--", linewidth=1.1, label=r"$e^{i\theta}$")
    argand(ax, (-3, 3), (-3, 3), r"$\theta \mapsto S_N(i\theta)$ on $[0, 2\pi]$: N = 20 is the circle")
    legend(ax, loc="upper left", fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    unit_circle(ax, color=INK, linestyle="--", linewidth=1.1)
    euler_path = integrate_ode(forward_euler_step, t_end, 20, keep_path=True)
    mid_path = integrate_ode(midpoint_step, t_end, 20, keep_path=True)
    ax.plot(euler_path.real, euler_path.imag, color=ORANGE, marker="o", markersize=3.5, linewidth=1.6,
            label=f"forward Euler, 20 steps: |f| = {euler_moduli[20]:.3f}")
    ax.plot(mid_path.real, mid_path.imag, color=BLUE, marker="o", markersize=3.5, linewidth=1.6,
            label=f"midpoint, 20 steps: |f| = {midpoint_moduli[20]:.3f}")
    for t_ in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        p = np.exp(1j * t_)
        arrow(ax, p, p + 0.45 * 1j * p, AQUA, width=1.4)
    ax.plot([], [], color=AQUA, label="velocity f' = if ⊥ position")
    argand(ax, (-2.8, 2.8), (-2.9, 3.6), r"$f' = if$: velocity ⊥ position ⇒ a circle")
    legend(ax, loc="upper left", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    th = 1.0
    fwd, bwd = np.exp(1j * th), np.exp(-1j * th)
    unit_circle(ax, color=MUTED, linewidth=1.1)
    ax.plot([fwd.real, (fwd + bwd).real, bwd.real], [fwd.imag, 0, bwd.imag], color=MUTED, linestyle=":")
    arrow(ax, 0, fwd, BLUE, label=r"$e^{i\theta}$ (counter-clockwise)")
    arrow(ax, 0, bwd, ORANGE, label=r"$e^{-i\theta}$ (clockwise)")
    arrow(ax, 0, fwd + bwd, AQUA, width=3, label=rf"sum $= 2\cos 1 = {2 * math.cos(th):.3f}$")
    assert abs((fwd + bwd).imag) < 1e-15
    argand(ax, (-1.3, 3.3), (-1.3, 1.3), r"$\theta = 1$: imaginary parts cancel")
    legend(ax, loc="upper right", fontsize=8.5)
    save_01(fig, name="03_eulers_formula.png")


# ================================================================ module 01, notebook 04
def power_de_moivre(z, n):
    r, theta = cmath.polar(z)
    return cmath.rect(r ** n, n * theta)


def sheet_01_04_de_moivres_theorem():
    cases = [(1 + 1j, 20), (1 - math.sqrt(3) * 1j, 7), (math.sqrt(3) + 1j, -4), (0.9 + 0.3j, 50)]
    printed = ["-1024.000000+0.000000j", "64.000000-110.851252j", "-0.031250-0.054127j", "-0.066680-0.026599j"]
    for (zz, n), text in zip(cases, printed):
        assert f"{power_de_moivre(zz, n):.6f}" == text and abs(power_de_moivre(zz, n) - zz ** n) < 1e-9 * max(
            1, abs(zz ** n))
    assert sp.expand((1 + sp.I) ** 20) == -1024
    assert sp.expand((1 - sp.sqrt(3) * sp.I) ** 7) == 64 - 64 * sp.sqrt(3) * sp.I
    c, s = sp.symbols("c s", real=True)
    for n in range(1, 7):
        from_dm = sp.expand(sp.re(sp.expand((c + sp.I * s) ** n)).subs(s ** 2, 1 - c ** 2))
        assert sp.simplify(from_dm - sp.expand(sp.chebyshevt(n, c))) == 0
    theta = sp.symbols("theta", real=True)
    assert sp.integrate(sp.sin(theta) ** 4, (theta, 0, 2 * sp.pi)) == 3 * sp.pi / 4
    assert sp.integrate(sp.cos(theta) ** 4, (theta, 0, sp.pi)) == 3 * sp.pi / 8
    powers_1i = [(1 + 1j) ** n for n in range(9)]
    assert np.allclose(powers_1i, [1, 1 + 1j, 2j, -2 + 2j, -4, -4 - 4j, -8j, 8 - 8j, 16])

    fig = new_figure(r"De Moivre: powers multiply the angle, $(\cos\theta + i\sin\theta)^n = \cos n\theta + i\sin n\theta$",
                     r"$(re^{i\theta})^n = r^n e^{in\theta}$ for every integer $n$.   Expand $(c + is)^n$ binomially:  "
                     r"$\cos n\theta = T_n(\cos\theta)$.   $z^n$ spirals out ($r > 1$), circles ($r = 1$), in ($r < 1$).")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    angle = np.pi / 8
    n_values = np.arange(0, 25)
    smooth = np.linspace(0, 24, 800)
    for modulus, color, label in [(1.08, ORANGE, "|z| = 1.08: outward"), (1.0, INK2, "|z| = 1: on the circle"),
                                  (0.92, BLUE, "|z| = 0.92: inward")]:
        zz = modulus * np.exp(1j * angle)
        spiral = modulus ** smooth * np.exp(1j * angle * smooth)
        ax.plot(spiral.real, spiral.imag, color=color, linewidth=1, alpha=0.5)
        pts = zz ** n_values
        ax.scatter(pts.real, pts.imag, color=color, s=14, zorder=3, label=label)
    argand(ax, (-6.8, 6.8), (-6.8, 6.8), r"$z^n$, $n \leq 24$, Arg z = π/8 (22.5° per step)")
    legend(ax, loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    shades = ["#cde2fb", "#b7d3f6", "#86b6ef", "#5f9fe9", "#3987e5", "#2a78d6", "#1c5cab", "#14477f", ORANGE]
    for n, (pw, color) in enumerate(zip(powers_1i, shades)):
        arrow(ax, 0, pw, color, width=2 if n < 8 else 2.8)
        offset = 0.9 * pw / abs(pw)
        ax.text(pw.real + offset.real, pw.imag + offset.imag, f"n={n}", fontsize=9, ha="center", va="center",
                color=INK2 if n < 8 else ORANGE)
    argand(ax, (-6, 18), (-10, 6), r"$(1 + i)^n$: turn 45°, grow ×√2 each step")
    ax.text(17.5, 5.5, "(1 + i)⁴ = −4,  (1 + i)⁸ = 16\n" + r"$(1 + i)^{20} = 2^{10}e^{5\pi i} = -1024$",
            ha="right", va="top", fontsize=9.5, color=INK2)

    panel(fig.add_subplot(gs[0, 2]), "Large powers in one line", [
        r"$(1 - \sqrt{3}i)^7 = (2e^{-i\pi/3})^7 = 128\,e^{-7\pi i/3}$",
        r"$\quad = 128\,e^{-i\pi/3} = 64 - 64\sqrt{3}\,i$",
        r"$(\sqrt{3} + i)^{-4} = 2^{-4}e^{-2\pi i/3} = -1/32 - (\sqrt{3}/32)\,i$",
        "",
        "`(0.9+0.3j)**50 = -0.066680-0.026599j`",
        "(same via polar: r⁵⁰ and 50θ)",
        "",
        "Integer n only: n = 1/2 has two candidate",
        "values, n = 1/3 three (next notebook).",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    th = np.linspace(0, np.pi, 40)
    x = np.linspace(-1, 1, 400)
    for n, color in [(3, BLUE), (4, ORANGE), (5, AQUA)]:
        tn = sp.lambdify(c, sp.chebyshevt(n, c), "numpy")
        ax.plot(x, tn(x), color=color, linewidth=2, label=f"$T_{n}(x)$")
        ax.scatter(np.cos(th), np.cos(n * th), color=color, s=12, zorder=3)
        assert np.allclose(tn(np.cos(th)), np.cos(n * th))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel(r"$x = \cos\theta$")
    ax.set_ylabel(r"$\cos n\theta$")
    ax.grid(True)
    ax.set_ylim(-1.15, 1.6)
    ax.set_title(r"Dots $(\cos\theta, \cos n\theta)$ lie on $T_n$")
    legend(ax, loc="upper center", ncol=3)

    ax = fig.add_subplot(gs[1, 1])
    thetas = np.linspace(0, np.pi / 2, 200)
    arc = np.exp(1j * thetas)
    unit_circle(ax, color=GRID)
    ax.plot(arc.real, arc.imag, color=BLUE, linewidth=6, alpha=0.55, label=r"$\theta \in [0, \pi/2]$")
    ax.plot((arc ** 3).real, (arc ** 3).imag, color=ORANGE, linewidth=2.5, label=r"image under $z^3$: angle $3\theta$")
    for degrees, symbol in [(20, "A"), (45, "B"), (80, "C")]:
        p = np.exp(1j * np.radians(degrees))
        q = p ** 3
        assert np.isclose(np.degrees(np.angle(q)) % 360, 3 * degrees)
        ax.scatter([p.real, q.real], [p.imag, q.imag], color=[BLUE, ORANGE], s=30, zorder=4)
        ax.text(1.32 * p.real, 1.25 * p.imag, f"{symbol}: {degrees}°", color=BLUE, ha="center", va="center",
                fontsize=9)
        ax.text(1.3 * q.real, 1.25 * q.imag, f"{symbol}³: {3 * degrees}°", color=ORANGE, ha="center",
                va="center", fontsize=9)
    argand(ax, (-1.6, 1.6), (-1.6, 1.6), "On the unit circle, cubing triples the angle")
    legend(ax, loc="lower right", fontsize=8)

    panel(fig.add_subplot(gs[1, 2]), "Identities for free", [
        "(c + is)³ = c³ + 3ic²s − 3cs² − is³:",
        "  cos 3θ = 4cos³θ − 3cos θ",
        "  sin 3θ = 3sin θ − 4sin³θ",
        "  cos 4θ = 8cos⁴θ − 8cos²θ + 1",
        "",
        r"Backwards, via $\cos\theta = (e^{i\theta} + e^{-i\theta})/2$:",
        "  cos³θ = (cos 3θ + 3cos θ)/4",
        "  sin⁴θ = (cos 4θ − 4cos 2θ + 3)/8",
        r"  ⇒ $\int_0^{2\pi} \sin^4\theta\, d\theta = 3\pi/4$",
    ], spacing=0.085)
    save_01(fig, name="04_de_moivres_theorem.png")


# ================================================================ module 01, notebook 05
def nth_roots(w, n):
    R, phi = cmath.polar(complex(w))
    return R ** (1 / n) * np.exp(1j * (phi + 2 * np.pi * np.arange(n)) / n)


def sheet_01_05_roots_of_unity_and_nth_roots():
    for n in (2, 3, 4, 6):
        roots = np.exp(2j * np.pi * np.arange(n) / n)
        assert abs(roots.sum()) < 1e-15 and np.max(np.abs(roots ** n - 1)) < 1e-14
    roots_m16 = nth_roots(-16, 4)
    roots_8i = nth_roots(8j, 3)
    roots_bomb = nth_roots(2 + 11j, 3)
    assert np.allclose(roots_m16, np.sqrt(2) * np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]))
    assert np.allclose(roots_8i, [np.sqrt(3) + 1j, -np.sqrt(3) + 1j, -2j])
    assert [f"{r:.6f}" for r in roots_bomb] == ["2.000000+1.000000j", "-1.866025+1.232051j", "-0.133975-2.232051j"]
    for w, n, roots in [(-16, 4, roots_m16), (8j, 3, roots_8i), (2 + 11j, 3, roots_bomb)]:
        assert np.isclose(complex(w) ** (1 / n), roots[0])
    n12 = 12
    omega = np.exp(2j * np.pi / n12)
    orders = []
    for k in range(n12):
        orders.append(next(o for o in range(1, n12 + 1) if abs(omega ** (k * o) - 1) < 1e-9))
    assert orders == [1, 12, 6, 4, 3, 12, 2, 12, 3, 4, 6, 12]
    assert [k for k in range(n12) if math.gcd(k, n12) == 1] == [1, 5, 7, 11] and sp.totient(12) == 4
    zsym = sp.symbols("z")
    assert sp.cyclotomic_poly(12, zsym) == zsym ** 4 - zsym ** 2 + 1
    p = -15
    xs_bomb = np.array([u + (-p / (3 * u)) for u in roots_bomb])
    assert np.allclose(xs_bomb, [4, -2 - np.sqrt(3), -2 + np.sqrt(3)])
    assert np.allclose(-p / (3 * roots_bomb), np.conj(roots_bomb))

    fig = new_figure(r"Every $w \neq 0$ has exactly $n$ $n$-th roots: the vertices of a regular $n$-gon",
                     r"$z_k = R^{1/n} e^{i(\phi + 2\pi k)/n},\ k = 0, \dots, n-1$  for $w = Re^{i\phi}$.   "
                     r"$\omega = e^{2\pi i/n}$: $\sum_k \omega^k = 0$;   $\omega^k$ primitive $\Leftrightarrow \gcd(k, n) = 1$.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    for roots, color, light, label in [(roots_m16, BLUE, BLUE_LIGHT, r"$z^4 = -16$: square, radius 2"),
                                       (nth_roots(1, 5), ORANGE, ORANGE_LIGHT, r"$z^5 = 1$: pentagon on |z| = 1"),
                                       (roots_8i, AQUA, AQUA_LIGHT, r"$z^3 = 8i$: triangle, radius 2")]:
        fill_shape(ax, roots, color, light, alpha=0.35)
        ax.scatter(roots.real, roots.imag, color=color, s=45, zorder=4, label=label)
        ax.scatter([roots[0].real], [roots[0].imag], color=color, s=160, facecolor="none", linewidth=1.6, zorder=5)
    unit_circle(ax, radius=2, color=GRID)
    ax.plot([], [], "o", markerfacecolor="none", markeredgecolor=INK2, markersize=10,
            label="principal root = w ** (1/n)")
    argand(ax, (-2.6, 2.6), (-2.6, 3.4), "n roots, spaced 2π/n, same modulus")
    legend(ax, loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(-0.02, 1.02))

    ax = fig.add_subplot(gs[0, 1])
    n = 8
    roots8 = np.exp(2j * np.pi * np.arange(n) / n)
    positions = np.concatenate([[0], np.cumsum(roots8)])
    assert abs(positions[-1]) < 1e-14
    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#f8c9b3", "#f2a07c", ORANGE, "#c4481a"]
    for k in range(n):
        arrow(ax, positions[k], positions[k + 1], ramp[k], width=2.2)
        mid = (positions[k] + positions[k + 1]) / 2
        out = mid - (positions.mean())
        out = out / abs(out) * 0.28
        ax.text(mid.real + out.real, mid.imag + out.imag, f"$\\omega^{k}$", ha="center", va="center", fontsize=10)
    ax.scatter([0], [0], color=INK, s=40, zorder=6)
    ax.text(-0.12, -0.42, "start = end", fontsize=9, ha="right")
    argand(ax, (-1.6, 2.4), (-0.8, 3.2), "n = 8, head to tail: the chain closes, sum = 0")

    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "Primitive 12th roots", [
        "`k     0  1  2  3  4  5  6  7  8  9 10 11`",
        "`order 1 12  6  4  3 12  2 12  3  4  6 12`",
        "",
        "order = 12 / gcd(k, 12); primitive k = 1, 5, 7, 11",
        "count = φ(12) = 4, roots of Φ₁₂ = z⁴ − z² + 1",
        "z¹² − 1 = Φ₁Φ₂Φ₃Φ₄Φ₆Φ₁₂ (d | 12)",
        "",
        "Sum zero: ωS = S (rotation permutes",
        "the roots) and ω ≠ 1, so S = 0.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    vertices = np.exp(2j * np.pi * np.arange(n12) / n12)
    ax.scatter(vertices.real, vertices.imag, color=MUTED, s=30, zorder=2)
    for k, color, style in [(5, AQUA, "-"), (4, ORANGE, "-"), (3, BLUE, "--")]:
        steps = np.exp(2j * np.pi * k * np.arange(n12 + 1) / n12)
        g = math.gcd(k, n12)
        tag = "primitive, visits all 12" if g == 1 else f"gcd {g}: closes after {n12 // g}"
        ax.plot(steps.real, steps.imag, color=color, linestyle=style, linewidth=1.8, label=f"k = {k}: {tag}")
    argand(ax, (-1.35, 1.35), (-1.35, 1.75), r"Stepping by $\omega^k$ around the 12-gon")
    legend(ax, loc="upper center", fontsize=8, ncol=1, bbox_to_anchor=(0.5, 1.03))

    ax = fig.add_subplot(gs[1, 1])
    colors = [BLUE, ORANGE, AQUA]
    unit_circle(ax, radius=np.sqrt(5), color=GRID)
    tri = np.append(roots_bomb, roots_bomb[0])
    ax.plot(tri.real, tri.imag, color=MUTED, linewidth=0.9)
    for k in range(3):
        u_ = roots_bomb[k]
        ax.scatter([u_.real], [u_.imag], color=colors[k], s=60, zorder=4)
        ax.scatter([u_.real], [-u_.imag], color=colors[k], s=60, marker="s", zorder=4)
        ax.plot([u_.real, u_.real], [u_.imag, -u_.imag], color=colors[k], linestyle=":")
        ax.scatter([xs_bomb[k].real], [0], color=colors[k], s=180, marker="*", zorder=5,
                   label=f"$u_{k} + \\bar u_{k} = {xs_bomb[k].real:.4f}$")
        ax.text(u_.real + 0.15, u_.imag + 0.12, f"$u_{k}$", color=colors[k], fontsize=11)
    argand(ax, (-4.2, 4.6), (-3.1, 3.1), r"Cube roots $u_k$ of $2 + 11i$ (circles), $\bar u_k$ (squares)")
    legend(ax, loc="lower left", fontsize=8.5, ncol=1)

    ax = fig.add_subplot(gs[1, 2])
    xc = np.linspace(-4.5, 4.8, 400)
    ax.plot(xc, xc ** 3 - 15 * xc - 4, color=INK2, linewidth=2, label="$x^3 - 15x - 4$")
    for k in range(3):
        ax.scatter([xs_bomb[k].real], [0], color=colors[k], s=180, marker="*", zorder=4)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(-40, 45)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bombelli completed: all three real roots")
    ax.text(4.6, -36, r"$x_k = 2\,\mathrm{Re}\,u_k$:  $4,\ -2 \pm \sqrt{3}$", ha="right", fontsize=10, color=INK2)
    legend(ax, loc="upper left")
    save_01(fig, name="05_roots_of_unity_and_nth_roots.png")


if __name__ == "__main__":
    sheet_00_01_why_complex_numbers_exist()
    sheet_00_02_cartesian_form_and_arithmetic()
    sheet_00_03_conjugate_modulus_and_division()
    sheet_00_04_the_complex_plane()
    sheet_01_01_polar_form_modulus_and_argument()
    sheet_01_02_multiplication_is_rotation_and_scaling()
    sheet_01_03_eulers_formula()
    sheet_01_04_de_moivres_theorem()
    sheet_01_05_roots_of_unity_and_nth_roots()
