#!/usr/bin/env python3
"""Concept sheets for modules/06_complex_integration (notebooks 01-07).

One page per notebook that summarizes its central idea. Every number is computed here from the
same functions, contours, grids and step counts the notebook uses, and is checked against the
notebook's printed outputs with asserts. Images are written to modules/06_complex_integration/img/.

Run from anywhere:  python scripts/make_module_06_concept_images.py
"""
import cmath
import math
from functools import partial

import numpy as np
import sympy as sp
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, CARD, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID, SEQ,
                       new_figure, grid)
from viz_style import save as _save

MODULE = REPO / "modules" / "06_complex_integration"
save = partial(_save, folder=MODULE / "img")

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"
SEQ_R = SEQ.reversed()
# diverging map for signed values: blue (negative) - surface - orange (positive)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", SURFACE, "#f2a07c", "#c4481a"])


# ================================================================ helpers
def trapezoid(values, x):
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return integrate(values, x)


def closed_integral(f, points):
    """Trapezoidal rule on a sampled curve (the helper of notebooks 02 and 06)."""
    values = f(points)
    return np.sum((values[1:] + values[:-1]) / 2 * np.diff(points))


def periodic_circle_integral(f, center, radius, n_points=4000):
    """Periodic trapezoidal rule on |z - center| = radius (notebook 06)."""
    t = 2 * np.pi * np.arange(n_points) / n_points
    z = center + radius * np.exp(1j * t)
    return np.mean(f(z) * 1j * radius * np.exp(1j * t)) * 2 * np.pi


def taylor_coefficients(f, center, radius, n_coefficients, n_points=256):
    t = 2 * np.pi * np.arange(n_points) / n_points
    samples = f(center + radius * np.exp(1j * t))
    n = np.arange(n_coefficients)
    fourier = (samples[None, :] * np.exp(-1j * n[:, None] * t[None, :])).mean(axis=1)
    return fourier / radius**n


def laurent_coefficients(f, center, radius, n_values, n_points=512):
    t = 2 * np.pi * np.arange(n_points) / n_points
    samples = f(center + radius * np.exp(1j * t))
    return {n: (samples * np.exp(-1j * n * t)).mean() / radius**n for n in n_values}


def colour_from_log(log_values):
    """Domain colouring of notebook 05: hue = arg f, brightness bands repeat each time |f| doubles."""
    hue = (log_values.imag / (2 * np.pi)) % 1.0
    bands = (log_values.real / np.log(2)) % 1.0
    value = 0.55 + 0.45 * bands
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.9), value], axis=-1))


def argand(ax, xlim, ylim, title, grid_on=True):
    """Complex plane: equal aspect, both axes through the origin, axes labelled Re and Im."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.grid(grid_on)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)


def image_axes(ax, title):
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)


def arrow(ax, start, end, color, width=2.0, zorder=4, head=14):
    start, end = complex(start), complex(end)
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag), zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width, mutation_scale=head,
                                shrinkA=0, shrinkB=0))


def direction_arrow(ax, points, index, color, step=40, width=2.0, head=14):
    arrow(ax, points[index], points[index + step], color, width=width, head=head)


def colorbar(fig, image, ax, label):
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(label)
    bar.outline.set_visible(False)
    return bar


def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """viz_style.panel, but a line holding a fraction is set in display size and gets more height."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor=CARD, edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5, weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        tall = r"\frac" in text or r"\oint" in text or r"\int" in text or r"\sum" in text
        if r"\frac" in text:
            text = text.replace(r"\frac", r"\dfrac")
        if tall:
            y -= spacing * 0.15
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0), family="monospace" if is_code else None)
        y -= spacing * (1.6 if tall else 1.0)


def new_figure_tall(title, subtitle):
    """new_figure with the subtitle set a little lower, for display-size fractions."""
    fig = new_figure(title, "")
    fig.text(0.5, 0.912, subtitle, ha="center", va="center", fontsize=13, color=INK2)
    return fig


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ================================================================ notebook 01
def sheet_01_contour_integrals():
    def contour_integral(f, gamma, dgamma, t_start, t_end, n_points=4001):
        t = np.linspace(t_start, t_end, n_points)
        return trapezoid(f(gamma(t)) * dgamma(t), t)

    # 4.1
    value = contour_integral(lambda z: 1 / z, lambda t: np.exp(1j * t), lambda t: 1j * np.exp(1j * t), 0, 2 * np.pi)
    assert abs(value - 2j * np.pi) < 1e-12

    # 4.2: only n = -1 survives, on every circle
    for center, radius in [(0, 1), (2 - 1j, 0.5), (-1 + 3j, 4)]:
        for n in range(-3, 4):
            v = contour_integral(lambda z: (z - center) ** n, lambda t: center + radius * np.exp(1j * t),
                                 lambda t: 1j * radius * np.exp(1j * t), 0, 2 * np.pi)
            v = complex(round(v.real, 10) + 0.0, round(v.imag, 10) + 0.0)
            assert str(v) == ("6.2831853072j" if n == -1 else "0j"), (center, n, v)

    # 4.3: exact values of the worked examples
    t_sym = sp.symbols("t", real=True)

    def exact_integral(f_expr, gamma_expr, t_start, t_end):
        integrand = f_expr(gamma_expr) * sp.diff(gamma_expr, t_sym)
        return sp.simplify(sp.integrate(sp.expand(integrand), (t_sym, t_start, t_end)))

    assert exact_integral(lambda z: z, (1 + sp.I) * t_sym, 0, 1) == sp.I
    path_a = exact_integral(sp.conjugate, (1 + sp.I) * t_sym, 0, 1)
    path_b = exact_integral(sp.conjugate, t_sym, 0, 1) + exact_integral(sp.conjugate, 1 + sp.I * t_sym, 0, 1)
    assert path_a == 1 and sp.simplify(path_b - (1 + sp.I)) == 0

    # 4.4: winding
    t = np.linspace(0, 1, 8001)
    windings = []
    for turns in [1, -1, 2, 3]:
        gamma = np.exp(2j * np.pi * turns * t)
        w = trapezoid(2j * np.pi * turns * gamma / gamma, t) / (2j * np.pi)
        assert abs(w - turns) < 1e-9
        windings.append(turns)

    # 4.5: ML bound on the upper half-circle
    t_half = np.linspace(0, np.pi, 20001)
    ml_rows = {}
    for radius in [2, 5, 10, 100]:
        z = radius * np.exp(1j * t_half)
        size = abs(trapezoid(1j * z / (z**2 + 1), t_half))
        bound = np.pi * radius / (radius**2 - 1)
        ml_rows[radius] = (size, bound)
    assert [f"{ml_rows[r][0]:.6f}" for r in (2, 5, 10, 100)] == ["0.927295", "0.394791", "0.199337", "0.019999"]
    assert [f"{ml_rows[r][1]:.6f}" for r in (2, 5, 10, 100)] == ["2.094395", "0.654498", "0.317333", "0.031419"]

    # 5.1: a quarter circle in 8 steps
    angles = np.linspace(0, np.pi / 2, 9)
    points = np.exp(1j * angles)
    steps = np.diff(points)
    products = (1 / points[1:]) * steps
    running8 = np.concatenate([[0], np.cumsum(products)])
    assert f"{running8[-1]:.3f}" == "0.154+1.561j"   # close to i pi/2 = 1.571i

    # 5.2: running integrals on the unit circle
    t2 = np.linspace(0, 2 * np.pi, 2001)
    z2 = np.exp(1j * t2)
    dt = t2[1] - t2[0]
    running = {}
    for label, values in [("$1/z$", 1 / z2), ("$z^2$", z2**2), ("$1/z^2$", 1 / z2**2)]:
        prod = values * 1j * z2
        running[label] = np.concatenate([[0], np.cumsum((prod[1:] + prod[:-1]) / 2 * dt)])
    assert abs(running["$1/z$"][-1] - 2j * np.pi) < 1e-9
    assert abs(running["$z^2$"][-1]) < 1e-9 and abs(running["$1/z^2$"][-1]) < 1e-9

    # 5.3: three paths from 0 to 1 + i
    s = np.linspace(0, 1, 4001)

    def concatenate(*legs):
        pts = np.concatenate([leg if k == 0 else leg[1:] for k, leg in enumerate(legs)])
        return pts, np.linspace(0, len(legs), len(pts))

    paths = {"straight": concatenate((1 + 1j) * s),
             "right then up": concatenate(s + 0j, 1 + 1j * s),
             "parabola y = x²": concatenate(s + 1j * s**2)}
    path_values = {}
    for name, (pts, par) in paths.items():
        derivative = np.gradient(pts, par)
        path_values[name] = (trapezoid(pts**2 * derivative, par), trapezoid(np.conj(pts) * derivative, par))
    for name, (v_z2, v_conj) in path_values.items():
        assert f"{v_z2:.3f}" == "-0.667+0.667j", (name, v_z2)
    assert [f"{v[1]:.3f}" for v in path_values.values()] == ["1.000+0.000j", "1.000+1.000j", "1.000+0.333j"]

    # ---------------------------------------------------------------- figure
    fig = new_figure("A contour integral adds f(z)·Δz along a curve; around a loop only 1/z survives",
                     r"$\int_\gamma f(z)\,dz = \int_a^b f(\gamma(t))\,\gamma'(t)\,dt$.     "
                     r"$\oint_{|z-a|=r}(z-a)^n\,dz = 2\pi i$ for $n=-1$, and $0$ for every other integer $n$.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    circle = np.exp(1j * np.linspace(0, 2 * np.pi, 300))
    ax.plot(circle.real, circle.imag, color=GRID, linewidth=1.5)
    for start, step in zip(points[:-1], steps):
        arrow(ax, start, start + step, INK, width=1.6, head=11)
    ax.scatter(points.real, points.imag, color=BLUE, zorder=5, s=22, label=r"$z_k = e^{i k\pi/16}$")
    ax.text(0.92, 0.72, r"$\Delta z_k$", fontsize=13, color=INK)
    argand(ax, (-0.35, 1.35), (-0.35, 1.35), r"8 steps $\Delta z_k$ along a quarter circle")
    ax.legend(loc="lower left", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    for a, b in zip(running8[:-1], running8[1:]):
        arrow(ax, a, b, ORANGE, width=1.8, head=11)
    ax.scatter([0], [0], color=ORANGE, s=22, zorder=5)
    ax.scatter([0], [np.pi / 2], marker="x", color=INK, s=70, zorder=6, linewidths=2)
    ax.text(0.3, running8[-1].imag, f"8-step sum = {running8[-1]:.3f}\n"
            r"exact $i\pi/2$ = " + f"{np.pi / 2:.3f}i", va="center", fontsize=10.5, color=INK2)
    ax.text(0.3, 0.45, r"each $f(z_k)\Delta z_k \approx (\pi/16)\, i$" + "\npoints straight up", fontsize=10.5,
            color=ORANGE)
    argand(ax, (-0.4, 1.6), (-0.2, 1.8), r"Products $\frac{1}{z_k}\,\Delta z_k$, head to tail")

    panel(fig.add_subplot(gs[0, 2]), "The one integral that survives", [
        r"On $z = a + re^{it}$:  $(z-a)^n\,dz = i r^{n+1} e^{i(n+1)t}\,dt$.",
        r"A full period of $e^{i(n+1)t}$ integrates to 0",
        r"unless $n = -1$, where the integrand is $i$.",
        "",
        "Notebook 4.2, n = −3 … 3 on three circles:",
        "`n = -1     ->  6.2831853072j  (every circle)`",
        "`n != -1    ->  0j`",
        "",
        r"Winding: $\frac{1}{2\pi i}\oint dz/z$ = 1, −1, 2, 3",
        "for 1 turn, clockwise, 2 and 3 turns.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[1, 0])
    for (label, path), color in zip(running.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(path.real, path.imag, color=color, linewidth=2.2, label=f"f = {label}")
        ax.scatter(path[-1].real, path[-1].imag, color=color, s=40, zorder=5)
    ax.scatter(0, 2 * np.pi, marker="x", color=INK, s=80, zorder=6, linewidths=2, label=r"$2\pi i$")
    argand(ax, (-3.5, 3.5), (-1, 7), r"Running integral $\int_0^s f(e^{it})\,ie^{it}\,dt$")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(0.35, 3.0, "1/z climbs to 2πi\nand never returns", fontsize=9.5, color=BLUE)

    ax = fig.add_subplot(gs[1, 1])
    labels = {"straight": "straight", "right then up": "right, then up", "parabola y = x²": "parabola y = x²"}
    conj_text = {"straight": "1", "right then up": "1 + i", "parabola y = x²": "1 + i/3"}
    for (name, (pts, _)), color in zip(paths.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(pts.real, pts.imag, color=color, linewidth=2.4,
                label=rf"{labels[name]}:  $\int\bar z\,dz$ = {conj_text[name]}")
    ax.scatter([0, 1], [0, 1], color=INK, zorder=5, s=25)
    ax.text(1.04, 1.04, "1 + i", fontsize=10)
    ax.text(-0.15, 1.58, r"$\int z^2 dz$ = $-\frac{2}{3}+\frac{2}{3}i$ on every path ($F = z^3/3$)",
            fontsize=10, color=INK, va="center")
    argand(ax, (-0.2, 1.6), (-0.2, 1.7), r"Same endpoints: $z^2$ agrees, $\bar z$ does not")
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.92))

    ax = fig.add_subplot(gs[1, 2])
    radii = np.logspace(0.2, 3, 40)
    actual = []
    for radius in radii:
        z = radius * np.exp(1j * t_half)
        actual.append(abs(trapezoid(1j * z / (z**2 + 1), t_half)))
    ax.loglog(radii, np.pi * radii / (radii**2 - 1), color=ORANGE, linewidth=2, label=r"ML bound $\pi R/(R^2-1)$")
    ax.loglog(radii, actual, "o", markersize=4, color=BLUE, label=r"$|\int_{C_R} dz/(z^2+1)|$")
    ax.text(0.03, 0.04, "\n".join(f"R = {r:3d}:  {size:.6f} ≤ {bound:.6f}" for r, (size, bound) in ml_rows.items()),
            transform=ax.transAxes, fontsize=9.5, family="monospace", color=INK2)
    ax.set_xlabel("radius R of the upper half-circle")
    ax.set_ylabel("size")
    ax.grid(True, which="both")
    ax.set_title("ML-inequality: the arc integral → 0")
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, name="01_contour_integrals.png")


# ================================================================ notebook 02
def sheet_02_cauchys_theorem():
    # 4.1: entire functions give 0 on every curve, 1/z gives 2 pi i
    def make_curves(n_points=8001):
        t = np.linspace(0, 2 * np.pi, n_points)
        circle = np.exp(1j * t)
        blob = (1 + 0.3 * np.cos(3 * t) + 0.2 * np.sin(5 * t)) * np.exp(1j * t)
        side = np.linspace(0, 1, n_points // 4)
        corners = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j, 1 + 1j]
        square = np.concatenate([a + (b - a) * side[:-1] for a, b in zip(corners[:-1], corners[1:])]
                                + [np.array([corners[0]])])
        return {"circle": circle, "square": square, "blob": blob}

    curves = make_curves()
    row_1z = [closed_integral(lambda z: 1 / z, pts) for pts in curves.values()]
    assert [f"{v.real:+.6f}{v.imag:+.6f}j" for v in row_1z] == ["-0.000000+6.283185j", "-0.000000+6.283185j",
                                                                "-0.000000+6.283187j"]
    for f in [lambda z: np.exp(z) * np.cos(z), lambda z: z**5 - 3 * z, lambda z: 1 / (z - 2), lambda z: 1 / z**2]:
        for pts in curves.values():
            assert abs(closed_integral(f, pts)) < 2e-6

    # 4.2: conj(z) measures area
    def shoelace_area(pts):
        x, y = pts.real, pts.imag
        return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    t = np.linspace(0, 2 * np.pi, 8001)
    shapes = {"unit circle": np.exp(1j * t), "circle r = 2": 3 + 1j + 2 * np.exp(1j * t),
              "ellipse 3 × 1": 3 * np.cos(t) + 1j * np.sin(t),
              "blob": (1 + 0.3 * np.cos(3 * t) + 0.2 * np.sin(5 * t)) * np.exp(1j * t)}
    conj_values = {name: closed_integral(np.conj, pts) for name, pts in shapes.items()}
    areas = {name: shoelace_area(pts) for name, pts in shapes.items()}
    assert [f"{v.imag:.6f}" for v in conj_values.values()] == ["6.283185", "25.132739", "18.849554", "6.691590"]
    assert all(f"{2 * areas[n]:.6f}" == f"{conj_values[n].imag:.6f}" for n in shapes)

    # 4.3: Green's integrands
    x, y = sp.symbols("x y", real=True)
    zs = x + sp.I * y
    for expression, expected in [(zs**3, (0, 0)), (sp.exp(zs), (0, 0)), (sp.conjugate(zs), (0, 2))]:
        u = sp.re(sp.expand_complex(expression))
        v = sp.im(sp.expand_complex(expression))
        got = (sp.simplify(-sp.diff(v, x) - sp.diff(u, y)), sp.simplify(sp.diff(u, x) - sp.diff(v, y)))
        assert got == expected

    # 4.4: Log and the jump across the cut
    ccw = closed_integral(lambda z: 1 / z, np.exp(1j * np.linspace(0, np.pi / 2, 4001)))
    cw = closed_integral(lambda z: 1 / z, np.exp(1j * np.linspace(0, -3 * np.pi / 2, 4001)))
    assert abs(ccw.imag - 1.5707962864221408) < 1e-12 and abs(cw.imag + 4.71238789032035) < 1e-12

    # 5.1: contours that give the same answer
    t = np.linspace(0, 2 * np.pi, 8001)
    contours = {"circle": 0.6 * np.exp(1j * t),
                "blob": (1.4 + 0.4 * np.cos(3 * t) + 0.25 * np.sin(5 * t)) * np.exp(1j * t),
                "shifted ellipse": 0.5 + 2.2 * np.cos(t) + 1j * 1.2 * np.sin(t),
                "away from 0": 2 + 1.2j + 0.6 * np.exp(1j * t)}
    windings = {name: closed_integral(lambda z: 1 / z, pts) / (2j * np.pi) for name, pts in contours.items()}
    assert [round(w.real, 3) + 0.0 for w in windings.values()] == [1.0, 1.0, 1.0, 0.0]

    # 5.2: deforming a square into a circle
    tt = np.linspace(0, 2 * np.pi, 8001)
    circle = 0.5 * np.exp(1j * tt)
    direction = np.exp(1j * tt)
    square = direction / np.maximum(np.abs(direction.real), np.abs(direction.imag))
    blend = np.linspace(0, 1, 120)
    functions = {"1/z": lambda z: 1 / z, r"$\bar z$": np.conj, "1/(z − 0.8)": lambda z: 1 / (z - 0.8)}
    results = {name: [] for name in functions}
    for s in blend:
        curve = (1 - s) * square + s * circle
        for name, f in functions.items():
            results[name].append(closed_integral(f, curve).imag)
    results = {k: np.array(v) for k, v in results.items()}
    assert np.allclose(results["1/z"], 2 * np.pi, atol=1e-5)
    assert abs(results[r"$\bar z$"][0] - 8) < 1e-5 and abs(results[r"$\bar z$"][-1] - 2 * np.pi * 0.25) < 1e-5
    jump = blend[np.argmax(results["1/(z − 0.8)"] < np.pi)]
    assert 0.39 < jump < 0.41

    # 5.3: keyhole
    gap, outer_radius, inner_radius = 0.08, 2.0, 0.5
    outer_angle, inner_angle = np.arcsin(gap / outer_radius), np.arcsin(gap / inner_radius)
    outer_arc = outer_radius * np.exp(1j * np.linspace(outer_angle, 2 * np.pi - outer_angle, 4000))
    lower_cut = np.linspace(outer_arc[-1], inner_radius * np.exp(-1j * inner_angle), 400)
    inner_arc = inner_radius * np.exp(1j * np.linspace(2 * np.pi - inner_angle, inner_angle, 4000))
    upper_cut = np.linspace(inner_arc[-1], outer_arc[0], 400)
    keyhole = np.concatenate([outer_arc, lower_cut[1:], inner_arc[1:], upper_cut[1:]])
    f_key = lambda z: np.exp(z) / z
    key_value = closed_integral(f_key, keyhole)
    outer_value = closed_integral(f_key, outer_radius * np.exp(1j * np.linspace(0, 2 * np.pi, 8001)))
    inner_value = closed_integral(f_key, inner_radius * np.exp(1j * np.linspace(0, 2 * np.pi, 8001)))
    assert abs(key_value - (-2.816511845082914e-06j)) < 1e-12
    assert abs(outer_value - 6.283184661215507j) < 1e-12 and abs(inner_value - 6.283184661215508j) < 1e-12

    # 5.4: Log z as the integral of 1/w along segments from 1
    n_grid, n_steps = 200, 400
    axis = np.linspace(-2, 2, n_grid)
    plane = axis[None, :] + 1j * axis[:, None]
    s = np.linspace(0, 1, n_steps + 1)[:, None, None]
    log_by_integration = np.empty_like(plane)
    for rows in np.array_split(np.arange(n_grid), 8):
        block = plane[rows]
        integrand = (block - 1)[None, :, :] / (1 + s * (block[None, :, :] - 1))
        log_by_integration[rows] = np.sum((integrand[1:] + integrand[:-1]) / 2, axis=0) / n_steps
    log_error = np.abs(log_by_integration - np.log(plane))
    # "around 1e-6", largest just beside the cut (the segment to -2 + 0.01i grazes 0)
    assert np.median(log_error) < 3e-6 and np.percentile(log_error, 99) < 3e-5
    assert abs(log_error.max() - 0.368) < 1e-3

    # ---------------------------------------------------------------- figure
    fig = new_figure("Holomorphic means closed integrals vanish: contours can deform around the holes",
                     r"$\oint_\gamma f\,dz = \iint_\Omega (-v_x - u_y)\,dA + i\iint_\Omega (u_x - v_y)\,dA = 0$"
                     r"  by Cauchy–Riemann.   Not holomorphic: $\oint \bar z\,dz = 2i\cdot\mathrm{Area}$.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    for (name, pts), color in zip(contours.items(), [BLUE, ORANGE, AQUA, INK2]):
        style = "--" if name == "away from 0" else "-"
        ax.plot(pts.real, pts.imag, style, color=color, linewidth=2,
                label=f"{name}: {round(windings[name].real, 3) + 0.0:.0f} × 2πi")
        direction_arrow(ax, pts, len(pts) // 8, color, step=60)
    ax.scatter(0, 0, marker="x", color=ORANGE, s=90, linewidths=2.5, zorder=6)
    argand(ax, (-2.5, 3.2), (-2.4, 3.3), r"$\oint dz/z$ only asks: does it wind around 0?")
    ax.legend(loc="upper left", fontsize=8.8, ncol=2, columnspacing=0.8, handlelength=1.4)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(keyhole.real, keyhole.imag, color=BLUE, linewidth=2)
    for piece, index, step in [(outer_arc, 1000, 30), (inner_arc, 1000, 60), (lower_cut, 200, 20),
                               (upper_cut, 200, 20)]:
        direction_arrow(ax, piece, index, BLUE, step=step)
    ax.scatter(0, 0, marker="x", color=ORANGE, s=90, linewidths=2.5, zorder=6)
    ax.text(-2.3, 2.05, rf"keyhole: $\oint e^z/z\,dz$ = {key_value.imag:.1e}i ≈ 0", fontsize=9.5, color=INK)
    ax.text(-2.3, -2.3, f"outer circle {outer_value.imag:.6f}i\ninner circle  {inner_value.imag:.6f}i",
            fontsize=9.5, color=INK2)
    argand(ax, (-2.4, 2.4), (-2.45, 2.4), "Keyhole: the cuts cancel, outer = inner")

    panel(fig.add_subplot(gs[0, 2]), "Green meets Cauchy–Riemann", [
        r"$f\,dz = (u+iv)(dx+i\,dy)$; Green turns each part",
        "into an area integral (notebook 4.3, sympy):",
        "`z^3, exp(z):  -v_x - u_y = 0,  u_x - v_y = 0`",
        "`conj(z):      -v_x - u_y = 0,  u_x - v_y = 2`",
        "",
        r"So $\oint \bar z\,dz = 2i\cdot$Area. Integral vs shoelace:",
        "`unit circle    6.283185 i = 2i·π`",
        "`ellipse 3 x 1  18.849554 i = 2i·3π`",
        "`blob            6.691590 i = 2i·3.345795`",
        "Holes matter: 1/z on ℂ∖{0} gives 2πi, not 0.",
    ], spacing=0.081, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    shades = [BLUE, "#3987e5", "#86b6ef", AQUA, "#1c5cab"]
    for s_value, color in zip([0, 0.25, 0.5, 0.75, 1], [BLUE, BLUE_LIGHT, AQUA, AQUA_LIGHT, INK2]):
        curve = (1 - s_value) * square + s_value * circle
        ax.plot(curve.real, curve.imag, color=color, linewidth=1.8, label=f"s = {s_value}")
    ax.scatter([0, 0.8], [0, 0], marker="x", color=ORANGE, s=80, linewidths=2.5, zorder=6)
    ax.text(0.82, 0.08, "0.8", color=ORANGE, fontsize=10)
    argand(ax, (-1.25, 1.25), (-1.25, 1.25), r"$\gamma_s = (1-s)\cdot$square $+\ s\cdot$circle")
    ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(1.0, 0.5))

    ax = fig.add_subplot(gs[1, 1])
    for (name, values), color, width in zip(results.items(), [BLUE_LIGHT, AQUA, ORANGE], [6, 2.4, 2.2]):
        ax.plot(blend, values, color=color, linewidth=width, label=f"f = {name}")
    ax.axhline(2 * np.pi, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(0.99, 2 * np.pi + 0.25, "2π", ha="right", color=INK2, fontsize=10)
    ax.axvline(0.4, color=ORANGE, linestyle="--", linewidth=1)
    ax.text(0.43, 0.35, "curve sweeps across 0.8:\njump to 0", color=ORANGE, fontsize=9.5, va="bottom")
    ax.set_xlabel("deformation parameter s")
    ax.set_ylabel(r"Im $\oint_{\gamma_s} f\,dz$")
    ax.set_ylim(-0.5, 8.8)
    ax.grid(True)
    ax.set_title("Deforming: 1/z fixed, z̄ follows the area")
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    image = ax.imshow(log_by_integration.imag, extent=[-2, 2, -2, 2], origin="lower", cmap=DIVERGING,
                      vmin=-np.pi, vmax=np.pi)
    bar = colorbar(fig, image, ax, "")
    bar.set_ticks([-np.pi, 0, np.pi], labels=["−π", "0", "π"])
    ax.plot([-2, 0], [0, 0], color=INK, linewidth=2.5)
    ax.scatter(1, 0, color=SURFACE, edgecolor=INK, zorder=3, s=40)
    ax.text(-1.95, 0.12, "cut: Arg jumps by 2π", fontsize=9.5, color=INK)
    ax.text(1.08, 0.12, "start 1", fontsize=9.5, color=INK)
    image_axes(ax, r"$\int_1^z dw/w$ along segments = Log z")
    save(fig, name="02_cauchys_theorem.png")
    return log_error


# ================================================================ notebook 03
def sheet_03_cauchys_integral_formula():
    def cauchy_value(f, a, radius=1, n_points=2001):
        t = np.linspace(0, 2 * np.pi, n_points)
        z = radius * np.exp(1j * t)
        return trapezoid(f(z) / (z - a) * 1j * radius * np.exp(1j * t), t) / (2j * np.pi)

    assert f"{cauchy_value(np.exp, 0.3):.10f}" == "1.3498588076+0.0000000000j"
    assert f"{cauchy_value(np.sin, -0.5 + 0.4j):.10f}" == "-0.5182937041+0.3604690784j"
    assert abs(cauchy_value(np.exp, 1.7)) < 1e-10

    def circle_integral(g, center, radius, n_points=4001):
        t = np.linspace(0, 2 * np.pi, n_points)
        z = center + radius * np.exp(1j * t)
        return trapezoid(g(z) * 1j * radius * np.exp(1j * t), t)

    worked = {
        "Ex 1": (circle_integral(lambda z: np.exp(z) / (z - 1), 0, 2), 2j * np.pi * np.e),
        "Ex 2": (circle_integral(lambda z: np.cos(z) / z**3, 0, 2), -1j * np.pi),
        "Ex 3": (circle_integral(lambda z: 1 / (z**2 + 1), 1j, 1), np.pi),
        "Ex 4": (circle_integral(lambda z: np.sin(z) / z**4, 0, 1), -1j * np.pi / 3),
    }
    for name, (value, expected) in worked.items():
        assert abs(value - expected) < 1e-12, name

    # 4.3: derivatives as integrals, f = e^z cos z at a = 0.2 + 0.1i
    def cauchy_derivative(f, a, order, radius=3.0, n_points=256):
        t = np.linspace(0, 2 * np.pi, n_points + 1)
        z = a + radius * np.exp(1j * t)
        return math.factorial(order) * trapezoid(f(z) / (z - a) ** (order + 1) * 1j * radius * np.exp(1j * t),
                                                 t) / (2j * np.pi)

    zs = sp.symbols("z")
    expression = sp.exp(zs) * sp.cos(zs)
    a = 0.2 + 0.1j
    orders = [0, 1, 2, 5, 10, 15]
    printed = {0: "1.19946262e+00+9.59196607e-02j", 5: "-3.87512777e+00+1.90931420e-01j",
               10: "-7.38178168e+00-4.59688050e+00j", 15: "1.83058342e+02+3.06652386e+01j"}
    rel_errors, exact_values = [], []
    for order in orders:
        exact = complex(sp.diff(expression, zs, order).subs(zs, sp.Float(a.real) + sp.I * sp.Float(a.imag)).evalf())
        numeric = cauchy_derivative(lambda w: np.exp(w) * np.cos(w), a, order)
        if order in printed:
            assert f"{numeric:.8e}" == printed[order], order
        rel_errors.append(abs(numeric - exact) / abs(exact))
        exact_values.append(exact)
    assert max(rel_errors) < 1e-11

    # 4.4
    assert f"{abs(cmath.sin(10j)):.4e}" == "1.1013e+04"

    # 5.1: rebuild e^z sin 2z inside the disc from 64 boundary samples
    n_samples = 64
    t = 2 * np.pi * np.arange(n_samples) / n_samples
    boundary = np.exp(1j * t)
    f = lambda z: np.exp(z) * np.sin(2 * z)
    boundary_values = f(boundary)
    axis = np.linspace(-1, 1, 301)
    plane = axis[None, :] + 1j * axis[:, None]
    inside = np.abs(plane) < 0.995
    reconstructed = np.full(plane.shape, np.nan, dtype=complex)
    pts = plane[inside]
    kernel = boundary[None, :] / (boundary[None, :] - pts[:, None])
    reconstructed[inside] = (kernel * boundary_values[None, :]).mean(axis=1)
    error = np.abs(reconstructed - f(plane))
    err_08 = np.nanmax(np.where(np.abs(plane) < 0.8, error, np.nan))
    err_all = np.nanmax(error)
    assert abs(err_08 - 1.8357027058505854e-06) < 1e-15 and abs(err_all - 11.49524466924423) < 1e-9

    # 5.2: mean value of z^2 + z + 1 on |z - 0.5i| = 1
    center = 0.5j
    g = lambda z: z**2 + z + 1
    tm = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = center + np.exp(1j * tm)
    image_curve = g(circle)
    average = image_curve.mean()
    assert abs(average - g(center)) < 1e-12 and abs(g(center) - (0.75 + 0.5j)) < 1e-12

    # 5.3: maximum modulus of z^2 + z + 1/2
    axis_m = np.linspace(-1, 1, 401)
    plane_m = axis_m[None, :] + 1j * axis_m[:, None]
    modulus = np.where(np.abs(plane_m) <= 1, np.abs(plane_m**2 + plane_m + 0.5), np.nan)
    location = plane_m.flat[np.nanargmax(modulus)]
    assert abs(location - 1) < 1e-12 and abs(np.nanmax(modulus) - 2.5) < 1e-12
    zeros = np.roots([1, 1, 0.5])

    # ---------------------------------------------------------------- figure
    fig = new_figure_tall("The boundary determines the inside: Cauchy's integral formula",
                     r"$f(a) = \dfrac{1}{2\pi i}\oint_\gamma \dfrac{f(z)}{z-a}\,dz$,    "
                     r"$f^{(n)}(a) = \dfrac{n!}{2\pi i}\oint_\gamma \dfrac{f(z)}{(z-a)^{n+1}}\,dz$.    "
                     "On a circle: f(centre) = average of f on the circle.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    true_max = np.abs(f(pts)).max()
    image = ax.imshow(np.abs(reconstructed), extent=[-1, 1, -1, 1], origin="lower", cmap=SEQ, vmin=0, vmax=true_max)
    colorbar(fig, image, ax, "|f| reconstructed")
    ax.scatter(boundary.real, boundary.imag, s=12, color=ORANGE, label="64 boundary samples", zorder=3)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    image_axes(ax, r"$|e^z\sin 2z|$ rebuilt from the circle")
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none")

    ax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(np.log10(error + 1e-17), extent=[-1, 1, -1, 1], origin="lower", cmap=SEQ, vmin=-12, vmax=1)
    colorbar(fig, image, ax, r"$\log_{10}$ error")
    ax.plot(0.8 * np.cos(tm), 0.8 * np.sin(tm), color=ORANGE, linewidth=1.4, linestyle="--")
    ax.text(0, -0.15, f"|a| < 0.8:\nmax error {err_08:.1e}", ha="center", fontsize=10, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85))
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    image_axes(ax, "Error: tiny inside, large only at the rim")

    ax = fig.add_subplot(gs[0, 2])
    closed = np.append(image_curve, image_curve[0])
    ax.plot(closed.real, closed.imag, color=BLUE, linewidth=2, label=r"$f$(circle $|z-0.5i|=1$)")
    ax.scatter(average.real, average.imag, s=140, facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=5,
               label=f"average over the circle = {average.real:.3f}{average.imag:+.3f}i")
    ax.scatter(0.75, 0.5, marker="x", color=INK, s=60, zorder=6, label="f(0.5i) = 0.750+0.500i")
    argand(ax, (-2.0, 3.6), (-2.9, 3.3), r"Mean value: $f(z)=z^2+z+1$ in the $w$-plane")
    ax.legend(loc="lower center", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1)

    ax = fig.add_subplot(gs[1, 0])
    pos = np.arange(len(orders))
    ax.bar(pos, rel_errors, width=0.6, color=BLUE, zorder=2)
    ax.set_yscale("log")
    for p, e in zip(pos, rel_errors):
        ax.text(p, e * 1.6, f"{e:.0e}", ha="center", fontsize=9.5)
    ax.set_xticks(pos, [str(o) for o in orders])
    ax.set_ylim(1e-17, 1e-10)
    ax.set_xlabel("derivative order n")
    ax.set_ylabel("relative error vs sympy")
    ax.grid(axis="y", zorder=0)
    ax.set_title(r"$f^{(n)}(a)$ as an integral, 256 points, $r = 3$")
    ax.text(0.02, 0.95, r"$f = e^z\cos z$, $a = 0.2+0.1i$;" + f"\nn = 15: {exact_values[-1].real:.2f}"
            f"{exact_values[-1].imag:+.2f}i", transform=ax.transAxes, va="top", fontsize=9.5, color=INK2)

    ax = fig.add_subplot(gs[1, 1])
    image = ax.contourf(axis_m, axis_m, modulus, levels=30, cmap=SEQ)
    ax.contour(axis_m, axis_m, modulus, levels=15, colors="white", linewidths=0.4, alpha=0.7)
    colorbar(fig, image, ax, "|f(z)|")
    ax.plot(np.cos(tm), np.sin(tm), color=INK, linewidth=1)
    ax.scatter(location.real, location.imag, marker="*", s=300, color=ORANGE, edgecolor=INK, zorder=5,
               label=f"max |f| = {np.nanmax(modulus):.1f} at z = 1 (boundary)")
    ax.scatter(zeros.real, zeros.imag, marker="x", color=ORANGE, s=70, linewidths=2.5, zorder=5,
               label=r"zeros $(-1\pm i)/2$")
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-1.05, 1.05)
    image_axes(ax, r"Max modulus: $|z^2+z+\frac{1}{2}|$ has no inner peak")
    ax.legend(loc="lower left", fontsize=8.8, frameon=True, facecolor=SURFACE, edgecolor="none")

    panel(fig.add_subplot(gs[1, 2]), "Integrals become one-liners", [
        r"$\oint_{|z|=2}\frac{e^z}{z-1}dz = 2\pi e\, i$" + f"  ({worked['Ex 1'][0].imag:.2f}i)",
        r"$\oint_{|z|=2}\frac{\cos z}{z^3}dz = \frac{2\pi i}{2!}\cos''(0) = -\pi i$",
        r"$\oint_{|z-i|=1}\frac{dz}{z^2+1} = 2\pi i\cdot\frac{1}{2i} = \pi$",
        "",
        r"Cauchy estimate $|f^{(n)}(a)| \leq n!M/r^n$ gives",
        "Liouville: bounded entire ⇒ constant. So sin",
        "is unbounded on ℂ (4.4):",
        "`|sin(10i)| = sinh 10 = 1.1013e+04`",
    ], spacing=0.085, fontsize=12)
    save(fig, name="03_cauchys_integral_formula.png")


# ================================================================ notebook 04
def sheet_04_power_series():
    # 4.1: ratio and root estimates
    a_n = lambda n: n / 3.0**n
    assert f"{abs(a_n(160) / a_n(161)):.4f}" == "2.9814" and f"{abs(a_n(160)) ** (-1 / 160):.4f}" == "2.9063"

    # 4.3: Taylor coefficients of 1/(1+z^2) at c = 1 by contour integral
    f = lambda z: 1 / (1 + z**2)
    numeric = np.round(taylor_coefficients(f, 1, 1.0, 8), 14) + 0.0
    n8 = np.arange(8)
    exact8 = (1 / (-1j - 1) ** (n8 + 1) - 1 / (1j - 1) ** (n8 + 1)) / 2j
    assert np.allclose(numeric, [0.5, -0.5, 0.25, 0, -0.125, 0.125, -0.0625, 0], atol=1e-12)
    assert np.allclose(exact8, numeric, atol=1e-12)

    # 4.4: reaching x = 1.5
    x = 1.5
    target = 1 / (1 + x**2)
    n = np.arange(60)
    c0 = np.where(n % 2 == 0, (-1.0) ** (n // 2), 0.0)
    c1 = ((1 / (-1j - 1) ** (n + 1) - 1 / (1j - 1) ** (n + 1)) / 2j).real
    sums = {terms: (np.sum(c0[:terms] * x ** n[:terms]), np.sum(c1[:terms] * (x - 1) ** n[:terms]))
            for terms in [5, 10, 20, 40, 60]}
    assert f"{sums[20][0]:.4f}" == "-1022.8482" and f"{sums[10][1]:.10f}" == "0.3076782227"
    assert f"{sums[60][0]:.4f}" == "-11313374989.5179" and f"{sums[60][1]:.10f}" == "0.3076923077"
    terms_all = np.arange(1, 61)
    err0 = np.array([abs(np.sum(c0[:k] * x ** n[:k]) - target) for k in terms_all])
    err1 = np.array([abs(np.sum(c1[:k] * (x - 1) ** n[:k]) - target) for k in terms_all])

    # 5.2: error of S_40 in the plane
    axis = np.linspace(-1.6, 1.6, 400)
    plane = axis[None, :] + 1j * axis[:, None]
    partial40 = sum((-1) ** k * plane ** (2 * k) for k in range(40))
    log_err = np.clip(np.log10(np.abs(partial40 - 1 / (1 + plane**2)) + 1e-16), -16, 4)

    # 5.4: root test from contour-integral coefficients
    root_series = {
        r"$1/(1+z^2)$, R = 1": (lambda z: 1 / (1 + z**2), 0.95, 1.0, BLUE),
        r"$\tan z$, R = $\pi/2$": (np.tan, 1.5, np.pi / 2, ORANGE),
        r"$1/((z-2)(z+3))$, R = 2": (lambda z: 1 / ((z - 2) * (z + 3)), 1.9, 2.0, AQUA),
    }
    nn = np.arange(1, 100)
    root_points = {}
    for label, (g, sample_radius, radius, color) in root_series.items():
        coefficients = taylor_coefficients(g, 0, sample_radius, 100, n_points=1024)[1:]
        keep = np.abs(coefficients) * sample_radius**nn > 1e-10
        estimate = np.abs(coefficients[keep]) ** (-1 / nn[keep])
        root_points[label] = (nn[keep], estimate)
        assert abs(estimate[-1] - radius) / radius < 0.06, label

    # Example 3 radii
    assert abs(min(abs(complex(p) - 1) for p in sp.solve(1 + sp.symbols("z") ** 2)) - np.sqrt(2)) < 1e-12

    # ---------------------------------------------------------------- figure
    fig = new_figure_tall("A Taylor series converges on the largest disc that avoids every singularity",
                          r"$f(z) = \sum a_n (z-c)^n$,  $a_n = \dfrac{1}{2\pi i}\oint \dfrac{f(w)}{(w-c)^{n+1}}dw$,  "
                          r"$\dfrac{1}{R} = \limsup |a_n|^{1/n}$.   $\dfrac{1}{1+x^2}$ stops at $|x| = 1$ "
                          r"because of the poles $\pm i$.")
    gs = grid(fig, top=0.84)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1.5, 1.5, 600)
    ax.plot(xs, 1 / (1 + xs**2), color=INK, linewidth=3, label=r"$1/(1+x^2)$")
    for terms, color in zip([2, 5, 10, 20], [BLUE_LIGHT, "#86b6ef", BLUE, ORANGE]):
        ax.plot(xs, sum((-1) ** k * xs ** (2 * k) for k in range(terms)), color=color, linewidth=1.6,
                label=f"{terms} terms")
    ax.axvline(-1, color=MUTED, linestyle=":")
    ax.axvline(1, color=MUTED, linestyle=":")
    ax.set_ylim(-1, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_title(r"Real line: $1 - x^2 + x^4 - \cdots$ fails at $|x|=1$")
    ax.legend(loc="lower center", fontsize=8.8, ncol=3, columnspacing=0.8)

    ax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(log_err, extent=[-1.6, 1.6, -1.6, 1.6], origin="lower", cmap=SEQ, vmin=-16, vmax=4)
    colorbar(fig, image, ax, r"$\log_{10}|S_{40}(z) - f(z)|$")
    tc = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tc), np.sin(tc), color=ORANGE, linewidth=1.6, label="|z| = 1")
    ax.plot([-1, 1], [0, 0], color=AQUA, linewidth=3.5, label="real interval (−1, 1)")
    ax.scatter([0, 0], [1, -1], marker="x", color=ORANGE, s=110, linewidths=3, zorder=5, label=r"poles $\pm i$")
    image_axes(ax, "The disc of convergence touches ±i")
    ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor="none")

    ax = fig.add_subplot(gs[0, 2])
    for center, color in zip([0, 1, 2.2, 3.8], [BLUE, ORANGE, AQUA, INK2]):
        radius = np.hypot(center, 1)
        ax.plot(center + radius * np.cos(tc), radius * np.sin(tc), color=color, linewidth=1.8,
                label=f"c = {center}: R = {radius:.3f}")
        ax.fill(center + radius * np.cos(tc), radius * np.sin(tc), color=color, alpha=0.07)
        ax.scatter(center, 0, color=color, zorder=4, s=20)
        ax.plot([center, 0], [0, 1], color=color, linestyle="--", linewidth=1)
    ax.scatter([0, 0], [1, -1], marker="x", color=ORANGE, s=110, linewidths=3, zorder=5)
    argand(ax, (-2, 8), (-4.5, 6.6), r"Discs along the axis: $R = \sqrt{c^2+1}$")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, columnspacing=0.8)

    ax = fig.add_subplot(gs[1, 0])
    for (label, (g, _, radius, color)) in root_series.items():
        nk, est = root_points[label]
        ax.plot(nk, est, "o", markersize=3.5, color=color, label=label)
        ax.axhline(radius, color=color, linestyle="--", linewidth=1)
    ax.set_ylim(0.5, 3)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|a_n|^{-1/n}$")
    ax.grid(True)
    ax.set_title(r"Root test: $|a_n|^{-1/n} \to$ distance to the pole")
    ax.legend(loc="upper right", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(terms_all, err0, color=ORANGE, linewidth=2, label="series at 0 (R = 1): diverges")
    ax.semilogy(terms_all, np.where(err1 > 0, err1, np.nan), color=BLUE, linewidth=2,
                label=r"series at 1 (R = $\sqrt{2}$): converges")
    ax.set_xlabel("number of terms")
    ax.set_ylabel(r"$|S_N(1.5) - 1/3.25|$")
    ax.set_ylim(1e-17, 1e12)
    ax.grid(True, which="major")
    ax.set_title("Analytic continuation: reaching x = 1.5")
    ax.text(31, 1e-13, f"60 terms at 1:\n{sums[60][1]:.10f}\n= 1/3.25", fontsize=9.5, color=BLUE)
    ax.legend(loc="upper left", fontsize=9)

    panel(fig.add_subplot(gs[1, 2]), "Radius without coefficients", [
        "`f              centre  nearest sing.  R`",
        "`1/(1+z^2)      0       ±i             1`",
        "`1/(1+z^2)      1       ±i             √2`",
        "`tan z          0       ±π/2           π/2`",
        "`1/(e^z-1)-1/z  0       ±2πi           2π`",
        "`Log z          1       0              1`",
        "",
        "Contour coefficients of 1/(1+z²) at c = 1:",
        "`0.5 -0.5 0.25 0 -0.125 0.125 -0.0625 0`",
        r"$|a_n| \leq 2^{-(n+1)/2}$, so $R = \sqrt{2}$.",
    ], spacing=0.082, fontsize=12)
    save(fig, name="04_power_series_and_radius_of_convergence.png")


# ================================================================ notebook 05
def sheet_05_laurent_series():
    # 4.2: Laurent coefficients of 1/(z(z-1)) on two circles
    f = lambda z: 1 / (z * (z - 1))
    inner = laurent_coefficients(f, 0, 0.5, range(-4, 3))
    outer = laurent_coefficients(f, 0, 2.0, range(-4, 3))
    assert [round(v.real, 3) + 0.0 for v in inner.values()] == [0, 0, 0, -1, -1, -1, -1]
    assert [round(v.real, 3) + 0.0 for v in outer.values()] == [1, 1, 1, 0, 0, 0, 0]
    ess = laurent_coefficients(lambda z: np.exp(1 / z), 0, 1.0, range(-6, 2))
    assert [f"{round(v.real, 5) + 0.0:.5f}" for v in ess.values()] == [
        "0.00139", "0.00833", "0.04167", "0.16667", "0.50000", "1.00000", "1.00000", "0.00000"]

    # 4.3: approaching 0 along four directions
    assert f"{abs(np.exp(1 / 0.05)):.4e}" == "4.8517e+08" and f"{abs(np.exp(1 / -0.05)):.4e}" == "2.0612e-09"
    assert f"{abs(0.05**3 * np.exp(0.05j) / (0.05j) ** 3):.4e}" == "1.0000e+00"

    # 4.4: Picard, e^{1/z} = 1 + i
    zk = {k: 1 / (cmath.log(1 + 1j) + 2j * cmath.pi * k) for k in [0, 1, 10, 100, 1000]}
    assert f"{abs(zk[1000]):.2e}" == "1.59e-04" and f"{zk[100]:.3e}" == "8.757e-07-1.590e-03j"
    for z_k in zk.values():
        assert abs(cmath.exp(1 / z_k) - (1 + 1j)) < 1e-9

    # 5.1: the two annuli
    inner_series = lambda z, terms: -sum(z ** (n - 1) for n in range(terms))
    outer_series = lambda z, terms: sum(z ** (-n) for n in range(2, terms + 2))
    axis = np.linspace(-2.5, 2.5, 500)
    plane = axis[None, :] + 1j * axis[:, None]
    modulus = np.abs(plane)
    error = np.where(modulus < 1, np.abs(inner_series(plane, 30) - f(plane)),
                     np.abs(outer_series(plane, 30) - f(plane)))

    # 5.2 / 5.3: domain colouring on a 600 x 600 grid
    axis_d = np.linspace(-1.0, 1.0, 600)
    plane_d = axis_d[None, :] + 1j * axis_d[:, None]
    solutions = np.array([1 / (cmath.log(1 + 1j) + 2j * cmath.pi * k) for k in range(-2000, 2001)])
    window = 0.5
    axis_w = np.linspace(-window, window, 600)
    plane_w = axis_w[None, :] + 1j * axis_w[:, None]
    visible = solutions[(np.abs(solutions.real) < window) & (np.abs(solutions.imag) < window)]

    # ---------------------------------------------------------------- figure
    fig = new_figure_tall("Laurent series add negative powers: the principal part names the singularity",
                          r"$f(z) = \sum_{-\infty}^{\infty} a_n (z-c)^n$ on an annulus,  "
                          r"$a_n = \dfrac{1}{2\pi i}\oint \dfrac{f(w)}{(w-c)^{n+1}}dw$.   "
                          "Principal part: none → removable, finite → pole, infinite → essential.")
    gs = grid(fig, top=0.84, wspace=0.36)

    panels = [
        (r"Removable: $\frac{\sin z}{z} = 1 - \frac{z^2}{6} + \cdots$", np.log(np.sin(plane_d) / plane_d), 1.0),
        (r"Pole of order 3: $\frac{1}{z^3}$, hues cycle 3 times", -3 * np.log(plane_d), 1.0),
        (r"Essential: $e^{1/z} = \sum \frac{1}{n!\,z^n}$", 1 / plane_w, window),
    ]
    for column, (title, log_values, half) in enumerate(panels):
        ax = fig.add_subplot(gs[0, column])
        ax.imshow(colour_from_log(log_values), extent=[-half, half, -half, half], origin="lower")
        if column == 2:
            ax.scatter(visible.real, visible.imag, s=11, color="white", edgecolor=INK, linewidth=0.5, zorder=3,
                       label=f"{len(visible)} solutions of " + r"$e^{1/z} = 1+i$")
            ax.legend(loc="lower left", fontsize=8.8, frameon=True, facecolor=SURFACE, edgecolor="none")
        image_axes(ax, title)

    ax = fig.add_subplot(gs[1, 0])
    image = ax.imshow(np.log10(error + 1e-16), extent=[-2.5, 2.5, -2.5, 2.5], origin="lower", cmap=SEQ,
                      vmin=-16, vmax=0)
    colorbar(fig, image, ax, r"$\log_{10}$ error, 30 terms")
    tc = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tc), np.sin(tc), color=ORANGE, linewidth=1.5)
    ax.scatter([0, 1], [0, 0], marker="x", color=ORANGE, s=90, linewidths=2.5, zorder=3)
    ax.text(-0.62, 0.32, r"$-\frac{1}{z} - 1 - z - \cdots$", color=INK, fontsize=11)
    ax.text(-2.35, 1.85, r"$\frac{1}{z^2} + \frac{1}{z^3} + \cdots$", color=INK, fontsize=11)
    image_axes(ax, r"$1/(z(z-1))$: one function, two annuli")

    ax = fig.add_subplot(gs[1, 1])
    r = np.logspace(-2, 0, 300)
    for (label, direction), color in zip({"positive real axis": 1, "negative real axis": -1,
                                         "imaginary axis": 1j}.items(), [ORANGE, BLUE, AQUA]):
        z = r * direction
        ax.plot(r, (1 / z).real / np.log(10), color=color, linewidth=2.2, label=rf"$e^{{1/z}}$ along {label}")
    ax.plot(r, np.log10(np.abs(1 / r**3)), color=INK2, linestyle="--", linewidth=1.5,
            label=r"pole $1/z^3$: same on every ray")
    ax.set_xscale("log")
    ax.set_xlabel("distance |z| from 0")
    ax.set_ylabel(r"$\log_{10}|f(z)|$")
    ax.set_ylim(-48, 48)
    ax.grid(True)
    ax.set_title("No limit at an essential singularity")
    ax.legend(loc="lower right", fontsize=8.6)

    panel(fig.add_subplot(gs[1, 2]), "Coefficients are contour integrals", [
        r"$\frac{1}{z(z-1)}$, $a_{-4} \ldots a_2$ on two circles (4.2):",
        "`|z| = 0.5:   0  0  0 -1 -1 -1 -1`",
        "`|z| = 2  :   1  1  1  0  0  0  0`",
        r"$e^{1/z}$ on $|z|=1$: $a_{-n} = 1/n!$",
        "`1.00000 0.50000 0.16667 0.04167 ...`",
        "",
        r"At $|z| = 0.05$, $|e^{1/z}|$ along $+1$, $-1$, $\pm i$:",
        "`4.8517e+08   2.0612e-09   1.0000`",
        r"Picard: $z_k = 1/(\mathrm{Log}(1+i) + 2\pi i k) \to 0$,",
        "`k = 1000:  |z_k| = 1.59e-04`",
    ], spacing=0.081, fontsize=12)
    save(fig, name="05_laurent_series_and_singularities.png")


# ================================================================ notebook 06
def sheet_06_residue_theorem():
    zs = sp.symbols("z")
    assert sp.residue(zs / (zs**2 + 1), zs, sp.I) == sp.Rational(1, 2)
    assert sp.residue(sp.exp(zs) / (zs - 1) ** 2, zs, 1) == sp.E
    assert sp.simplify(sp.residue(1 / (zs**2 + 1) ** 2, zs, sp.I) + sp.I / 4) == 0

    # 4.2: numerical contour integrals vs 2 pi i sum Res
    examples = [
        ("z/(z²+1), |z|=2", lambda z: z / (z**2 + 1), 0, 2, 2j * np.pi),
        ("eᶻ/(z−1)², |z|=2", lambda z: np.exp(z) / (z - 1) ** 2, 0, 2, 2j * np.pi * np.e),
        ("z²/(z−1)³, |z|=3", lambda z: z**2 / (z - 1) ** 3, 0, 3, 2j * np.pi),
        ("e^(1/z), |z|=1", lambda z: np.exp(1 / z), 0, 1, 2j * np.pi),
        ("tan z, |z|=2", np.tan, 0, 2, -4j * np.pi),
        ("1/(z⁴+1), |z|=2", lambda z: 1 / (z**4 + 1), 0, 2, 0),
        ("(1−cos z)/z⁴, |z|=1", lambda z: (1 - np.cos(z)) / z**4, 0, 1, 0),
        ("1/(z(z−1)(z−3)), |z−1|=½", lambda z: 1 / (z * (z - 1) * (z - 3)), 1, 0.5, -1j * np.pi),
        ("1/(z(z−1)(z−3)), |z−1|=3/2", lambda z: 1 / (z * (z - 1) * (z - 3)), 1, 1.5, -1j * np.pi / 3),
    ]
    printed = ["+6.2831853072", "+17.0794684453", "+6.2831853072", "+6.2831853072", "-12.5663706144",
               "+0.0000000000", "+0.0000000000", "-3.1415926536", "-1.0471975512"]
    numeric_values = []
    for (name, f, center, radius, predicted), text in zip(examples, printed):
        value = periodic_circle_integral(f, center, radius)
        value = complex(round(value.real, 10) + 0.0, round(value.imag, 10) + 0.0)
        assert f"{value.imag:+.10f}" == text and f"{value.real:+.10f}" == "+0.0000000000", name
        assert abs(value - predicted) < 1e-9
        numeric_values.append(value)

    # 4.3 / 5.1: a rational function with hand-picked poles
    numerator = np.poly1d([1, 0, 2])
    denominator = np.poly1d(np.poly([1, -1, 2j, 0.5 - 1.5j, -2 + 0.5j]))
    poles = denominator.roots
    residues = numerator(poles) / denominator.deriv()(poles)
    center, radius = 0.3 + 0.2j, 1.6
    inside = np.abs(poles - center) < radius
    prediction = 2j * np.pi * residues[inside].sum()
    value = periodic_circle_integral(lambda z: numerator(z) / denominator(z), center, radius)
    assert [f"{p:+.3f}: {r:+.4f}" for p, r in zip(poles, residues)] == [
        "-0.000+2.000j: -0.0320-0.0320j", "-2.000+0.500j: +0.2120+0.0715j", "+0.500-1.500j: -0.0394-0.0027j",
        "+1.000-0.000j: +0.1395+0.0032j", "-1.000-0.000j: -0.2800-0.0400j"]
    assert abs(prediction - (0.2309495139936279 - 0.8830422593874016j)) < 1e-13 and abs(value - prediction) < 1e-12
    assert abs(residues.sum()) < 1e-13

    # 4.4: limaçon
    tl = np.linspace(0, 2 * np.pi, 40001)
    limacon = (0.5 + np.cos(tl)) * np.exp(1j * tl)
    wind = {p: closed_integral(lambda z: 1 / (z - p), limacon) / (2j * np.pi) for p in (0.25, 1.0)}
    assert f"{wind[0.25].real:.6f}" == "2.000000" and f"{wind[1.0].real:.6f}" == "1.000000"
    lim_value = closed_integral(lambda z: 1 / ((z - 0.25) * (z - 1)), limacon)
    assert abs(lim_value - (-8.37758022554568j)) < 1e-12

    # 5.2: big curve = sum of small circles
    f52 = lambda z: np.exp(z) / (z * (z - 1) * (z + 1j))
    pole52 = np.array([0, 1, -1j])
    t = np.linspace(0, 2 * np.pi, 20001)
    big_curve = (1.9 + 0.3 * np.cos(3 * t)) * np.exp(1j * t) + 0.2
    big_value = closed_integral(f52, big_curve)
    small_values = [closed_integral(f52, p + 0.15 * np.exp(1j * t)) for p in pole52]
    assert abs(sum(small_values) - (1.3103995846902887 + 4.198765334692214j)) < 1e-12
    assert abs(big_value - (1.3103995020835013 + 4.198765352324122j)) < 1e-12

    # 5.3: tan z staircase
    radii = np.linspace(0.2, 8.5, 400)
    pole_distances = np.pi / 2 + np.pi * np.arange(3)
    radii = radii[np.min(np.abs(radii[:, None] - pole_distances[None, :]), axis=1) > 0.03]
    stairs = np.array([periodic_circle_integral(np.tan, 0, r, n_points=6000) for r in radii]) / (2j * np.pi)
    expected = -2 * np.sum(radii[:, None] > pole_distances[None, :], axis=1)
    assert np.allclose(stairs.real, expected, atol=1e-6)

    # ---------------------------------------------------------------- figure
    fig = new_figure("Only the residue survives the loop: integrals become sums",
                     r"$\oint_\gamma f\,dz = 2\pi i \sum_{a\ \mathrm{inside}} \mathrm{Res}(f, a)$,   "
                     r"$\mathrm{Res}(f, a) = a_{-1}$.   Simple pole of $g/h$: $\mathrm{Res} = g(a)/h'(a)$.")
    gs = grid(fig, top=0.84, wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    axis_x = np.linspace(-3, 3, 600)
    axis_y = np.linspace(-2.5, 3, 550)
    plane = axis_x[None, :] + 1j * axis_y[:, None]
    image = ax.imshow(np.log10(np.abs(numerator(plane) / denominator(plane))), extent=[-3, 3, -2.5, 3],
                      origin="lower", cmap=SEQ, vmin=-3, vmax=1.5)
    colorbar(fig, image, ax, r"$\log_{10}|f(z)|$")
    tc = np.linspace(0, 2 * np.pi, 300)
    ax.plot(center.real + radius * np.cos(tc), center.imag + radius * np.sin(tc), color=ORANGE, linewidth=2)
    for p, res, ins in zip(poles, residues, inside):
        color = ORANGE if ins else MUTED
        ax.scatter(p.real, p.imag, marker="x", s=90, linewidths=2.5, color=color, zorder=4)
        if abs(p - 1) < 1e-9:
            offset = (-12, -22)
        elif abs(p + 1) < 1e-9:
            offset = (-50, -22)
        else:
            offset = (6, 6)
        ax.annotate(f"{res.real:+.2f}{res.imag:+.2f}i", (p.real, p.imag), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK if ins else INK2,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.text(-2.9, -2.35, f"2πi·(sum inside) = {prediction.real:.4f}{prediction.imag:+.4f}i", fontsize=9.5,
            color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9))
    image_axes(ax, r"$(z^2+2)/q(z)$: orange poles are inside")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(big_curve.real, big_curve.imag, color=BLUE, linewidth=2,
            label=f"big curve: {big_value.real:.4f}{big_value.imag:+.4f}i")
    for p, name, v, color in zip(pole52, ["0", "1", "−i"], small_values, [ORANGE, AQUA, INK2]):
        small = p + 0.15 * np.exp(1j * t)
        ax.plot(small.real, small.imag, color=color, linewidth=2, label=f"circle at {name}: {v.real:+.4f}{v.imag:+.4f}i")
        ax.scatter(p.real, p.imag, marker="x", color=color, s=50, zorder=5)
    total = sum(small_values)
    ax.text(-2.2, -2.75, f"sum of small circles = {total.real:.4f}{total.imag:+.4f}i", fontsize=9.5, color=INK)
    argand(ax, (-2.3, 2.7), (-3.0, 4.4), r"$\oint e^z/(z(z-1)(z+i))\,dz$: big = sum of small")
    ax.legend(loc="upper left", fontsize=8.5)

    panel(fig.add_subplot(gs[0, 2]), "Computing residues", [
        r"simple pole:  $\lim_{z\to a}(z-a)f(z)$,   or  $\frac{g(a)}{h'(a)}$",
        r"order $m$:  $\frac{1}{(m-1)!}\lim \frac{d^{m-1}}{dz^{m-1}}[(z-a)^m f]$",
        "essential: read a₋₁ off the Laurent series",
        "",
        r"$\frac{z}{z^2+1}$ at $\pm i$: 1/2 each  →  2πi",
        r"$\tan z$ at $\pm\pi/2$: −1 each  →  −4πi on |z| = 2",
        r"$e^{1/z} = 1 + \frac{1}{z} + \cdots$: residue 1",
        r"$\frac{1-\cos z}{z^4} = \frac{1}{2z^2} - \frac{1}{24}+\cdots$: pole, residue 0",
    ], spacing=0.082, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(limacon.real, limacon.imag, color=BLUE, linewidth=2)
    for k in [2000, 15000, 22000]:
        direction_arrow(ax, limacon, k, BLUE, step=300)
    ax.scatter([0.25, 1.0], [0, 0], marker="x", color=ORANGE, s=80, linewidths=2.5, zorder=5)
    ax.text(-1.75, -0.55, "pole 0.25: winding 2,\nRes = −4/3", color=ORANGE, fontsize=9.5)
    ax.text(-1.75, -1.05, "pole 1: winding 1,\nRes = +4/3", color=ORANGE, fontsize=9.5)
    ax.text(0.22, 0.07, "0.25", color=ORANGE, fontsize=8.5)
    ax.text(1.03, 0.07, "1", color=ORANGE, fontsize=8.5)
    ax.text(-1.75, 1.12, rf"$2\pi i\,(2\cdot(-\frac{{4}}{{3}}) + \frac{{4}}{{3}}) = -\frac{{8}}{{3}}\pi i$"
            f"\nnumerical: {lim_value.imag:.6f}i", fontsize=9.5, color=INK)
    argand(ax, (-1.8, 1.75), (-1.3, 1.6), r"Winding numbers weight residues")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(radii, stairs.real, ".", markersize=4, color=BLUE)
    for d, label in zip(pole_distances, ["π/2", "3π/2", "5π/2"]):
        ax.axvline(d, color=ORANGE, linestyle=":", linewidth=1.2)
        ax.text(d + 0.08, 0.45, label, color=ORANGE, fontsize=9.5)
    ax.set_xlabel("radius r")
    ax.set_ylabel("(1/2πi) ∮ tan z dz on |z| = r")
    ax.set_ylim(-6.8, 1)
    ax.grid(True)
    ax.set_title("tan z: −2 each time a pair of poles enters")

    ax = fig.add_subplot(gs[1, 2])
    names = ["z/(z²+1)", "eᶻ/(z−1)²", "z²/(z−1)³", "exp(1/z)", "tan z", "1/(z⁴+1)", "(1−cos z)/z⁴",
             "1/(z(z−1)(z−3)), ½", "1/(z(z−1)(z−3)), 3/2"]
    pos = np.arange(len(examples))[::-1]
    predicted_imag = [complex(e[4]).imag for e in examples]
    ax.barh(pos, predicted_imag, height=0.6, color=BLUE_LIGHT, zorder=2, label=r"$2\pi i\sum$Res (imag part)")
    ax.scatter([v.imag for v in numeric_values], pos, color=ORANGE, s=28, zorder=3, label="contour integral")
    ax.set_yticks(pos, names, fontsize=9)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("imaginary part (real parts are 0)")
    ax.grid(axis="x", zorder=0)
    ax.set_title("Nine checks: agreement to 1e-10")
    ax.legend(loc="lower right", fontsize=8.8)
    save(fig, name="06_the_residue_theorem.png")


# ================================================================ notebook 07
def sheet_07_real_integrals():
    def by_residues(num, den):
        P, Q = np.poly1d(num), np.poly1d(den)
        upper = Q.roots[Q.roots.imag > 0]
        return 2j * np.pi * (P(upper) / Q.deriv()(upper)).sum()

    def by_quadrature(f, n_intervals=200000):
        step = np.pi / n_intervals
        u = -np.pi / 2 + (np.arange(n_intervals) + 0.5) * step
        return np.sum(f(np.tan(u)) / np.cos(u) ** 2) * step

    rational = {
        "1/(1+x²)": ([1], [1, 0, 1], np.pi, "π"),
        "1/(1+x⁴)": ([1], [1, 0, 0, 0, 1], np.pi / np.sqrt(2), "π/√2"),
        "1/(x²+2x+2)": ([1], [1, 2, 2], np.pi, "π"),
        "x²/(x⁴+5x²+4)": ([1, 0, 0], [1, 0, 5, 0, 4], np.pi / 3, "π/3"),
    }
    printed = {"1/(1+x²)": "3.1415926536", "1/(1+x⁴)": "2.2214414691", "1/(x²+2x+2)": "3.1415926536",
               "x²/(x⁴+5x²+4)": "1.0471975512"}
    for name, (num, den, exact, _) in rational.items():
        r = by_residues(num, den)
        q = by_quadrature(lambda x: np.polyval(num, x) / np.polyval(den, x))
        assert f"{r.real:.10f}" == f"{q:.10f}" == f"{exact:.10f}" == printed[name], name

    theta = 2 * np.pi * np.arange(4096) / 4096
    trig = {"1/(2+cos θ)": (1 / (2 + np.cos(theta)), 2 * np.pi / np.sqrt(3), "2π/√3"),
            "1/(5−4 sin θ)": (1 / (5 - 4 * np.sin(theta)), 2 * np.pi / 3, "2π/3"),
            "sin²θ/(5+4 cos θ)": (np.sin(theta) ** 2 / (5 + 4 * np.cos(theta)), np.pi / 4, "π/4")}
    for name, (values, exact, _) in trig.items():
        assert f"{2 * np.pi * values.mean():.10f}" == f"{exact:.10f}", name

    # 4.3 Fourier-type integrals
    xs = np.linspace(-2000.0, 2000.0, 2_000_001)
    lorentz = {}
    for k, a in [(1, 1), (2, 1), (1, 3), (5, 0.5)]:
        q = trapezoid(np.cos(k * xs) / (xs**2 + a**2), xs)
        lorentz[(k, a)] = q
    assert [f"{v:.8f}" for v in lorentz.values()] == ["1.15572781", "0.42516816", "0.05213736", "0.51575523"]

    # 5.1 semicircle for 1/(1+z^4)
    poles4 = np.exp(1j * np.pi * np.array([1, 3, 5, 7]) / 4)
    upper = poles4[poles4.imag > 0]
    res_upper = -upper / 4
    assert abs(2j * np.pi * res_upper.sum() - np.pi / np.sqrt(2)) < 1e-12

    # 5.2 arc integrals
    def arc_integral(f, radius, n_points=40001):
        t = np.linspace(0, np.pi, n_points)
        z = radius * np.exp(1j * t)
        return trapezoid(f(z) * 1j * z, t)

    radii = np.logspace(0.4, 2.5, 40)
    arc_cases = {
        r"$1/(1+z^2)$": (lambda z: 1 / (1 + z**2), BLUE),
        r"$e^{iz}/(1+z^2)$": (lambda z: np.exp(1j * z) / (1 + z**2), ORANGE),
        r"$z\,e^{iz}/(1+z^2)$": (lambda z: z * np.exp(1j * z) / (1 + z**2), AQUA),
    }
    arc_sizes = {label: np.array([abs(arc_integral(f, r)) for r in radii]) for label, (f, _) in arc_cases.items()}
    jordan = np.pi * radii / (radii**2 - 1)
    assert np.all(arc_sizes[r"$z\,e^{iz}/(1+z^2)$"] <= jordan) and arc_sizes[r"$z\,e^{iz}/(1+z^2)$"][-1] < 0.02

    # 5.3 |e^{iz}| vs |cos z| on R = 10
    t_arc = np.linspace(0, np.pi, 500)
    z_arc = 10 * np.exp(1j * t_arc)
    assert abs(np.abs(np.cos(z_arc)).max() - np.cosh(10)) / np.cosh(10) < 1e-4

    # 5.4 roots of z^2 + 4z + 1
    roots = np.roots([1, 4, 1])
    assert abs(roots.prod() - 1) < 1e-12

    # ---------------------------------------------------------------- figure
    fig = new_figure("Close the real line with a vanishing arc: real integrals become residue sums",
                     r"$\int_{-\infty}^{\infty}\frac{P}{Q}dx = 2\pi i\sum_{\mathrm{Im}\,a>0}\mathrm{Res}$,   "
                     r"$\int_0^{2\pi}R(\cos\theta,\sin\theta)\,d\theta$ with $z = e^{i\theta}$, "
                     r"$d\theta = \frac{dz}{iz}$.   Use $e^{ikz}$, never $\cos kz$.")
    gs = grid(fig, top=0.84, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    radius = 2.0
    axis_x = np.linspace(-2.6, 2.6, 520)
    axis_y = np.linspace(-1.6, 2.6, 420)
    plane = axis_x[None, :] + 1j * axis_y[:, None]
    image = ax.imshow(np.log10(np.abs(1 / (1 + plane**4))), extent=[-2.6, 2.6, -1.6, 2.6], origin="lower",
                      cmap=SEQ, vmin=-2.5, vmax=1)
    colorbar(fig, image, ax, r"$\log_{10}|f|$")
    tt = np.linspace(0, np.pi, 300)
    ax.plot([-radius, radius], [0, 0], color=BLUE, linewidth=3)
    ax.plot(radius * np.cos(tt), radius * np.sin(tt), color=ORANGE, linewidth=3)
    arrow(ax, 0, 0.7, BLUE, width=2.4)
    arrow(ax, radius * np.exp(1.4j), radius * np.exp(1.75j), ORANGE, width=2.4)
    ax.scatter(upper.real, upper.imag, marker="x", s=110, linewidths=3, color=ORANGE, zorder=4)
    lower = poles4[poles4.imag < 0]
    ax.scatter(lower.real, lower.imag, marker="x", s=110, linewidths=3, color=MUTED, zorder=4)
    for p, res in zip(upper, res_upper):
        ax.annotate(f"Res = {res.real:+.3f}{res.imag:+.3f}i", (p.real, p.imag),
                    xytext=(-70, 16) if p.real < 0 else (-40, -26), textcoords="offset points", fontsize=9,
                    color=INK, bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.text(-2.5, -1.45, r"$2\pi i\,(\mathrm{sum}) = \pi/\sqrt{2} = 2.2214$", fontsize=10, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9))
    image_axes(ax, r"$\int \frac{dx}{1+x^4}$: close with $C_R$ above")

    ax = fig.add_subplot(gs[0, 1])
    for label, (f, color) in arc_cases.items():
        ax.loglog(radii, arc_sizes[label], "o", markersize=3.5, color=color, label=label)
    ax.loglog(radii, jordan, "--", color=INK, linewidth=1.2, label=r"$\pi R/(R^2-1)$: ML, and Jordan for $ze^{iz}$")
    ax.loglog(radii, np.pi * radii**2 / (radii**2 - 1), ":", color=AQUA, linewidth=1.6,
              label=r"ML for $ze^{iz}/(1+z^2)$ → π: useless")
    ax.set_xlabel("R")
    ax.set_ylabel(r"$|\int_{C_R} f\,dz|$")
    ax.set_ylim(1e-9, 20)
    ax.grid(True, which="major")
    ax.set_title("The arc integral vanishes as R grows")
    ax.legend(loc="lower left", fontsize=8.5)

    ax = fig.add_subplot(gs[0, 2])
    tc = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tc), np.sin(tc), color=BLUE, linewidth=2, label=r"$z = e^{i\theta}$, $\theta\in[0,2\pi]$")
    arrow(ax, np.exp(0.3j), np.exp(0.55j), BLUE, width=2.2)
    for root in roots:
        ins = abs(root) < 1
        color = ORANGE if ins else MUTED
        ax.scatter(root.real, root.imag, marker="x", s=110, linewidths=3, color=color, zorder=4)
        ax.annotate(f"{root.real:.3f}" + (" inside" if ins else " outside"), (root.real, root.imag),
                    xytext=(0, 12), textcoords="offset points", ha="center", color=color, fontsize=9.5)
    ax.text(-4.2, -1.35, r"Res at $\alpha$: $\frac{2}{i(\alpha-\beta)} = \frac{1}{\sqrt{3}\,i}$  →  "
            r"$\frac{2\pi}{\sqrt{3}} = 3.6276$", fontsize=10, color=INK)
    argand(ax, (-4.6, 1.5), (-1.7, 1.6), r"$\int_0^{2\pi}\frac{d\theta}{2+\cos\theta}$ on the unit circle")
    ax.legend(loc="upper left", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(t_arc, np.abs(np.exp(1j * z_arc)), color=BLUE, linewidth=2.2, label=r"$|e^{iz}| = e^{-R\sin t}$")
    ax.semilogy(t_arc, np.abs(np.cos(z_arc)), color=ORANGE, linewidth=2.2, label=r"$|\cos z|$, up to $\cosh 10$")
    ax.axhline(1, color=MUTED, linestyle=":")
    ax.text(np.pi / 2, np.cosh(10) * 2.2, f"{np.cosh(10):.3g}", ha="center", fontsize=9.5, color=ORANGE)
    ax.set_xlabel("t (angle along the arc)")
    ax.set_ylabel("modulus (log scale)")
    ax.set_ylim(1e-5, 1e9)
    ax.grid(True, which="major")
    ax.set_title(r"On $z = 10e^{it}$: why $e^{iz}$ and not $\cos z$")
    ax.legend(loc="upper right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    ks = np.linspace(0, 6, 200)
    for a, color in [(0.5, AQUA), (1, BLUE), (3, ORANGE)]:
        ax.plot(ks, np.pi / a * np.exp(-ks * a), color=color, linewidth=2, label=f"a = {a}")
    for (k, a), q in lorentz.items():
        color = {0.5: AQUA, 1: BLUE, 3: ORANGE}[a]
        ax.scatter(k, q, s=55, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=4)
        ax.annotate(f"{q:.4f}", (k, q), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("frequency k")
    ax.set_ylabel(r"$\int \frac{\cos kx}{x^2+a^2}dx$")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 3.4)
    ax.grid(True)
    ax.set_title(r"Lorentzian: residue $\frac{\pi}{a}e^{-ka}$, dots = quadrature")
    ax.legend(loc="upper right", fontsize=9.5)

    panel(fig.add_subplot(gs[1, 2]), "Residues vs numerical quadrature", [
        "`integral           residues     quadrature`",
        "`1/(1+x^4)          2.2214414691 2.2214414691`",
        "`x^2/(x^4+5x^2+4)   1.0471975512 1.0471975512`",
        "`1/(2+cos t)        3.6275987285 3.6275987285`",
        "`1/(5-4 sin t)      2.0943951024 2.0943951024`",
        "`sin^2/(5+4 cos t)  0.7853981634 0.7853981634`",
        "`cos x/(1+x^2)      1.15572735   1.15572781`",
        "",
        r"$\frac{\cos x}{1+x^2}$: only the pole $i$ counts, $e^{i\cdot i} = e^{-1}$,",
        r"so the integral is $2\pi i\cdot\frac{e^{-1}}{2i} = \pi/e$.",
    ], spacing=0.08, fontsize=11.5)
    save(fig, name="07_real_integrals_via_residues.png")


if __name__ == "__main__":
    sheet_01_contour_integrals()
    log_error = sheet_02_cauchys_theorem()
    print(f"   notebook 02 5.4: median |F - Log| = {np.median(log_error):.1e}, max = {log_error.max():.1e}")
    sheet_03_cauchys_integral_formula()
    sheet_04_power_series()
    sheet_05_laurent_series()
    sheet_06_residue_theorem()
    sheet_07_real_integrals()
