#!/usr/bin/env python3
"""Concept sheets for module 07 (Applications): phasors, the damped oscillator, Fourier series and the
DFT, the z-transform, Mandelbrot and Julia sets, and qubits.

One page per notebook that summarizes its central idea. Every number is computed here from the same
circuits, equations, signals, seeds and grids the notebook uses, and is checked against the values
the notebook prints with asserts. Images are written to modules/07_applications/img/.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_module_07_concept_images.py
"""
import cmath
import math
from functools import partial

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, GRID, SEQ,
                       new_figure, grid, panel, trapezoid)
from viz_style import save as _save

LESSON = REPO / "modules" / "07_applications"
save = partial(_save, folder=LESSON / "img")

PURPLE = "#8a5cc6"
ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"


# ================================================================ helpers
def arrow(ax, start, end, color, width=2.2, zorder=4, label=None, style="-|>", linestyle="-", alpha=1.0):
    start, end = complex(start), complex(end)
    ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag), zorder=zorder,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=width, mutation_scale=15,
                                shrinkA=0, shrinkB=0, linestyle=linestyle, alpha=alpha))
    if label is not None:
        ax.plot([], [], color=color, linewidth=width, linestyle=linestyle, label=label)


def argand(ax, xlim, ylim, title, xlabel="Re", ylabel="Im", grid_on=True):
    """Complex-plane axes: equal aspect, both axes through the origin, labelled Re / Im."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=1)
    if grid_on:
        ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def unit_circle(ax, radius=1.0, center=0j, color=MUTED, linestyle="--", linewidth=1.0, zorder=1, label=None):
    t = np.linspace(0, 2 * np.pi, 400)
    w = center + radius * np.exp(1j * t)
    ax.plot(w.real, w.imag, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, label=label)


def legend(ax, **kw):
    params = dict(fontsize=9, framealpha=0.92, frameon=True, edgecolor="none", facecolor=SURFACE)
    params.update(kw)
    ax.legend(**params)


def ramp(k, count, lo=0.35, hi=0.95):
    return SEQ(lo + (hi - lo) * k / max(count - 1, 1))


def rk4_path(rhs, state0, times):
    states = np.zeros((len(times), len(state0)))
    states[0] = state0
    for k in range(len(times) - 1):
        t, y, dt = times[k], states[k], times[k + 1] - times[k]
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = rhs(t + dt, y + dt * k3)
        states[k + 1] = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


# ================================================================ notebook 01
def sheet_01_phasors_and_ac_circuits():
    # 4.1: Examples 1-3
    V1 = cmath.rect(10, math.pi / 6)
    assert f"{V1.real:+.4f} {V1.imag:+.4f}" == "+8.6603 +5.0000"
    assert f"{cmath.phase(3 - 4j):+.4f}" == "-0.9273"
    Z3 = 3 + 1j * 100 * 0.04
    I3 = 10 / Z3
    assert abs(Z3 - (3 + 4j)) < 1e-12 and abs(I3 - (1.2 - 1.6j)) < 1e-12
    lag_deg, lag_ms = math.degrees(cmath.phase(Z3)), 1000 * cmath.phase(Z3) / 100
    assert f"{lag_deg:.2f} {lag_ms:.2f}" == "53.13 9.27"

    # 4.2: Example 4, series RLC below resonance
    R, L, C, omega, V = 10.0, 0.1, 1e-4, 200.0, 10.0
    Z_L, Z_C = 1j * omega * L, 1 / (1j * omega * C)
    Z = R + Z_L + Z_C
    I = V / Z
    V_R, V_L, V_C = R * I, Z_L * I, Z_C * I
    assert abs(Z - (10 - 30j)) < 1e-12 and round(abs(Z), 4) == 31.6228
    assert abs(I - (0.1 + 0.3j)) < 1e-12 and round(abs(I), 4) == 0.3162
    assert round(math.degrees(cmath.phase(I)), 2) == 71.57
    assert np.allclose([V_R, V_L, V_C], [1 + 3j, -6 + 2j, 15 - 5j]) and abs(V_R + V_L + V_C - V) < 1e-12
    omega_0 = 1 / math.sqrt(L * C)
    Q = omega_0 * L / R
    assert f"{omega_0:.2f} {Q:.3f} {R / L:.1f}" == "316.23 3.162 100.0"
    assert abs(1 / (1 / 10 + 1 / (-10j)) - (5 - 5j)) < 1e-12          # Example 5

    # 4.4: RK4 of the circuit ODE from rest against the phasor prediction
    V_m = 10.0
    dt, t_end = 1e-4, 0.4
    times = np.linspace(0.0, int(round(t_end / dt)) * dt, int(round(t_end / dt)) + 1)
    states = rk4_path(lambda t, s: np.array([s[1], (V_m * np.cos(omega * t) - R * s[1] - s[0] / C) / L]),
                      np.array([0.0, 0.0]), times)
    phasor_current = np.real(I * np.exp(1j * omega * times))
    late_err = np.max(np.abs(states[:, 1] - phasor_current)[times > 0.25])
    early_err = np.max(np.abs(states[:, 1] - phasor_current)[times < 0.05])
    assert f"{late_err:.2e}" == "1.34e-06" and f"{early_err:.2e}" == "3.86e-01"

    # Exercise 8: average power
    power = 0.5 * (V * np.conj(I)).real
    assert abs(power - 0.5) < 1e-12 and abs(0.5 * R * abs(I) ** 2 - 0.5) < 1e-12
    # half-power frequencies (4.3)
    omega_1 = (-R + math.sqrt(R ** 2 + 4 * L / C)) / (2 * L)
    omega_2 = (R + math.sqrt(R ** 2 + 4 * L / C)) / (2 * L)
    assert abs(omega_2 - omega_1 - R / L) < 1e-9 and abs(omega_1 * omega_2 - omega_0 ** 2) < 1e-6

    fig = new_figure("A sinusoid is the shadow of a rotating arrow: AC circuits become complex arithmetic",
                     r"$v(t) = V_m\cos(\omega t + \varphi) = \mathrm{Re}(Ve^{i\omega t})$,  $V = V_m e^{i\varphi}$.   "
                     r"Ohm's law $V = ZI$ with $Z_R = R$,  $Z_L = i\omega L$,  $Z_C = 1/(i\omega C)$;   "
                     r"series RLC resonates at $\omega_0 = 1/\sqrt{LC}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # rotating phasor of Example 1 and its shadow
    ax = fig.add_subplot(gs[0, 0])
    unit_circle(ax, radius=10, color=GRID, linestyle="-", linewidth=1.5)
    snaps = np.linspace(0, 2 * np.pi / 100, 6, endpoint=False)
    for k, t in enumerate(snaps):
        tip = V1 * cmath.exp(1j * 100 * t)
        color = ramp(k, len(snaps), 0.55, 1.0)
        arrow(ax, 0, tip, color, width=2.2)
        ax.plot([tip.real, tip.real], [tip.imag, 0], ":", color=color, linewidth=1.2)
        ax.plot(tip.real, 0, "o", color=color, markersize=7, zorder=5)
    ax.text(V1.real + 0.4, V1.imag + 0.6, r"$V = 10e^{i\pi/6}$ at $t = 0$", fontsize=10, color=SEQ(0.55))
    ax.text(-11.5, -11.2, "dots on the real axis = v(t) at six instants", fontsize=9.5, color=INK2)
    argand(ax, (-12, 12), (-12, 12), r"$V e^{i\omega t}$ spins; $v(t)$ is its real part")

    # Kirchhoff's voltage law as vector addition (Example 4)
    ax = fig.add_subplot(gs[0, 1])
    start = 0j
    for name, piece, color, offset in ((r"$V_R = 1 + 3i$", V_R, AQUA, (0.35, -0.2)),
                                       (r"$V_L = -6 + 2i$", V_L, ORANGE, (0.2, 0.45)),
                                       (r"$V_C = 15 - 5i$", V_C, BLUE, (0.5, 0.35))):
        arrow(ax, start, start + piece, color, width=2.6)
        middle = start + piece / 2
        ax.text(middle.real + offset[0], middle.imag + offset[1], name, color=color, fontsize=10.5,
                ha="left" if offset[0] > 0 else "right")
        start += piece
    arrow(ax, 0, V, INK, width=2.8)
    ax.text(V / 2, -0.85, "source V = 10", ha="center", fontsize=10.5, color=INK)
    arrow(ax, 0, 20 * I, MUTED, width=1.5, linestyle="--")
    ax.text((20 * I).real - 0.3, (20 * I).imag + 0.35, r"$20I$ (current direction)", fontsize=9.5, color=INK2,
            ha="center")
    argand(ax, (-7, 13), (-2, 8), r"KVL at $\omega = 200$: $V_R + V_L + V_C = V$")
    ax.set_xlabel("Re (volts)")
    ax.set_ylabel("Im (volts)")

    panel(fig.add_subplot(gs[0, 2]), "One law V = ZI for R, L and C", [
        r"$|V| = |Z|\,|I|$,   $\arg V = \arg Z + \arg I$",
        "|Z| scales the amplitude, arg Z is the lead",
        "of the voltage over the current.",
        "",
        "Series RL (Example 3): R = 3, L = 0.04, ω = 100",
        f"Z = 3 + 4i = 5 at {lag_deg:.2f}°  →  I = 10/Z = 1.2 − 1.6i",
        f"current lags by {lag_deg:.2f}° = {lag_ms:.2f} ms",
        "",
        "Series RLC (Example 4): ω = 200 rad/s",
        r"$Z = 10 + 20i - 50i = 10 - 30i$,  $I = 0.1 + 0.3i$",
        f"|I| = {abs(I):.4f} A, leading by {math.degrees(cmath.phase(I)):.2f}°",
        f"|V_C| = {abs(V_C):.1f} V exceeds the 10 V source:",
        "V_L and V_C point opposite ways and cancel.",
    ], spacing=0.073, fontsize=12)

    # resonance curves
    ax = fig.add_subplot(gs[1, 0])
    w = np.linspace(50, 1000, 2000)
    for R_k, color in ((5.0, ORANGE), (10.0, BLUE), (20.0, AQUA)):
        I_w = V / (R_k + 1j * (w * L - 1 / (w * C)))
        ax.plot(w, np.abs(I_w), color=color, linewidth=2.2, label=f"R = {R_k:g} ohms, Q = {omega_0 * L / R_k:.2f}")
    ax.axvline(omega_0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(omega_0 + 12, 1.93, r"$\omega_0 = 316.2$", fontsize=10, color=INK2)
    ax.plot([omega_1, omega_2], [V / (math.sqrt(2) * R)] * 2, color=BLUE, linewidth=4, alpha=0.35,
            solid_capstyle="butt")
    ax.annotate(f"bandwidth R/L = {R / L:.0f}", (omega_2, V / (math.sqrt(2) * R)), xytext=(430, 0.85),
                fontsize=9.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.8))
    ax.plot(omega, abs(I), "o", color=INK, markersize=6, zorder=5)
    ax.annotate("Example 4", (omega, abs(I)), xytext=(70, 0.62), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8))
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel("|I| (amperes)")
    ax.set_ylim(0, 2.15)
    ax.grid(True)
    legend(ax, loc="upper right")
    ax.set_title("Resonance: |I| peaks where Im Z = 0")

    # admittance circle
    ax = fig.add_subplot(gs[1, 1])
    w = np.geomspace(20, 5000, 2000)
    Y = 1 / (R + 1j * (w * L - 1 / (w * C)))
    ax.plot(Y.real, Y.imag, color=BLUE, linewidth=2.4, zorder=3, label=r"$Y(\omega) = 1/Z(\omega)$, R = 10")
    for w_k in (150.0, 200.0, 316.23, 500.0, 700.0):
        y_k = 1 / (R + 1j * (w_k * L - 1 / (w_k * C)))
        ax.plot(y_k.real, y_k.imag, "o", color=ORANGE if w_k == 200.0 else INK, markersize=6, zorder=5)
        out = (y_k - 0.05) / abs(y_k - 0.05)
        pos = y_k + 0.009 * out
        ax.text(pos.real, pos.imag, f"{w_k:g}", fontsize=9, ha="left" if out.real > 0.3 else
                ("right" if out.real < -0.3 else "center"), va="center", color=INK2)
    assert abs(1 / Z - (0.01 + 0.03j)) < 1e-12
    ax.annotate(r"$\omega = 200$: $Y = 0.01 + 0.03i = I/V$", (0.01, 0.03), xytext=(0.03, 0.063), fontsize=9.5,
                color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8))
    ax.plot(0.05, 0, "+", color=INK2, markersize=9)
    argand(ax, (-0.04, 0.125), (-0.07, 0.075), r"Admittance $Y = 1/Z$: a circle of diameter 1/R")
    ax.text(0.052, -0.066, "R = 10, centre 1/(2R), labels = ω (rad/s)", fontsize=9, color=INK2, ha="center")

    panel(fig.add_subplot(gs[1, 2]), "Resonance, and the ODE check", [
        f"ω₀ = 1/√(LC) = {omega_0:.2f} rad/s, where Z = R is real",
        f"Q = ω₀L/R = {Q:.3f},   bandwidth R/L = {R / L:.0f} rad/s",
        f"half-power at ω₁ = {omega_1:.1f}, ω₂ = {omega_2:.1f}",
        "Z(ω) climbs the line Re Z = R; inversion",
        "maps that line to the admittance circle.",
        "",
        "RK4 on  L i′ + R i + q/C = 10 cos 200t,  from rest:",
        f"  max |sim − Re(I e^(iωt))| = {early_err:.2e} for t < 0.05 s",
        f"  max |sim − Re(I e^(iωt))| = {late_err:.2e} for t > 0.25 s",
        "The transient dies; the phasor answer remains.",
        "",
        r"Power $\frac{1}{2}\mathrm{Re}(V\bar{I})$" + f" = {power:.1f} W = " + r"$\frac{1}{2}R|I|^2$",
    ], spacing=0.074, fontsize=12)
    save(fig, name="01_phasors_and_ac_circuits.png")


# ================================================================ notebook 02
def sheet_02_linear_odes_and_the_damped_oscillator():
    # 4.1: roots and regimes
    cases = [(2, 5), (4, 4), (5, 4), (0, 1), (-0.2, 1)]
    roots = {c: np.roots([1, c[0], c[1]]) for c in cases}
    assert np.allclose(sorted(roots[(2, 5)], key=lambda r: r.imag), [-1 - 2j, -1 + 2j])
    assert np.allclose(roots[(4, 4)], [-2, -2]) and np.allclose(sorted(roots[(5, 4)]), [-4, -1])
    assert np.allclose(sorted(np.round(roots[(-0.2, 1)], 4), key=lambda r: r.imag), [0.1 - 0.995j, 0.1 + 0.995j])

    # 4.3: one complex constant fits the initial conditions
    lam = (-2 + np.sqrt(complex(4 - 20))) / 2
    sigma, omega_d = -lam.real, lam.imag
    C = 1 - 1j * (0 + sigma * 1) / omega_d
    assert abs(lam - (-1 + 2j)) < 1e-12 and abs(C - (1 - 0.5j)) < 1e-12
    times = np.arange(0, 8 + 1e-3 / 2, 1e-3)
    numeric = rk4_path(lambda t, y: np.array([y[1], -2 * y[1] - 5 * y[0]]), np.array([1.0, 0.0]), times)[:, 0]
    fit_err = np.max(np.abs(np.real(C * np.exp(lam * times)) - numeric))
    assert fit_err < 1e-12

    # 4.4: driven oscillator, Example 5
    c1, c0, F, omega = 2.0, 5.0, 10.0, 2.0
    X = F / ((1j * omega) ** 2 + c1 * (1j * omega) + c0)
    assert f"{X.real:.4f} {X.imag:.4f} {abs(X):.4f} {np.angle(X):.4f}" == "0.5882 -2.3529 2.4254 -1.3258"
    h0, hv0 = -X.real, -(1j * omega * X).real
    C5 = h0 - 1j * (hv0 + sigma * h0) / omega_d
    t5 = np.arange(0, 10 + 1e-3 / 2, 1e-3)
    sim5 = rk4_path(lambda t, y: np.array([y[1], F * np.cos(omega * t) - c1 * y[1] - c0 * y[0]]),
                    np.array([0.0, 0.0]), t5)[:, 0]
    steady = np.real(X * np.exp(1j * omega * t5))
    transient = np.real(C5 * np.exp(lam * t5))
    drive_err = np.max(np.abs(steady + transient - sim5))
    assert drive_err < 1e-11 and f"{abs(C5) * np.exp(-10):.2e}" == "1.23e-04"
    lag_deg = -math.degrees(np.angle(X))
    assert round(lag_deg) == 76

    # Exercise 7: resonance peak for zeta = 0.2
    zeta_r = 0.2
    w = np.linspace(0.01, 2, 400001)
    gain = np.abs(1 / (1 - w ** 2 + 2j * zeta_r * w))
    assert f"{w[np.argmax(gain)]:.5f} {gain.max():.6f}" == "0.95917 2.551552"
    assert f"{np.sqrt(1 - 2 * zeta_r ** 2):.5f} {1 / (2 * zeta_r * np.sqrt(1 - zeta_r ** 2)):.6f}" == "0.95917 2.551552"

    fig = new_figure("Guess $e^{\\lambda t}$: the roots of a quadratic decide how an oscillator moves",
                     r"$\ddot x + 2\zeta\omega_0\dot x + \omega_0^2x = 0$  →  $\lambda^2 + 2\zeta\omega_0\lambda + \omega_0^2 = 0$;   "
                     r"Re $\lambda$ = decay rate, Im $\lambda$ = frequency;   $x(t) = \mathrm{Re}(Ce^{\lambda t})$;   "
                     r"driven: $X = F/p(i\omega)$")
    gs = grid(fig, top=0.84, wspace=0.3)

    zetas = [0.0, 0.15, 0.5, 1.0, 2.0]
    colors = [INK2, BLUE, AQUA, ORANGE, PURPLE]

    ax = fig.add_subplot(gs[0, 0])
    unit_circle(ax, color=GRID, linestyle="-", linewidth=1.5)
    for z_locus in np.linspace(0, 2.5, 26):
        r = np.roots([1, 2 * z_locus, 1]).astype(complex)
        ax.plot(r.real, r.imag, ".", color=BLUE_LIGHT, markersize=5, zorder=2)
    for zeta, color in zip(zetas, colors):
        r = np.roots([1, 2 * zeta, 1]).astype(complex)
        ax.plot(r.real, r.imag, "o", color=color, markersize=9, zorder=4, label=f"ζ = {zeta:g}")
    ax.annotate("double root −1\n(critical)", (-1, 0), xytext=(-2.35, 0.55), fontsize=9.5, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8))
    argand(ax, (-4.0, 1.1), (-1.35, 1.35), r"Roots for $\omega_0 = 1$ (pale: ζ every 0.1)")
    ax.set_xlabel("Re λ (decay rate)")
    ax.set_ylabel("Im λ (frequency)")
    legend(ax, loc="lower left", ncol=2, fontsize=8.5)

    ax = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 20, 2000)
    for zeta, color in zip(zetas, colors):
        if zeta < 1:
            lam_z = complex(-zeta, np.sqrt(1 - zeta ** 2))
            x = np.real((1 - 1j * zeta / lam_z.imag) * np.exp(lam_z * t))
        elif zeta == 1:
            x = (1 + t) * np.exp(-t)
        else:
            l1, l2 = np.sort(np.roots([1, 2 * zeta, 1]).real)
            A = -l2 / (l1 - l2)
            x = A * np.exp(l1 * t) + (1 - A) * np.exp(l2 * t)
        ax.plot(t, x, color=color, linewidth=2, label=f"ζ = {zeta:g}")
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_ylim(-1.1, 1.75)
    ax.grid(True)
    legend(ax, loc="upper right", ncol=3, fontsize=8.5)
    ax.set_title("Same equation, x(0) = 1, x′(0) = 0")

    panel(fig.add_subplot(gs[0, 2]), "Three regimes, one quadratic", [
        r"Underdamped  $\ddot x + 2\dot x + 5x = 0$:  $\lambda = -1 \pm 2i$",
        r"$C = 1 - i/2$,  $x = e^{-t}(\cos 2t + 0.5\sin 2t)$",
        f"(RK4 agrees to {fit_err:.1e})",
        r"Critical  $\ddot x + 4\dot x + 4x = 0$:  $\lambda = -2$ (double)",
        r"$x = (1 + 2t)e^{-2t}$",
        r"Overdamped  $\ddot x + 5\dot x + 4x = 0$:  $\lambda = -1, -4$",
        r"$x = (4/3)e^{-t} - (1/3)e^{-4t}$",
        "",
        "ζ ≤ 1: conjugate roots on the circle |λ| = ω₀;",
        "ζ > 1: they split along the negative real axis.",
        "Stable ⇔ every root in the left half-plane",
        r"($\ddot x - 0.2\dot x + x = 0$: $\lambda = 0.1 \pm 0.995i$ grows).",
    ], spacing=0.074, fontsize=12)

    # the complex solution is a shrinking spiral
    ax = fig.add_subplot(gs[1, 0])
    ts = np.linspace(0, 4, 800)
    z = C * np.exp(lam * ts)
    ax.plot(z.real, z.imag, color=BLUE, linewidth=2, zorder=3, label=r"$z(t) = Ce^{(-1+2i)t}$, $C = 1 - i/2$")
    marks = [0.0, 0.4, 0.8, 1.2]
    for k, tk in enumerate(marks):
        zk = C * np.exp(lam * tk)
        arrow(ax, 0, zk, ORANGE, width=1.8, zorder=4)
        ax.plot([zk.real, zk.real], [zk.imag, 0], ":", color=MUTED)
        ax.plot(zk.real, 0, "o", color=ORANGE, markersize=5, zorder=5)
        lab = zk * (1 + 0.16 / abs(zk))
        ax.text(lab.real, lab.imag, f"t = {tk}", fontsize=9, ha="center", va="center", color=INK2)
    ax.plot([], [], "o", color=ORANGE, markersize=5, label="shadow x(t) = Re z(t)")
    argand(ax, (-0.7, 1.3), (-0.8, 0.75), "Example 1: the solution is a shrinking spiral")
    legend(ax, loc="lower left", fontsize=8.5)

    # complex amplitude X(omega)
    ax = fig.add_subplot(gs[1, 1])
    w_full = np.linspace(0, 6, 4000)
    sides = {(0.2, 0.5): 1, (0.2, 1.0): 1, (0.2, 1.5): -1, (0.5, 0.5): -1, (0.5, 1.0): -1, (0.5, 1.5): 1}
    for zeta, color in ((0.2, BLUE), (0.5, AQUA)):
        Xw = 1 / (1 - w_full ** 2 + 2j * zeta * w_full)
        ax.plot(Xw.real, Xw.imag, color=color, linewidth=2.2, label=f"ζ = {zeta}")
        for w_k in (0.5, 1.0, 1.5):
            p = 1 / (1 - w_k ** 2 + 2j * zeta * w_k)
            side = sides[(zeta, w_k)]
            text = f"ω = {w_k}" if (zeta, w_k) != (0.2, 1.0) else "ω = ω₀: X = −2.5i, lag 90°"
            ax.plot(p.real, p.imag, "o", color=color, markersize=6, zorder=4)
            dy = -0.17 if (zeta, w_k) == (0.2, 1.0) else 0.0
            ax.text(p.real + side * 0.09, p.imag + dy, text, fontsize=8.5, color=color, va="center",
                    ha="left" if side > 0 else "right")
    ax.plot(1, 0, "o", color=INK, markersize=5)
    ax.text(1.0, 0.12, "ω = 0", fontsize=8.5, ha="center", color=INK2)
    argand(ax, (-1.6, 1.8), (-2.8, 0.45), r"Driven: $X(\omega) = 1/(1 - \omega^2 + 2i\zeta\omega)$")
    legend(ax, loc="upper left")

    panel(fig.add_subplot(gs[1, 2]), r"Example 5: $\ddot x + 2\dot x + 5x = 10\cos 2t$", [
        r"$p(2i) = (2i)^2 + 2(2i) + 5 = 1 + 4i$",
        f"X = 10/(1 + 4i) = {X.real:.4f} − {-X.imag:.4f}i",
        f"|X| = {abs(X):.4f},  arg X = {np.angle(X):.4f} rad:",
        f"the response lags the force by {lag_deg:.0f}°.",
        "",
        "From rest: x = steady Re(X e^(2it)) + transient",
        f"Re(C e^((−1+2i)t)), C = {C5.real:.4f} {'+' if C5.imag >= 0 else '−'} {abs(C5.imag):.4f}i.",
        f"RK4 agrees to {drive_err:.1e} over [0, 10];",
        f"transient left at t = 10: {abs(C5) * np.exp(-10):.2e}.",
        "Peak gain at ω_r = ω₀√(1 − 2ζ²); for ζ = 0.2:",
        f"ω_r = {np.sqrt(1 - 2 * zeta_r ** 2):.5f}, peak gain {gain.max():.6f}",
    ], spacing=0.074, fontsize=12)
    save(fig, name="02_linear_odes_and_the_damped_oscillator.png")


# ================================================================ notebook 03
def square_coefficient(n):
    return (1 - (-1) ** n) / (1j * np.pi * n) if n != 0 else 0j


def square_partial_sum(x, N):
    total = np.zeros_like(x, dtype=complex)
    for n in range(-N, N + 1):
        if n % 2 != 0:
            total += (2 / (1j * np.pi * n)) * np.exp(1j * n * x)
    return total.real


def sheet_03_fourier_series_and_the_dft():
    # 4.1: coefficients by numerical integration
    x = np.linspace(-np.pi, np.pi, 200001)
    square_numeric = {n: trapezoid(np.sign(x) * np.exp(-1j * n * x), x) / (2 * np.pi) for n in (1, 2, 3, -3)}
    saw_numeric = {n: trapezoid(x * np.exp(-1j * n * x), x) / (2 * np.pi) for n in (1, 2, 3, -3)}
    assert [f"{square_numeric[n].imag:+.5f}" for n in (1, 2, 3, -3)] == ["-0.63662", "-0.00000", "-0.21221", "+0.21221"]
    assert [f"{saw_numeric[n].imag:+.5f}" for n in (1, 2, 3, -3)] == ["-1.00000", "+0.50000", "-0.33333", "+0.33333"]

    # 4.2: DFT from the definition
    def dft(signal):
        k = np.arange(len(signal))
        return np.exp(-2j * np.pi * np.outer(k, k) / len(signal)) @ signal

    assert np.allclose(dft(np.array([1, 2, 3, 4], dtype=complex)), [10, -2 + 2j, -2, -2 - 2j])
    tone = np.cos(2 * np.pi * 3 * np.arange(16) / 16)
    bins = {n: v for n, v in enumerate(np.fft.fft(tone)) if abs(v) > 1e-9}
    assert sorted(bins) == [3, 13] and all(abs(v - 8) < 1e-9 for v in bins.values())
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(2048)
    fft_diff = np.max(np.abs(dft(signal) - np.fft.fft(signal)))
    assert fft_diff < 1e-9

    # 4.3: sampled square wave, N = 64
    N = 64
    xs = 2 * np.pi * np.arange(N) / N
    sq = np.where(xs < np.pi, 1.0, -1.0)
    sq[0] = sq[N // 2] = 0.0
    X64 = np.fft.fft(sq)
    approx = {n: X64[n % N] / N for n in (1, 3, 5, -1, 15, 25)}
    assert [f"{approx[n].imag:+.5f}" for n in (1, 3, 5, 15, 25)] == ["-0.63611", "-0.21067", "-0.12476", "-0.03448", "-0.01118"]

    # 4.4: tones in noise
    sample_rate = 1000
    t = np.arange(0, 1.0, 1 / sample_rate)
    clean = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    noisy = clean + np.random.default_rng(42).normal(0, 1.0, size=t.size)
    spectrum = np.fft.fft(noisy)
    freqs = np.fft.fftfreq(t.size, d=1 / sample_rate)
    positive = freqs >= 0
    amplitude = 2 * np.abs(spectrum[positive]) / t.size
    top = np.argsort(amplitude)[::-1][:4]
    assert [f"{freqs[positive][b]:.1f} {amplitude[b]:.3f}" for b in top] == \
        ["50.0 0.977", "120.0 0.497", "171.0 0.157", "82.0 0.154"]
    assert round(float(np.median(amplitude)), 3) == 0.052
    keep = np.abs(spectrum) > 0.3 * t.size / 2
    assert keep.sum() == 4
    denoised = np.fft.ifft(np.where(keep, spectrum, 0)).real
    rms_noisy = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    rms_denoised = float(np.sqrt(np.mean((denoised - clean) ** 2)))
    assert f"{rms_noisy:.3f} {rms_denoised:.3f}" == "0.989 0.028"

    # 4.5 aliasing, 4.6 convolution theorem
    k100 = np.arange(100)
    assert np.allclose(np.cos(2 * np.pi * 70 * k100 / 100), np.cos(2 * np.pi * 30 * k100 / 100))
    xc, hc = np.array([1.0, 2.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])
    y_conv = np.round(np.fft.ifft(np.fft.fft(xc) * np.fft.fft(hc)).real, 12)
    assert np.allclose(y_conv, [1, 3, 2, 0])
    assert np.allclose(np.fft.fft(xc) * np.fft.fft(hc), [6, -1 - 3j, 0, -1 + 3j])

    # 5.2: Gibbs
    gibbs = {n: square_partial_sum(np.linspace(0, 0.5, 20001), n).max() for n in (9, 31, 101)}
    assert [f"{gibbs[n]:.4f}" for n in (9, 31, 101)] == ["1.1823", "1.1793", "1.1790"]

    # 5.3: chain of arrows at x = 1, N = 15
    order = [0] + [s * n for n in range(1, 16, 2) for s in (1, -1)]
    terms = [square_coefficient(n) * np.exp(1j * n * 1.0) for n in order]
    S15 = sum(terms)
    assert abs(S15.imag) < 1e-12

    # 5.6: 9-point moving average
    N_ma, width = 256, 9
    kernel = np.zeros(N_ma)
    kernel[:width] = 1 / width
    H = np.fft.fft(kernel)
    f_ma = np.fft.fftfreq(N_ma)
    assert abs(H[0] - 1) < 1e-12

    fig = new_figure("A periodic signal is a sum of rotating arrows; sampled, it becomes the DFT",
                     r"$f(x) = \sum_n c_n e^{inx}$,  $c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)e^{-inx}dx$;   "
                     r"$X_n = \sum_{k} x_k e^{-2\pi i nk/N}$;   $\mathrm{DFT}(x * h)_n = X_n H_n$")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    position = 0j
    shown = [(n, tm) for n, tm in zip(order, terms) if abs(tm) > 1e-12]
    for k, (n, tm) in enumerate(shown):
        color = ramp(k, len(shown), 0.55, 1.0)
        arrow(ax, position, position + tm, color, width=1.8)
        if abs(n) == 1:
            spot = (0.12, -0.3, "left") if n == 1 else (0.66, -0.13, "right")
            ax.text(spot[0], spot[1], f"n = {n:+d}", fontsize=10, color=color, ha=spot[2], va="center")
        position += tm
    ax.plot(position.real, position.imag, "*", color=ORANGE, markersize=16, zorder=6,
            label=f"$S_{{15}}(1)$ = {S15.real:.3f} (Im = 0)")
    ax.plot(1, 0, "o", color=INK, markersize=6, zorder=6, label="f(1) = 1")
    argand(ax, (-0.1, 1.3), (-0.5, 0.35), r"Square wave at $x = 1$: arrows $c_ne^{inx}$ head to tail")
    legend(ax, loc="lower right")
    ax.text(0.02, 0.27, "conjugate pairs ±n mirror each other,\nso the endpoint is real", fontsize=9, color=INK2,
            va="top")

    ax = fig.add_subplot(gs[0, 1])
    xp = np.linspace(-np.pi, np.pi, 4001)
    ax.plot(xp, np.sign(xp), color=INK, linewidth=2.2, label="square wave sign(x)")
    for n, color in ((1, BLUE_LIGHT), (3, AQUA), (9, BLUE), (31, ORANGE)):
        ax.plot(xp, square_partial_sum(xp, n), color=color, linewidth=1.6, label=f"$S_{{{n}}}$")
    ax.axhline(1.1790, color=MUTED, linestyle="--", linewidth=1)
    ax.text(-3.05, 1.24, "Gibbs limit 1.1790", fontsize=9.5, color=INK2)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.45, 2.05)
    ax.set_xlabel("x")
    ax.grid(True)
    legend(ax, loc="upper center", ncol=5, fontsize=8.5, columnspacing=1.0, handlelength=1.5)
    ax.set_title("Partial sums: the 9% overshoot never shrinks")

    panel(fig.add_subplot(gs[0, 2]), "Coefficients are inner products", [
        r"$\langle e^{imx}, e^{inx}\rangle = 1$ if $m = n$, else $0$",
        "square wave: cₙ = (1 − (−1)ⁿ)/(iπn) = −2i/(πn), n odd",
        "sawtooth x:   cₙ = i(−1)ⁿ/n",
        f"numerically: square c₁ = {square_numeric[1].imag:.5f}i, c₃ = {square_numeric[3].imag:.5f}i",
        "jump → |cₙ| ~ 1/n,  corner (|x|) → 1/n²",
        f"max S_N: {gibbs[9]:.4f} (N = 9), {gibbs[31]:.4f} (31), {gibbs[101]:.4f} (101)",
        "",
        "4-point DFT of (1, 2, 3, 4): (10, −2+2i, −2, −2−2i)",
        "cos(2π·3k/16): X₃ = X₁₃ = 8, every other Xₙ = 0",
        f"N = 2048: direct O(N²) sum vs np.fft differ by {fft_diff:.0e}",
    ], spacing=0.082, fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    fpos, amp = freqs[positive], amplitude
    ax.vlines(fpos, 0, amp, color=BLUE, linewidth=1)
    ax.axhline(0.3, color=ORANGE, linestyle="--", linewidth=1.3, label="threshold 0.3: keep 4 bins (±50, ±120 Hz)")
    for b, dy in zip(top[:2], (0.05, 0.05)):
        ax.annotate(f"{fpos[b]:.0f} Hz: {amp[b]:.3f}", (fpos[b], amp[b]), xytext=(8, 2), textcoords="offset points",
                    fontsize=10, color=INK)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"amplitude $2|X_n|/N$")
    ax.grid(True, axis="y")
    legend(ax, loc="upper right")
    ax.set_title("Tones in noise σ = 1 (seed 42): two peaks pop out")

    ax = fig.add_subplot(gs[1, 1])
    order_f = np.argsort(np.abs(f_ma))
    sc = ax.scatter(H.real[order_f], H.imag[order_f], c=np.abs(f_ma)[order_f], cmap=SEQ, s=7, zorder=3,
                    vmin=0, vmax=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|frequency| (cycles/sample)")
    cbar.outline.set_visible(False)
    ax.plot(1, 0, "o", color=ORANGE, markersize=8, zorder=4, label=r"$H_0 = 1$: DC passes")
    h3 = H[3]
    arrow(ax, 0, h3, AQUA, width=1.8, zorder=5, label=f"$H_3$ = {h3.real:.3f} − {abs(h3.imag):.3f}i (the 3-cycle tone)")
    argand(ax, (-0.45, 1.15), (-0.95, 0.8), "9-point moving average: $H_n$ in the plane")
    legend(ax, loc="lower left", fontsize=8.5)

    panel(fig.add_subplot(gs[1, 2]), "Sampling: aliasing and convolution", [
        r"Riemann sum of $c_n$:  $c_n \approx X_n/N$  (N = 64 samples)",
        f"n = 1:   {approx[1].imag:.5f}i  (exact {square_coefficient(1).imag:.5f}i)",
        f"n = 25: {approx[25].imag:.5f}i  (exact {square_coefficient(25).imag:.5f}i):",
        r"near N/2 the aliases $c_{n \pm N}$ mix in.",
        "70 Hz sampled at 100 Hz gives the samples of 30 Hz.",
        "",
        f"Noise: RMS error {rms_noisy:.3f} → {rms_denoised:.3f}, keeping 4 bins",
        f"(median noise amplitude per bin {np.median(amplitude):.3f}).",
        "",
        "x = (1, 2, 0, 0), h = (1, 1, 0, 0):  x * h = (1, 3, 2, 0)",
        r"$X\cdot H = (6, -1-3i, 0, -1+3i)$ = DFT of (1, 3, 2, 0)",
        "Filtering multiplies each Xₙ by a complex gain Hₙ:",
        "scaled by |Hₙ|, rotated by arg Hₙ.",
    ], spacing=0.074, fontsize=12)
    save(fig, name="03_fourier_series_and_the_dft.png")


# ================================================================ notebook 04
def run_filter(b, a, x):
    y = np.zeros(len(x))
    for n in range(len(x)):
        total = sum(b[k] * x[n - k] for k in range(len(b)) if n - k >= 0)
        total -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n - k >= 0)
        y[n] = total
    return y


def frequency_response(b, a, omega):
    e = np.exp(-1j * np.asarray(omega))
    return sum(bk * e ** k for k, bk in enumerate(b)) / sum(ak * e ** k for k, ak in enumerate(a))


def notch(omega_0, r):
    b = np.array([1, -2 * np.cos(omega_0), 1])
    a = np.array([1, -2 * r * np.cos(omega_0), r ** 2])
    return b * a.sum() / b.sum(), a


def sheet_04_z_transform_poles_and_zeros():
    impulse = np.zeros(12)
    impulse[0] = 1.0
    assert np.allclose(run_filter([1], [1, -0.5], impulse), 0.5 ** np.arange(12))
    assert np.allclose(run_filter([1], [1, -2], impulse), 2.0 ** np.arange(12))
    r3, th3 = 0.9, np.pi / 4
    n12 = np.arange(12)
    assert np.allclose(run_filter([1], [1, -2 * r3 * np.cos(th3), r3 ** 2], impulse),
                       r3 ** n12 * np.sin((n12 + 1) * th3) / np.sin(th3))
    p3 = np.roots([1, -2 * 0.9 * np.cos(np.pi / 4), 0.81])
    assert np.allclose(sorted(np.round(p3, 4), key=lambda p: p.imag), [0.6364 - 0.6364j, 0.6364 + 0.6364j])
    p5 = np.roots([1, -2 * np.cos(1.0), 1])
    assert np.allclose(np.abs(p5), 1)
    g1 = np.abs(frequency_response([1], [1, -0.5], [0, np.pi]))
    assert np.allclose(np.round(g1, 5), [2.0, 0.66667])

    # 4.5: notch for 50 Hz at 1 kHz
    sample_rate, hum, r = 1000, 50, 0.95
    omega_0 = 2 * np.pi * hum / sample_rate
    b, a = notch(omega_0, r)
    assert np.allclose(np.round(b, 5), [0.97554, -1.85559, 0.97554]) and np.allclose(np.round(a, 5), [1, -1.80701, 0.9025])
    assert np.allclose(np.abs(np.roots(b)), 1) and np.allclose(np.abs(np.roots(a)), 0.95)
    gains = {f: abs(frequency_response(b, a, 2 * np.pi * f / sample_rate)) for f in (0, 3, 11, 45, 50, 55, 200)}
    assert [f"{gains[f]:.4f}" for f in (0, 3, 11, 45, 50, 55, 200)] == \
        ["1.0000", "0.9997", "0.9960", "0.5343", "0.0000", "0.5346", "1.0246"]
    t = np.arange(0, 2.0, 1 / sample_rate)
    measurement = np.sin(2 * np.pi * 3 * t) + 0.4 * np.sin(2 * np.pi * 11 * t)
    recorded = measurement + np.sin(2 * np.pi * hum * t + 0.3)
    cleaned = run_filter(b, a, recorded)
    late = t >= 1.0
    rms_before = np.sqrt(np.mean((recorded - measurement)[late] ** 2))
    rms_after = np.sqrt(np.mean((cleaned - measurement)[late] ** 2))
    assert f"{rms_before:.4f} {rms_after:.4f}" == "0.7071 0.0250"

    # Exercise 8: notch widths
    omega_grid = np.linspace(0, np.pi, 100001)
    widths = {}
    for r_k in (0.9, 0.95, 0.99):
        bk, ak = notch(np.pi / 10, r_k)
        below = omega_grid[np.abs(frequency_response(bk, ak, omega_grid)) < 1 / np.sqrt(2)]
        widths[r_k] = below.max() - below.min()
    assert [f"{widths[r_k]:.4f}" for r_k in (0.9, 0.95, 0.99)] == ["0.1778", "0.0980", "0.0200"]

    # 5.2 / 5.3: resonator with poles 0.9 e^{±i pi/4}, zeros ±1; gain at omega = 1.2
    res_poles = np.array([0.9 * np.exp(1j * np.pi / 4), 0.9 * np.exp(-1j * np.pi / 4)])
    res_zeros = np.array([1.0, -1.0])
    point = np.exp(1.2j)
    gain_12 = np.prod(np.abs(point - res_zeros)) / np.prod(np.abs(point - res_poles))
    assert abs(gain_12 - abs(np.prod(point - res_zeros) / np.prod(point - res_poles))) < 1e-12

    fig = new_figure("Poles inside the unit circle mean stability; on the circle, the gain is a ratio of distances",
                     r"$H(z) = \frac{b_0 + b_1z^{-1} + \cdots}{1 + a_1z^{-1} + \cdots}$,   $h[n] = \sum_j r_jp_j^{\,n}$,   "
                     r"stable $\Leftrightarrow$ all $|p_j| < 1$;   "
                     r"$|H(e^{i\omega})| = |b_0|\,\prod_k|e^{i\omega} - q_k|\,/\,\prod_j|e^{i\omega} - p_j|$")
    gs = grid(fig, top=0.84, wspace=0.3)

    radii = [(0.9, BLUE, "0.9: decays"), (1.0, INK2, "1: rings forever"), (1.05, ORANGE, "1.05: blows up")]
    ax = fig.add_subplot(gs[0, 0])
    unit_circle(ax, color=MUTED, linestyle="--", linewidth=1.2)
    for rad, color, label in radii:
        poles = rad * np.exp(np.array([1j, -1j]) * np.pi / 6)
        ax.plot(poles.real, poles.imag, "x", color=color, markersize=12, markeredgewidth=2.6, zorder=4,
                label=rf"poles ${'' if rad == 1 else f'{rad:g}'}e^{{\pm i\pi/6}}$ ({label.split(': ')[1]})")
    ax.plot(0, 0, "o", markerfacecolor="none", color=INK2, markersize=10, markeredgewidth=1.8, zorder=4,
            label="double zero at 0")
    argand(ax, (-1.35, 1.35), (-1.35, 1.35), "Three pole pairs at angle ±π/6")
    legend(ax, loc="lower left", fontsize=8.3)

    ax = fig.add_subplot(gs[0, 1])
    n40 = np.arange(40)
    imp40 = np.zeros(40)
    imp40[0] = 1.0
    for rad, color, label in radii:
        a_k = np.real(np.poly(rad * np.exp(np.array([1j, -1j]) * np.pi / 6)))
        h = run_filter([1.0], a_k, imp40)
        assert np.allclose(h, rad ** n40 * np.sin((n40 + 1) * np.pi / 6) / np.sin(np.pi / 6))
        ax.plot(n40, h, "o-", color=color, markersize=3.5, linewidth=1.4, label=f"r = {label}")
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xlabel("n (samples)")
    ax.set_ylabel("h[n]")
    ax.set_ylim(-14, 14)
    ax.grid(True)
    legend(ax, loc="upper left", fontsize=8.8)
    ax.set_title(r"$h[n] = r^n\sin((n+1)\pi/6)/\sin(\pi/6)$: period 12")

    panel(fig.add_subplot(gs[0, 2]), "Each pole is a geometric sequence", [
        r"Delay by one sample $\mapsto$ multiply by $z^{-1}$;",
        r"$a^n \mapsto \frac{1}{1 - az^{-1}} = \frac{z}{z - a}$  for $|z| > |a|$: a pole at $a$",
        "",
        "Ex 1  y[n] = 0.5 y[n−1] + x[n]: pole 0.5, h = 0.5ⁿ,",
        f"      gain {g1[0]:.0f} at DC and {g1[1]:.3f} at Nyquist (low-pass)",
        "Ex 2  y[n] = 2 y[n−1] + x[n]: pole 2, h = 2ⁿ, unstable",
        r"Ex 3  poles $0.9e^{\pm i\pi/4}$ = 0.6364 ± 0.6364i:",
        r"      $h[n] = 0.9^n\sin((n+1)\pi/4)/\sin(\pi/4)$",
        "Ex 4  ½(x[n] + x[n−1]): zero at −1, |H| = cos(ω/2)",
        "Ex 5  cos(n): poles e^(±i), |p| = 1 (marginal)",
        "Inside the circle: decay. On it: rings. Outside: blow-up.",
    ], spacing=0.074, fontsize=11.5)

    # log|H| over the plane and the walk on the unit circle
    ax = fig.add_subplot(gs[1, 0])
    g = np.linspace(-1.6, 1.6, 600)
    Zp = g[None, :] + 1j * g[:, None]
    Hp = np.ones_like(Zp)
    for q in res_zeros:
        Hp *= Zp - q
    for p in res_poles:
        Hp /= Zp - p
    with np.errstate(divide="ignore"):
        log_gain = np.log10(np.abs(Hp))
    image = ax.imshow(np.clip(log_gain, -1.5, 1.5), extent=[-1.6, 1.6, -1.6, 1.6], origin="lower", cmap=SEQ,
                      vmin=-1.5, vmax=1.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.text(-1.52, 1.5, r"colour: $\log_{10}|H(z)|$", fontsize=9.5, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85), zorder=8)
    cbar.outline.set_visible(False)
    unit_circle(ax, color=SURFACE, linestyle="-", linewidth=1.2, zorder=2)
    upper = np.exp(1j * np.linspace(0, np.pi, 200))
    ax.plot(upper.real, upper.imag, color=SURFACE, linewidth=3, zorder=2)
    for q in res_zeros:
        arrow(ax, q, point, AQUA, width=2, zorder=5)
    for p in res_poles:
        arrow(ax, p, point, ORANGE, width=2, zorder=5)
    ax.plot(res_poles.real, res_poles.imag, "x", color=ORANGE, markersize=11, markeredgewidth=2.5, zorder=6)
    ax.plot(res_zeros.real, res_zeros.imag, "o", markerfacecolor="none", color=AQUA, markersize=10,
            markeredgewidth=2.2, zorder=6)
    ax.plot(point.real, point.imag, "o", color=INK, markersize=6, zorder=7)
    ax.text(point.real - 0.05, point.imag + 0.12, r"$e^{1.2i}$", fontsize=11, ha="center", color=INK)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(r"Zeros $\pm 1$, poles $0.9e^{\pm i\pi/4}$:  $|H(e^{1.2i})|$" + f" = {gain_12:.3f}")

    ax = fig.add_subplot(gs[1, 1])
    fr = np.linspace(0, 150, 3000)
    for r_k, color in ((0.8, BLUE_LIGHT), (0.95, BLUE), (0.99, ORANGE)):
        bk, ak = notch(omega_0, r_k)
        ax.plot(fr, np.abs(frequency_response(bk, ak, 2 * np.pi * fr / sample_rate)), color=color, linewidth=2,
                label=f"r = {r_k}")
    for f_m in (3, 11):
        ax.axvline(f_m, color=AQUA, linewidth=1, linestyle=":")
    ax.text(12.5, 0.12, "signal at\n3 and 11 Hz", fontsize=9, color=AQUA)
    ax.text(52, 0.04, "zero on the circle:\ngain 0 at 50 Hz", fontsize=9, color=INK2)
    ax.axhline(1 / np.sqrt(2), color=MUTED, linestyle="--", linewidth=0.9)
    ax.set_xlabel("frequency (Hz), sampling at 1000 Hz")
    ax.set_ylabel(r"$|H(e^{i\omega})|$")
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.3)
    ax.grid(True)
    legend(ax, loc="lower right")
    ax.set_title(r"Notch: zeros $e^{\pm i\pi/10}$, poles $re^{\pm i\pi/10}$")

    panel(fig.add_subplot(gs[1, 2]), "Removing 50 Hz mains hum", [
        "ω₀ = 2π·50/1000 = π/10 rad per sample,  r = 0.95,",
        "b scaled so that the DC gain is exactly 1:",
        "`b = [0.97554 -1.85559 0.97554]`",
        "`a = [1. -1.80701 0.9025]`",
        f"gain at 3 and 11 Hz: {gains[3]:.4f}, {gains[11]:.4f}",
        f"gain at 45, 50, 55 Hz: {gains[45]:.4f}, {gains[50]:.4f}, {gains[55]:.4f}",
        "",
        "Recurrence on 3 Hz + 11 Hz + hum, 2 s (t ≥ 1 s):",
        f"RMS deviation {rms_before:.4f} → {rms_after:.4f}",
        "",
        "Notch width (|H| < 1/√2) ≈ 2(1 − r):",
        f"r = 0.9: {widths[0.9]:.4f},  0.95: {widths[0.95]:.4f},  0.99: {widths[0.99]:.4f} rad",
    ], spacing=0.077, fontsize=12)
    save(fig, name="04_z_transform_poles_and_zeros.png")


# ================================================================ notebook 05
def escape_counts(c, z0, max_iter, radius_rule="max"):
    z = np.array(z0, dtype=complex).copy()
    c = np.broadcast_to(np.asarray(c, dtype=complex), z.shape)
    counts = np.full(z.shape, max_iter, dtype=int)
    radius = np.maximum(2.0, np.abs(c)) if radius_rule == "max" else np.full(z.shape, 2.0)
    active = np.ones(z.shape, dtype=bool)
    for n in range(max_iter):
        z[active] = z[active] ** 2 + c[active]
        escaped = active & (np.abs(z) > radius)
        counts[escaped] = n + 1
        active &= ~escaped
        if not active.any():
            break
    return counts


def orbit_of_zero(c, steps):
    z, points = 0j, [0j]
    for _ in range(steps):
        z = z * z + c
        points.append(z)
    return points


def sheet_05_mandelbrot_and_julia_sets():
    # 4.1: orbits
    assert np.allclose(orbit_of_zero(-1, 7), [0, -1, 0, -1, 0, -1, 0, -1])
    assert np.allclose(orbit_of_zero(1j, 7), [0, 1j, -1 + 1j, -1j, -1 + 1j, -1j, -1 + 1j, -1j])
    assert np.allclose(orbit_of_zero(1, 5), [0, 1, 2, 5, 26, 677])
    half = orbit_of_zero(0.5, 6)
    assert [f"{p.real:.4g}" for p in half] == ["0", "0.5", "0.75", "1.062", "1.629", "3.153", "10.44"]
    assert f"{abs(half[5]) * (abs(half[5]) - 1):.2f}" == "6.79"
    # 4.2: fixed points of c = 0.2
    fp = np.sort(np.roots([1, -1, 0.2]).real)
    assert np.allclose(np.round(fp, 4), [0.2764, 0.7236]) and np.allclose(np.round(2 * fp, 4), [0.5528, 1.4472])
    assert round(orbit_of_zero(0.2, 40)[-1].real, 6) == 0.276393

    # 4.3: vectorized escape time on the notebook's grid
    re = np.linspace(-2.2, 0.8, 600)
    im = np.linspace(-1.3, 1.3, 520)
    C = re[None, :] + 1j * im[:, None]
    counts = escape_counts(C, np.zeros_like(C), 200)
    inside = counts == 200
    area = inside.sum() * (3.0 / 599) * (2.6 / 519)
    assert f"{inside.mean():.4f} {area:.3f}" == "0.1946 1.524"

    # 4.4: cardioid and bulb test with rng(0)
    rng = np.random.default_rng(0)
    samples = rng.uniform(-2, 2, 200000) + 1j * rng.uniform(-2, 2, 200000)
    s_counts = escape_counts(samples, np.zeros_like(samples), 300, radius_rule="two")
    cardioid = np.abs(1 - np.sqrt(1 - 4 * samples)) < 1
    bulb = np.abs(samples + 1) < 0.25
    bounded = s_counts == 300
    share = ((cardioid | bulb) & bounded).sum() / bounded.sum()
    assert cardioid.sum() == 14821 and bulb.sum() == 2475
    assert (s_counts[cardioid] < 300).sum() == 0 and (s_counts[bulb] < 300).sum() == 0
    assert round(share, 3) == 0.902

    # Exercise 4: c = 0.3 + 0.6i escapes after 15 steps
    z, escaped_at = 0j, None
    for n in range(1, 1001):
        z = z * z + (0.3 + 0.6j)
        if abs(z) > 2:
            escaped_at = n
            break
    assert escaped_at == 15

    fig = new_figure("Square, add c, repeat: bounded orbits of 0 draw the Mandelbrot set",
                     r"$f_c(z) = z^2 + c$;   $c \in M \Leftrightarrow |z_n| \leq 2$ for all $n$ ($z_0 = 0$);   "
                     r"main cardioid $c = \frac{\mu}{2} - \frac{\mu^2}{4}$, $|\mu| < 1$;   period-2 disk $|c + 1| < \frac{1}{4}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    # orbits of 0
    ax = fig.add_subplot(gs[0, 0])
    unit_circle(ax, radius=2, color=MUTED, linestyle="--", linewidth=1.1, label="escape radius 2")
    cases = [(-0.3 + 0.4j, "fixed point", BLUE), (-1.0 + 0j, "2-cycle", AQUA),
             (-0.1226 + 0.7449j, "3-cycle (rabbit)", PURPLE), (0.3 + 0.6j, "escapes after 15", ORANGE)]
    for c, label, color in cases:
        z, pts = 0j, [0j]
        for _ in range(60):
            z = z * z + c
            pts.append(z)
            if abs(z) > 2.5:
                break
        pts = np.array(pts)
        ax.plot(pts.real, pts.imag, "-", color=color, linewidth=0.9, alpha=0.55, zorder=2)
        ax.plot(pts.real, pts.imag, "o", color=color, markersize=3.6, zorder=3,
                label=f"c = {c.real:g}{c.imag:+g}i: {label}")
    ax.plot(0, 0, "o", color=INK, markersize=6, zorder=4)
    argand(ax, (-2.7, 2.7), (-2.7, 2.7), "Orbits of $z_0 = 0$ for four values of c")
    legend(ax, loc="lower left", fontsize=8)

    # the Mandelbrot set
    ax = fig.add_subplot(gs[0, 1])
    img = np.ma.masked_where(inside, np.log(counts))
    cmap = SEQ.copy()
    cmap.set_bad(INK)
    ax.imshow(img, extent=[re[0], re[-1], im[0], im[-1]], origin="lower", cmap=cmap)
    tt = np.linspace(0, 2 * np.pi, 500)
    card = np.exp(1j * tt) / 2 - np.exp(2j * tt) / 4
    ax.plot(card.real, card.imag, color=ORANGE, linewidth=1.6, label="cardioid (attracting fixed point)")
    unit_circle(ax, radius=0.25, center=-1, color=AQUA, linestyle="-", linewidth=1.8,
                label="|c + 1| = 1/4 (attracting 2-cycle)")
    ax.set_xlim(re[0], re[-1])
    ax.set_ylim(im[0], im[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("Re c")
    ax.set_ylabel("Im c")
    ax.set_title(f"M: 600 × 520 grid, 200 steps, area ≈ {area:.3f}")
    legend(ax, loc="upper left", fontsize=8.3)

    panel(fig.add_subplot(gs[0, 2]), "One step of z → z² + c", [
        "squares |z|, doubles arg z, then shifts by c.",
        "c = −1:  0 → −1 → 0 → −1 …  (2-cycle, in M)",
        "c = i:   0 → i → −1+i → −i → −1+i …  (in M)",
        "c = 1:   0 → 1 → 2 → 5 → 26 → 677  (escapes)",
        f"c = 0.2: orbit → {fp[0]:.4f}, multiplier |2z*| = {2 * fp[0]:.4f} < 1",
        "",
        "Escape lemma: |z| > 2 and |z| ≥ |c| give",
        "|z² + c| ≥ |z|(|z| − 1) > |z|: the orbit runs off.",
        f"c = 0.5: z₅ = {half[5].real:.4f}, so |z₆| ≥ {abs(half[5]) * (abs(half[5]) - 1):.2f} (actual {half[6].real:.2f}).",
        "Hence c ∈ M ⇔ |zₙ| ≤ 2 for all n, and M ⊂ {|c| ≤ 2}.",
    ], spacing=0.078, fontsize=12)

    # Julia sets
    sub = gs[1, 0].subgridspec(2, 2, hspace=0.35, wspace=0.12)
    gj = np.linspace(-1.7, 1.7, 400)
    Z0 = gj[None, :] + 1j * gj[:, None]
    julia = [(-1 + 0j, "basilica"), (-0.1226 + 0.7449j, "rabbit"), (-0.5 + 0.5j, "spiral"),
             (0.3 + 0.6j, "dust (c ∉ M)")]
    for k, (c, name) in enumerate(julia):
        axj = fig.add_subplot(sub[k // 2, k % 2])
        cj = escape_counts(np.full(Z0.shape, c), Z0, 200)
        axj.imshow(np.ma.masked_where(cj == 200, np.log(cj)), extent=[-1.7, 1.7, -1.7, 1.7], origin="lower",
                   cmap=cmap)
        axj.set_aspect("equal")
        axj.set_xticks([])
        axj.set_yticks([])
        axj.set_title(f"c = {c.real:g}{c.imag:+g}i: {name}", fontsize=9.5, pad=4)
        if k // 2 == 1:
            axj.set_xlabel("Re z", fontsize=9)
        if k % 2 == 0:
            axj.set_ylabel("Im z", fontsize=9)
        for spine in axj.spines.values():
            spine.set_visible(False)
        if k == 0:
            black_share = (cj == 200).mean()
    assert black_share > 0.05

    # escape speed
    ax = fig.add_subplot(gs[1, 1])
    for c, color in ((1.0, ORANGE), (0.5, AQUA), (0.3 + 0.6j, PURPLE), (0.26, BLUE), (-0.75 + 0.05j, INK2)):
        z, moduli = 0j, []
        for _ in range(200):
            z = z * z + c
            moduli.append(abs(z))
            if abs(z) > 1e30:
                break
        ax.semilogy(range(1, len(moduli) + 1), moduli, "o-", markersize=2.5, linewidth=1.3, color=color,
                    label=f"c = {c}  ({len(moduli)} steps to 1e30)")
    ax.axhline(2, color=INK, linestyle="--", linewidth=1)
    ax.text(42, 4.5, "|z| = 2", fontsize=9.5, color=INK)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|z_n|$")
    ax.set_xlim(0, 75)
    ax.grid(True, which="major")
    legend(ax, loc="upper right", fontsize=8.3)
    ax.set_title("A long wait below 2, then doubly exponential")

    panel(fig.add_subplot(gs[1, 2]), "Cardioid, bulb, and the dichotomy", [
        r"Attracting fixed point: $|f_c'(z^*)| = |2z^*| < 1$.",
        r"With $\mu = 2z^*$:  $c = z^* - z^{*2} = \mu/2 - \mu^2/4$,",
        "cusp 1/4 (μ = 1), left end −3/4 (μ = −1).",
        r"2-cycle multiplier $4(c + 1)$: the disk $|c + 1| < 1/4$.",
        "",
        "200,000 random c in [−2, 2]² (seed 0), 300 steps:",
        f"{cardioid.sum():,} in the cardioid, {bulb.sum():,} in the disk, 0 escaped;",
        f"together {100 * share:.1f}% of the bounded samples.",
        "",
        "c ∈ M: Julia set K_c connected (basilica, rabbit);",
        "c ∉ M: Cantor dust, no interior, no area.",
    ], spacing=0.077, fontsize=12)
    save(fig, name="05_mandelbrot_and_julia_sets.png")


# ================================================================ notebook 06
def bloch_vector(state):
    alpha, beta = state / np.linalg.norm(state)
    cross = np.conj(alpha) * beta
    return np.array([2 * cross.real, 2 * cross.imag, abs(alpha) ** 2 - abs(beta) ** 2])


def probs(state, basis):
    return np.array([abs(np.vdot(e, state)) ** 2 for e in basis])


def sheet_06_qubits_amplitudes_and_phases():
    ket0, ket1 = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    s = 1 / np.sqrt(2)
    plus, minus = (ket0 + ket1) * s, (ket0 - ket1) * s
    plus_i, minus_i = (ket0 + 1j * ket1) * s, (ket0 - 1j * ket1) * s

    # 4.1
    psi = (1 + 1j) * ket0 + 2 * ket1
    psi /= np.linalg.norm(psi)
    assert np.allclose(np.round(psi, 4), [0.4082 + 0.4082j, 0.8165]) and np.allclose(probs(psi, [ket0, ket1]), [1 / 3, 2 / 3])
    assert np.allclose(probs(plus, [plus, minus]), [1, 0]) and np.allclose(probs(minus, [plus, minus]), [0, 1])
    assert np.allclose(probs(plus_i, [plus, minus]), [0.5, 0.5])

    # 4.2: global vs relative phase, rng(7)
    rng = np.random.default_rng(7)
    psi_r = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi_r /= np.linalg.norm(psi_r)
    glob = np.exp(1.234j) * psi_r
    rel = np.array([psi_r[0], np.exp(1.234j) * psi_r[1]])
    bases = {"Z": [ket0, ket1], "X": [plus, minus], "Y": [plus_i, minus_i]}
    table = {name: (probs(psi_r, b)[0], probs(glob, b)[0], probs(rel, b)[0]) for name, b in bases.items()}
    assert [f"{v:.4f}" for v in table["Z"]] == ["0.0785", "0.0785", "0.0785"]
    assert [f"{v:.4f}" for v in table["X"]] == ["0.7554", "0.7554", "0.5047"]
    assert [f"{v:.4f}" for v in table["Y"]] == ["0.5844", "0.5844", "0.7689"]

    # 4.3: Bloch angles of Example 2
    alpha, beta = psi
    theta = 2 * np.arccos(np.clip(abs(alpha), 0, 1))
    phi = (np.angle(beta) - np.angle(alpha)) % (2 * np.pi)
    assert f"{theta:.4f} {np.degrees(theta):.2f} {phi:.4f}" == "1.9106 109.47 5.4978"
    r_ex = bloch_vector(psi)
    assert np.allclose(r_ex, [2 / 3, -2 / 3, -1 / 3])

    # 4.4: gates
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1, -1]).astype(complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) * s
    S = np.diag([1, 1j])
    T = np.diag([1, np.exp(1j * np.pi / 4)])
    assert np.allclose(T @ T, S) and np.allclose(S @ S, Z) and np.allclose(H @ Z @ H, X)
    for U in (X, Z, H, S, T):
        assert np.allclose(U.conj().T @ U, np.eye(2))

    # 4.5: 1000 shots, rng(2024)
    rng = np.random.default_rng(2024)
    shot_phis = [0, np.pi / 4, np.pi / 2, 2 * np.pi / 3, np.pi, 3 * np.pi / 2]
    shot_rows = []
    for ph in shot_phis:
        final = H @ np.diag([1, np.exp(1j * ph)]) @ H @ ket0
        p0 = abs(final[0]) ** 2
        outcomes = rng.choice([0, 1], size=1000, p=[p0, 1 - p0])
        shot_rows.append((ph, final[0], p0, np.mean(outcomes == 0)))
    assert [f"{row[2]:.4f}" for row in shot_rows] == ["1.0000", "0.8536", "0.5000", "0.2500", "0.0000", "0.5000"]
    assert [f"{row[3]:.3f}" for row in shot_rows] == ["1.000", "0.849", "0.504", "0.255", "0.000", "0.511"]

    # 5.3: 17 phases, rng(5)
    rng = np.random.default_rng(5)
    phi_samples = np.linspace(0, 2 * np.pi, 17)
    observed = []
    for ph in phi_samples:
        final = H @ np.diag([1, np.exp(1j * ph)]) @ H @ ket0
        observed.append(np.mean(rng.random(1000) < abs(final[0]) ** 2))
    observed = np.array(observed)
    assert np.max(np.abs(observed - np.cos(phi_samples / 2) ** 2)) < 0.05

    fig = new_figure("A qubit is two complex amplitudes: moduli give probabilities, phases decide interference",
                     r"$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$,  $P(0) = |\alpha|^2$,  $P(1) = |\beta|^2$;   "
                     r"up to a global phase $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\varphi}\sin\frac{\theta}{2}|1\rangle$;   "
                     r"$H R(\phi) H|0\rangle$: $P(0) = \cos^2\frac{\phi}{2}$")
    gs = grid(fig, top=0.84, wspace=0.3)

    sub = gs[0, 0].subgridspec(2, 2, hspace=0.7, wspace=0.3)
    states = [(r"$|+\rangle$: relative 0", np.array([s, s])),
              (r"$|-\rangle$: relative $\pi$", np.array([s, -s])),
              (r"$|{+i}\rangle$: relative $\pi/2$", np.array([s, 1j * s])),
              (r"$e^{i\pi/3}|+\rangle$: global only", np.exp(1j * np.pi / 3) * np.array([s, s]))]
    for k, (title, (a_k, b_k)) in enumerate(states):
        axs = fig.add_subplot(sub[k // 2, k % 2])
        unit_circle(axs, color=GRID, linestyle="-", linewidth=1.2)
        arrow(axs, 0, a_k, BLUE, width=5, alpha=0.45)
        arrow(axs, 0, b_k, ORANGE, width=1.8)
        for value, color, label, side in ((a_k, BLUE, r"$\alpha$", -1), (b_k, ORANGE, r"$\beta$", 1)):
            normal = 1j * value / abs(value)
            pos = 1.28 * value + side * 0.2 * normal
            axs.text(pos.real, pos.imag, label, color=color, fontsize=11, ha="center", va="center")
        argand(axs, (-1.15, 1.15), (-1.15, 1.15), title, grid_on=False)
        axs.title.set_fontsize(10)
        axs.set_xticks([-1, 0, 1])
        axs.set_yticks([-1, 0, 1])
        axs.tick_params(labelsize=8)
        axs.xaxis.label.set_fontsize(9)
        axs.yaxis.label.set_fontsize(9)

    # Bloch sphere
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRID, linewidth=0.5)
    eq = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(eq), np.sin(eq), 0, color=MUTED, linewidth=1)
    for axis in np.eye(3):
        ax.plot(*zip(-1.15 * axis, 1.15 * axis), color=INK2, linewidth=0.6)
    landmarks = {r"$|0\rangle$": ket0, r"$|1\rangle$": ket1, r"$|+\rangle$": plus, r"$|-\rangle$": minus,
                 r"$|{+i}\rangle$": plus_i, r"$|{-i}\rangle$": minus_i}
    for label, st in landmarks.items():
        rv = bloch_vector(st)
        ax.scatter(*rv, color=BLUE, s=30, depthshade=False)
        ax.text(*(1.4 * rv + (np.array([0, 0, 0.3]) if abs(rv[1]) > 0.5 else 0)), label, fontsize=11, ha="center", va="center", color=INK)
    walk = []
    st = plus.copy()
    for _ in range(8):
        walk.append(bloch_vector(st))
        st = T @ st
    walk = np.array(walk)
    assert np.allclose(walk[2], bloch_vector(plus_i)) and np.allclose(walk[4], bloch_vector(minus))
    ax.scatter(walk[:, 0], walk[:, 1], walk[:, 2], color=AQUA, s=20, depthshade=False)
    ax.quiver(0, 0, 0, *r_ex, color=ORANGE, linewidth=2.4, arrow_length_ratio=0.12)
    ax.text(*(1.25 * r_ex), "Example 2", color=ORANGE, fontsize=10)
    ax.set_box_aspect((1, 1, 1))
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-125)
    ax.set_title("Bloch sphere: relative phase = longitude\n" + r"green: $T^k|+\rangle$, k = 0..7, 45° apart",
                 pad=0)

    panel(fig.add_subplot(gs[0, 2]), "Amplitudes, probabilities, phases", [
        "Ex 1: ((1+i)|0⟩ + 2|1⟩)/√6:  P(0) = 1/3,  P(1) = 2/3",
        r"Ex 2: drop the global phase $e^{i\pi/4}$:",
        f"θ = {theta:.4f} rad ({np.degrees(theta):.2f}°), φ = 7π/4 = {phi:.4f}",
        r"Bloch vector $(2\mathrm{Re}\,\bar\alpha\beta, 2\mathrm{Im}\,\bar\alpha\beta, |\alpha|^2 - |\beta|^2)$",
        "= (2/3, −2/3, −1/3): below the equator, at −45°",
        "",
        "Random state (seed 7), P(first outcome):",
        "`basis    psi    e^(ig)psi  rel. phase`",
        f"`  Z    {table['Z'][0]:.4f}    {table['Z'][1]:.4f}     {table['Z'][2]:.4f}`",
        f"`  X    {table['X'][0]:.4f}    {table['X'][1]:.4f}     {table['X'][2]:.4f}`",
        f"`  Y    {table['Y'][0]:.4f}    {table['Y'][1]:.4f}     {table['Y'][2]:.4f}`",
        "A global phase changes nothing; a relative one does.",
    ], spacing=0.072, fontsize=12)

    # interference arrows
    ax = fig.add_subplot(gs[1, 0])
    unit_circle(ax, radius=0.5, center=0.5, color=GRID, linestyle="-", linewidth=1.4)
    arrow(ax, 0, 0.5, INK, width=2.4, label=r"path via $|0\rangle$: $\frac{1}{2}$")
    for ph, color in ((0, BLUE), (np.pi / 2, AQUA), (2 * np.pi / 3, PURPLE), (np.pi, ORANGE)):
        second = 0.5 * np.exp(1j * ph)
        total = 0.5 + second
        arrow(ax, 0.5, total, color, width=2)
        ax.plot([0, total.real], [0, total.imag], ":", color=color, linewidth=1.4)
        ax.plot(total.real, total.imag, "o", color=color, markersize=7, zorder=5,
                label=f"φ = {ph / np.pi:.3g}π: |amp(0)| = {abs(total):.3f}, P(0) = {abs(total) ** 2:.2f}")
    argand(ax, (-0.15, 1.2), (-0.55, 1.2), r"amp(0) $= \frac{1}{2} + \frac{1}{2}e^{i\phi}$: arrows add, then square")
    legend(ax, loc="upper left", fontsize=8.3)

    ax = fig.add_subplot(gs[1, 1])
    pg = np.linspace(0, 2 * np.pi, 400)
    ax.plot(pg, np.cos(pg / 2) ** 2, color=BLUE, linewidth=2.2, label=r"$P(0) = \cos^2(\phi/2)$")
    ax.plot(phi_samples, observed, "o", color=ORANGE, markersize=6, label="1000 simulated shots (seed 5)")
    ax.axhline(0.5, color=MUTED, linestyle="--", label="adding probabilities: 1/2")
    ax.set_xticks(np.arange(0, 2.01 * np.pi, np.pi / 2), ["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel("phase φ of R(φ)")
    ax.set_ylabel("probability of outcome 0")
    ax.set_ylim(-0.05, 1.35)
    ax.grid(True)
    legend(ax, loc="upper center", ncol=2, fontsize=8.5)
    ax.set_title("A phase you cannot see controls the outcome")

    panel(fig.add_subplot(gs[1, 2]), "Gates rotate; phases interfere", [
        "Unitary 2×2 matrices: X, Y, Z half-turns,",
        r"$R(\phi) = \mathrm{diag}(1, e^{i\phi})$ turns by $\phi$ about z:",
        r"$T^2 = S$,  $S^2 = Z$;   H swaps x and z,  $HZH = X$.",
        "",
        "H R(φ) H |0⟩, 1000 shots (seed 2024):",
        "`  phi     P(0)    observed`",
    ] + [f"`{label:>6s}   {row[2]:.4f}    {row[3]:.3f}`"
         for label, row in zip(["0", "π/4", "π/2", "2π/3", "π", "3π/2"], shot_rows)] + [
        "S → 1/2,  T → cos²(π/8) = 0.854,  Z → 0 (HZH = X)",
    ], spacing=0.07, fontsize=12)
    save(fig, name="06_qubits_amplitudes_and_phases.png")


if __name__ == "__main__":
    import sys

    sheets = [sheet_01_phasors_and_ac_circuits, sheet_02_linear_odes_and_the_damped_oscillator,
              sheet_03_fourier_series_and_the_dft, sheet_04_z_transform_poles_and_zeros,
              sheet_05_mandelbrot_and_julia_sets, sheet_06_qubits_amplitudes_and_phases]
    chosen = sys.argv[1:]          # optional notebook numbers, e.g. "03 05"; default: all, in order
    for sheet in sheets:
        if not chosen or sheet.__name__.split("_")[1] in chosen:
            sheet()
