"""Thermodynamic cycle for the ABFE calculations (main-text Figure 2), drawn as original vector artwork.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

# Window counts shown on the arrows -- edit here if the protocol description changes.
WINDOWS = {
    "solv_elec": "11 $\\lambda$ windows",
    "solv_vdw": "20 $\\lambda$ windows",  # one 31-state solvent schedule: 11 Coulomb + 20 LJ (shared state)
    "solv_restr": "analytical",
    "prot_restr": "12 $\\lambda$ windows",
    "prot_elec": "11 $\\lambda$ windows",
    "prot_vdw": "21 $\\lambda$ windows",
}

OUT = Path(__file__).resolve().parent.parent / "figures" / "fig3_cycle.pdf"

# Colors (dataviz reference palette + neutral grays)
WATER, WATER_EDGE = "#cde2fb", "#86b6ef"
PROT, PROT_EDGE = "#ddd6c6", "#8f8a7c"
LIG_ON, LIG_ON_EDGE = "#eb6834", "#a8431c"
LIG_OFF, LIG_OFF_EDGE = "#aaa8a0", "#6b6a65"
SPRING = "#4a3aa7"
TEXT, TEXT_2 = "#0b0b0b", "#52514e"

R = 0.46                      # water-circle radius
X_PROT, X_PLUS, X_LIG, X_CPLX = 0.0, 0.62, 1.30, 4.25
ROWS = {1: 4.95, 2: 3.3, 3: 1.65, 4: 0.0}


def protein_path(cx, cy, scale=1.0, pocket=(0.12, 0.16), notch_r=0.14):
    """Ellipse with a rounded binding pocket carved into its upper-right rim."""
    a, b = 0.34 * scale, 0.27 * scale
    c = np.array([cx + pocket[0] * scale, cy + pocket[1] * scale])
    rn = notch_r * scale
    th = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    pts = np.c_[cx + a * np.cos(th), cy + b * np.sin(th)]
    inside = np.linalg.norm(pts - c, axis=1) < rn
    idx = np.where(inside)[0]
    i0, i1 = idx.min(), idx.max()
    e, x = pts[i0 - 1], pts[(i1 + 1) % len(pts)]
    ae, ax_ = np.arctan2(*(e - c)[::-1]), np.arctan2(*(x - c)[::-1])
    inward = np.arctan2(cy - c[1], cx - c[0])
    # choose the arc direction that passes through the inward-pointing angle
    for sgn in (1, -1):
        span = (ax_ - ae) % (2 * np.pi) if sgn == 1 else -((ae - ax_) % (2 * np.pi))
        arc = ae + np.linspace(0, span, 60)
        if np.min(np.abs(np.angle(np.exp(1j * (arc - inward))))) < 0.3:
            break
    arc_pts = np.c_[c[0] + rn * np.cos(arc), c[1] + rn * np.sin(arc)]
    poly = np.vstack([pts[: i0], arc_pts, pts[i1 + 1:]])
    return poly, c


def draw_protein(ax, cx, cy, scale=1.0):
    poly, pocket = protein_path(cx, cy, scale)
    ax.add_patch(Polygon(poly, closed=True, fc=PROT, ec=PROT_EDGE, lw=1.0, zorder=2))
    return pocket


def draw_ligand(ax, cx, cy, state, size=0.062, charges=False):
    """Two fused hexagons; state in {'on', 'off', 'ghost'}."""
    hexagon = lambda ox: np.c_[ox + size * np.cos(np.pi / 6 + np.arange(6) * np.pi / 3),
                               cy + size * np.sin(np.pi / 6 + np.arange(6) * np.pi / 3)]
    dx = size * np.cos(np.pi / 6)
    style = {"on": dict(fc=LIG_ON, ec=LIG_ON_EDGE, ls="-"),
             "off": dict(fc=LIG_OFF, ec=LIG_OFF_EDGE, ls="-"),
             "ghost": dict(fc="white", ec=LIG_OFF_EDGE, ls=(0, (2, 1.5)))}[state]
    for ox in (cx - dx, cx + dx):
        ax.add_patch(Polygon(hexagon(ox), closed=True, lw=1.0, zorder=5, **style))
    if state == "on" and charges:
        ax.text(cx - 2.9 * dx, cy + 1.5 * size, "+", fontsize=8, ha="center", va="center", color=TEXT, zorder=6)
        ax.text(cx + 2.9 * dx, cy - 1.5 * size, "−", fontsize=8, ha="center", va="center", color=TEXT, zorder=6)


def draw_spring(ax, start, end, coils=5, amp=0.022):
    start, end = np.asarray(start), np.asarray(end)
    v = end - start
    n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    t = np.linspace(0, 1, 2 * coils + 3)
    offs = np.r_[0, [amp * (-1) ** k for k in range(2 * coils + 1)], 0]
    pts = start + np.outer(t, v) + np.outer(offs, n)
    ax.plot(pts[:, 0], pts[:, 1], color=SPRING, lw=1.1, zorder=6, solid_joinstyle="round")
    # three anchor atoms at the far end
    for d in (-0.035, 0, 0.035):
        ax.add_patch(Circle(end + d * n, 0.012, fc=SPRING, ec="white", lw=0.5, zorder=7))


def water(ax, cx, cy):
    ax.add_patch(Circle((cx, cy), R, fc=WATER, ec=WATER_EDGE, lw=1.0, zorder=1))


def solvent_state(ax, row, lig_state, restrained=False):
    y = ROWS[row]
    draw_protein(ax, X_PROT, y)
    ax.text(X_PLUS, y, "+", fontsize=18, fontweight="bold", ha="center", va="center", color=TEXT)
    water(ax, X_LIG, y)
    draw_ligand(ax, X_LIG, y, lig_state)
    if restrained:
        draw_spring(ax, (X_LIG + 0.08, y - 0.03), (X_LIG + 0.30, y - 0.22))


def complex_state(ax, row, lig_state, restrained):
    y = ROWS[row]
    water(ax, X_CPLX, y)
    pocket = draw_protein(ax, X_CPLX - 0.05, y - 0.05, scale=1.05)
    draw_ligand(ax, *pocket, lig_state)
    if restrained:
        draw_spring(ax, (pocket[0] - 0.02, pocket[1] - 0.04), (X_CPLX - 0.18, y - 0.16))


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=TEXT, zorder=3,
                                 shrinkA=0, shrinkB=0))


def leg_label(ax, x, y, main, sub, ha):
    ax.text(x, y + 0.07, main, fontsize=12, ha=ha, va="bottom", color=TEXT)
    ax.text(x, y - 0.02, sub, fontsize=8.5, ha=ha, va="top", color=TEXT_2, style="italic")


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(6.5, 7.6))

    for row, (lig, restr) in {1: ("on", False), 2: ("off", False), 3: ("ghost", False), 4: ("ghost", True)}.items():
        solvent_state(ax, row, lig, restr)
    for row, (lig, restr) in {1: ("on", False), 2: ("on", True), 3: ("off", True), 4: ("ghost", True)}.items():
        complex_state(ax, row, lig, restr)

    for row, letter in zip(ROWS, "abcd"):
        ax.text(X_PROT - 0.62, ROWS[row], letter, fontsize=16, fontweight="bold", ha="center", va="center", color=TEXT)
    for row, letter in zip(ROWS, "efgh"):
        ax.text(X_CPLX + 0.66, ROWS[row], letter, fontsize=16, fontweight="bold", ha="center", va="center", color=TEXT)

    top = ROWS[1] + R + 0.22
    for x, label in ((X_PROT, "Protein"), (X_LIG, "Ligand"), (X_CPLX, "Complex")):
        ax.text(x, top, label, fontsize=13, ha="center", va="bottom", color=TEXT)

    # solvent leg (down), complex leg (up)
    gap = 0.08
    legs_down = [(1, 2, r"$\Delta G^{\mathrm{solv}}_{\mathrm{elec}}$", WINDOWS["solv_elec"]),
                 (2, 3, r"$\Delta G^{\mathrm{solv}}_{\mathrm{vdW}}$", WINDOWS["solv_vdw"]),
                 (3, 4, r"$\Delta G^{\mathrm{solv}}_{\mathrm{restr}}$", WINDOWS["solv_restr"])]
    for r0, r1, lab, sub in legs_down:
        arrow(ax, (X_LIG, ROWS[r0] - R - gap), (X_LIG, ROWS[r1] + R + gap))
        leg_label(ax, X_LIG + 0.14, (ROWS[r0] + ROWS[r1]) / 2, lab, sub, "left")
    legs_up = [(4, 3, r"$\Delta G^{\mathrm{prot}}_{\mathrm{vdW}}$", WINDOWS["prot_vdw"]),
               (3, 2, r"$\Delta G^{\mathrm{prot}}_{\mathrm{elec}}$", WINDOWS["prot_elec"]),
               (2, 1, r"$\Delta G^{\mathrm{prot}}_{\mathrm{restr}}$", WINDOWS["prot_restr"])]
    for r0, r1, lab, sub in legs_up:
        arrow(ax, (X_CPLX, ROWS[r0] + R + gap), (X_CPLX, ROWS[r1] - R - gap))
        leg_label(ax, X_CPLX - 0.14, (ROWS[r0] + ROWS[r1]) / 2, lab, sub, "right")

    arrow(ax, (X_LIG + R + gap, ROWS[1]), (X_CPLX - R - gap, ROWS[1]))
    ax.text((X_LIG + X_CPLX) / 2, ROWS[1] + 0.08, r"$\Delta G_{\mathrm{bind}}$", fontsize=13, ha="center",
            va="bottom", color=TEXT)
    arrow(ax, (X_LIG + R + gap, ROWS[4]), (X_CPLX - R - gap, ROWS[4]))
    ax.text((X_LIG + X_CPLX) / 2, ROWS[4] + 0.08, r"$\Delta G = 0$", fontsize=13, ha="center", va="bottom",
            color=TEXT)

    # key
    ky = ROWS[4] - R - 0.45
    items = [("on", "fully interacting"), ("off", "charges off"), ("ghost", "decoupled")]
    for kx, (state, label) in zip((-0.55, 0.95, 2.2), items):
        draw_ligand(ax, kx, ky, state, size=0.045)
        ax.text(kx + 0.16, ky, label, fontsize=9, va="center", color=TEXT_2)
    kx = 3.45
    draw_spring(ax, (kx - 0.1, ky), (kx + 0.12, ky), coils=4, amp=0.02)
    ax.text(kx + 0.2, ky, "Boresch restraints", fontsize=9, va="center", color=TEXT_2)

    ax.set_xlim(-0.8, 5.05)
    ax.set_ylim(ky - 0.2, top + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
