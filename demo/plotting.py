import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

GENE_COLORS = ["#D87867", "#2C6DB2", "#2A9D8F"]
INK, MUTED, FAINT = "#111111", "#5F6368", "#9AA0A6"
CELL_FILL, NUC_FILL, OUTLINE, OTHER_TX = "#F1F3F4", "#DADDE0", "#5F6368", "#BCC0C4"


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
    })


def gene_colors(genes):
    if len(genes) > len(GENE_COLORS):
        raise ValueError(f"the figures show at most {len(GENE_COLORS)} genes at a time")
    return dict(zip(genes, GENE_COLORS))


def ramp(color):
    """Sequential map for one gene: white -> the gene's colour -> a darker shade of it."""
    rgb = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list("g", [(1, 1, 1), rgb, rgb * 0.55])


def _italic_legend(leg):
    for t in leg.get_texts():
        t.set_fontstyle("italic")


def _outline(ax, cb, nb, cell, fill=True, lw=1.0, color=OUTLINE):
    c = cb[cb.cell == cell]
    n = nb[nb.cell == cell]
    if fill:
        ax.fill(c.x, c.y, fc=CELL_FILL, ec="none", zorder=0)
        ax.fill(n.x, n.y, fc=NUC_FILL, ec="none", zorder=0)
    ax.plot(np.r_[c.x, c.x.iloc[:1]], np.r_[c.y, c.y.iloc[:1]], color=color, lw=lw, zorder=1)
    ax.plot(np.r_[n.x, n.x.iloc[:1]], np.r_[n.y, n.y.iloc[:1]], color=FAINT, lw=0.8, zorder=1)
    ax.plot(c.centerX.iloc[0], c.centerY.iloc[0], "+", color=MUTED, ms=7, mew=1.2, zorder=4)


def _frame_cell(ax, cb, cell, pad=40):
    c = cb[cb.cell == cell]
    ax.set_xlim(c.x.min() - pad, c.x.max() + pad)
    ax.set_ylim(c.y.max() + pad, c.y.min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _unit_circle(ax, center=0.0, radius=1.0):
    ax.add_patch(Circle((center, center), radius, fill=False, ec=FAINT, lw=1.0, ls="--", zorder=0))


def _map(ax, m, cmap, vmax, n_pix, discrete=False):
    """A registered map with the unit circle drawn on its pixel grid. discrete=True: one colour per integer count."""
    if discrete:
        norm = BoundaryNorm(np.arange(-0.5, max(int(vmax), 1) + 1), cmap.N)
        im = ax.imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    _unit_circle(ax, center=(n_pix - 1) / 2, radius=n_pix / 2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(FAINT); s.set_linewidth(0.6)
    return im


def _colorbar(fig, ax, im):
    cbar = fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]))
    cbar.set_label("transcripts per bin", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    if isinstance(im.norm, BoundaryNorm):
        n = int(im.norm.boundaries[-1] - 0.5)
        cbar.set_ticks(np.arange(0, n + 1, int(np.ceil((n + 1) / 8))))
        cbar.minorticks_off()
    return cbar


# ------------------------------------------------------------------------------------------- Part 1
def raw_data(tx, cb, nb, cell, genes):
    col = gene_colors(genes)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.08})
    a = ax[0]
    for c in cb.cell.unique():
        _outline(a, cb, nb, c, lw=1.6 if c == cell else 0.6, color=INK if c == cell else FAINT)
        g = cb[cb.cell == c]
        a.text(g.centerX.iloc[0], g.centerY.iloc[0] - 70, c, ha="center", va="bottom", fontsize=9,
               color=INK if c == cell else MUTED, fontweight="bold" if c == cell else "normal")
    a.scatter(tx.x, tx.y, s=0.02, color=OTHER_TX, rasterized=True, zorder=0.5)
    a.set_aspect("equal"); a.invert_yaxis(); a.axis("off")
    a.set_title(f"Field of view: {tx.cell.nunique()} cells, {len(tx):,} molecules", loc="left")

    sub = tx[tx.cell == cell]
    b = ax[1]
    _outline(b, cb, nb, cell)
    b.scatter(sub.x, sub.y, s=0.6, color=MUTED, alpha=0.6, lw=0, rasterized=True, zorder=2)
    _frame_cell(b, cb, cell)
    b.set_title(f"Cell {cell}: {len(sub):,} molecules, {sub.gene.nunique():,} genes", loc="left")

    c = ax[2]
    _outline(c, cb, nb, cell)
    for g in genes:
        s = sub[sub.gene == g]
        c.scatter(s.x, s.y, s=9, color=col[g], lw=0.3, ec="white", label=g, zorder=3)
    _frame_cell(c, cb, cell)
    c.set_title("Molecules of two genes", loc="left")
    _italic_legend(c.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), markerscale=1.6, handletextpad=0.3))
    handles = [Rectangle((0, 0), 1, 1, fc=CELL_FILL, ec=OUTLINE), Rectangle((0, 0), 1, 1, fc=NUC_FILL, ec=FAINT),
               Line2D([], [], marker="+", ls="", color=MUTED, ms=7, mew=1.2)]
    fig.legend(handles, ["cell segmentation", "nucleus segmentation", "nuclear center"], loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, 0.03), fontsize=10)
    return fig


# ------------------------------------------------------------------------------------------- Part 2
def registration(registered, cb, nb, cells, genes):
    col = gene_colors(genes)
    fig, ax = plt.subplots(2, len(cells), figsize=(4.3 * len(cells), 8.4), gridspec_kw={"hspace": 0.12, "wspace": 0.05})
    for j, cell in enumerate(cells):
        sub = registered[registered.cell == cell]
        a = ax[0, j]
        _outline(a, cb, nb, cell)
        for g in genes:
            s = sub[sub.gene == g]
            a.scatter(s.x, s.y, s=7, color=col[g], lw=0.25, ec="white", zorder=3, label=g)
        _frame_cell(a, cb, cell)
        a.set_title(f"Cell {cell}", loc="center")
        b = ax[1, j]
        _unit_circle(b)
        for g in genes:
            s = sub[sub.gene == g]
            b.scatter(s.x_norm, s.y_norm, s=7, color=col[g], lw=0.25, ec="white", zorder=3)
        b.plot(0, 0, "+", color=MUTED, ms=7, mew=1.2)
        b.set_xlim(-1.08, 1.08); b.set_ylim(1.08, -1.08); b.set_aspect("equal"); b.axis("off")
    ax[0, 0].text(-0.06, 0.5, "Original\ncoordinates", transform=ax[0, 0].transAxes, ha="right", va="center", fontsize=12)
    ax[1, 0].text(-0.06, 0.5, "Registered\n(unit circle)", transform=ax[1, 0].transAxes, ha="right", va="center", fontsize=12)
    _italic_legend(fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(genes),
                              bbox_to_anchor=(0.5, 0.0), markerscale=3, fontsize=14))
    return fig


# ------------------------------------------------------------------------------------------- Part 3
def representations(registered, maps48, maps12, cell, genes):
    """maps48, maps12: (n_genes, 48, 48) and (n_genes, 12, 12) for one cell."""
    col = gene_colors(genes)
    fig, ax = plt.subplots(len(genes), 3, figsize=(12, 3.3 * len(genes)), squeeze=False,
                           gridspec_kw={"wspace": 0.45, "hspace": 0.15})
    sub = registered[registered.cell == cell]
    for i, g in enumerate(genes):
        s = sub[sub.gene == g]
        a = ax[i, 0]
        _unit_circle(a)
        a.scatter(s.x_norm, s.y_norm, s=6, color=col[g], lw=0)
        a.set_xlim(-1.02, 1.02); a.set_ylim(1.02, -1.02); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(FAINT); sp.set_linewidth(0.6)
        for k, (m, n_pix) in enumerate([(maps48[i], 48), (maps12[i], 12)], start=1):
            _colorbar(fig, ax[i, k], _map(ax[i, k], m, ramp(col[g]), m.max(), n_pix, discrete=True))
        ax[i, 0].text(-0.08, 0.5, g, transform=ax[i, 0].transAxes, ha="right", va="center",
                      fontsize=13, fontstyle="italic", fontweight="bold")
    for k, t in enumerate(["Registered molecules", "48 × 48 count grid", "12 × 12 map (4 × 4 sum pooling)"]):
        ax[0, k].set_title(t)
    return fig


# ------------------------------------------------------------------------------------------- Part 5
def observed_vs_predicted(cell, genes, cell_names, gene_names, truth, mu, tx, sampled, cb, nb):
    """truth, mu: (n_cells, n_genes, 12, 12) observed counts and predicted NB mean."""
    col = gene_colors(genes)
    ci = list(cell_names).index(cell)
    fig, ax = plt.subplots(len(genes), 4, figsize=(16, 3.6 * len(genes)), squeeze=False,
                           gridspec_kw={"wspace": 0.3, "hspace": 0.15})
    for i, g in enumerate(genes):
        obs = truth[ci, gene_names.index(g)].astype(float)
        pred = mu[ci, gene_names.index(g)]
        vmax = max(obs.max(), pred.max())
        _map(ax[i, 0], obs, ramp(col[g]), vmax, 12)
        _colorbar(fig, ax[i, 1], _map(ax[i, 1], pred, ramp(col[g]), vmax, 12))
        for k, (df, xc, yc) in enumerate([(tx, "x", "y"), (sampled, "x_original", "y_original")], start=2):
            a = ax[i, k]
            _outline(a, cb, nb, cell)
            s = df[(df.cell == cell) & (df.gene == g)]
            a.scatter(s[xc], s[yc], s=7, color=col[g], lw=0.25, ec="white", zorder=3)
            _frame_cell(a, cb, cell)
        ax[i, 0].text(-0.1, 0.5, g, transform=ax[i, 0].transAxes, ha="right", va="center",
                      fontsize=13, fontstyle="italic", fontweight="bold")
    for k, t in enumerate(["Observed counts", r"Predicted NB mean $\hat{\mu}$", "Observed molecules",
                           "Sampled transcripts"]):
        ax[0, k].set_title(t)
    return fig
