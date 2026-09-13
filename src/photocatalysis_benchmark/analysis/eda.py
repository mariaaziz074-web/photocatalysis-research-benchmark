"""
src/photocatalysis_benchmark/analysis/eda.py
===========================================

Publication-quality exploratory figures using saturated, high-contrast,
colorblind-conscious colors.

Each figure is exported as:
    1. 600-DPI PNG
    2. Vector PDF
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("eda_analysis")


# ---------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#111111",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "#FFFFFF",
        "savefig.edgecolor": "none",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "grid.color": "#B8B8B8",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# Saturated Okabe-Ito colors.
# These have stronger contrast than Seaborn's muted/pastel palettes.
VIVID_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8A2BE2",  # strong violet
    "#000000",  # black
]

DEEP_BLUE = "#0057B8"
DEEP_GREEN = "#008A5A"
VERMILLION = "#D43F00"
DARK_EDGE = "#161616"

# Saturated diverging map. White is retained at zero so weak correlations
# are not visually exaggerated.
VIVID_DIVERGING = LinearSegmentedColormap.from_list(
    "vivid_blue_white_red",
    ["#0033A0", "#FFFFFF", "#C00000"],
    N=256,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """
    Save both a high-resolution PNG and a vector PDF.

    Parameters
    ----------
    fig
        Matplotlib figure.
    output_path
        PNG destination. A PDF with the same stem is also created.
    """
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    plt.close(fig)

    logger.info("Saved PNG figure: %s", output_path)
    logger.info("Saved vector figure: %s", pdf_path)


def _light_source_palette(df: pd.DataFrame) -> dict[str, str]:
    """Create a stable saturated color map for light-source categories."""
    categories = sorted(
        df["light_source_type"].dropna().astype(str).unique().tolist()
    )

    return {
        category: VIVID_COLORS[index % len(VIVID_COLORS)]
        for index, category in enumerate(categories)
    }


# ---------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------

def plot_figure_1_dataset_overview(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Figure 1: dataset composition and target distribution."""
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10))

    # Panel a: pollutant distribution
    dye_counts = df["dye_name"].value_counts().head(8)

    y_positions = np.arange(len(dye_counts))

    axs[0, 0].barh(
        y_positions,
        dye_counts.values,
        color=DEEP_BLUE,
        edgecolor=DARK_EDGE,
        linewidth=0.8,
        alpha=1.0,
    )
    axs[0, 0].set_yticks(y_positions)
    axs[0, 0].set_yticklabels(dye_counts.index)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_xlim(0, max(dye_counts.max() * 1.15, 1))

    for position, value in enumerate(dye_counts.values):
        axs[0, 0].text(
            value + dye_counts.max() * 0.015,
            position,
            str(value),
            va="center",
            ha="left",
            fontsize=9,
            color=DARK_EDGE,
            fontweight="bold",
        )

    axs[0, 0].set_title(
        "a) Model organic pollutant representation",
        loc="left",
    )
    axs[0, 0].set_xlabel("Number of experiments")
    axs[0, 0].set_ylabel("Pollutant")
    axs[0, 0].grid(axis="x", linestyle="--")
    axs[0, 0].grid(axis="y", visible=False)

    # Panel b: catalyst distribution
    catalyst_counts = df["catalyst_name"].value_counts().head(8)
    y_positions = np.arange(len(catalyst_counts))

    axs[0, 1].barh(
        y_positions,
        catalyst_counts.values,
        color=DEEP_GREEN,
        edgecolor=DARK_EDGE,
        linewidth=0.8,
        alpha=1.0,
    )
    axs[0, 1].set_yticks(y_positions)
    axs[0, 1].set_yticklabels(catalyst_counts.index)
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_xlim(0, max(catalyst_counts.max() * 1.15, 1))

    for position, value in enumerate(catalyst_counts.values):
        axs[0, 1].text(
            value + catalyst_counts.max() * 0.015,
            position,
            str(value),
            va="center",
            ha="left",
            fontsize=9,
            color=DARK_EDGE,
            fontweight="bold",
        )

    axs[0, 1].set_title(
        "b) Photocatalyst-system representation",
        loc="left",
    )
    axs[0, 1].set_xlabel("Number of experiments")
    axs[0, 1].set_ylabel("Photocatalyst")
    axs[0, 1].grid(axis="x", linestyle="--")
    axs[0, 1].grid(axis="y", visible=False)

    # Panel c: light-source distribution
    light_counts = df["light_source_type"].value_counts()
    light_palette = _light_source_palette(df)
    pie_colors = [light_palette[str(name)] for name in light_counts.index]

    axs[1, 0].pie(
        light_counts.values,
        labels=light_counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={
            "fontsize": 9,
            "color": DARK_EDGE,
            "fontweight": "bold",
        },
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.5,
        },
    )
    axs[1, 0].set_title(
        "c) Irradiation-source distribution",
        loc="left",
    )

    # Panel d: efficiency distribution
    sns.histplot(
        data=df,
        x="degradation_efficiency_percent",
        bins=25,
        kde=True,
        color=VERMILLION,
        alpha=1.0,
        edgecolor=DARK_EDGE,
        linewidth=0.6,
        line_kws={
            "linewidth": 2.5,
        },
        ax=axs[1, 1],
    )

    axs[1, 1].set_title(
        "d) Degradation-efficiency distribution",
        loc="left",
    )
    axs[1, 1].set_xlabel("Degradation efficiency (%)")
    axs[1, 1].set_ylabel("Number of experiments")
    axs[1, 1].grid(axis="y", linestyle="--")
    axs[1, 1].grid(axis="x", visible=False)

    fig.tight_layout(pad=2.0)
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------

def plot_figure_2_chemical_space(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Figure 2: catalyst-property and experimental operating spaces."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))

    panel_a = df.dropna(
        subset=[
            "surface_area_m2g",
            "bandgap_ev",
            "degradation_efficiency_percent",
        ]
    )

    scatter_a = axs[0].scatter(
        panel_a["surface_area_m2g"],
        panel_a["bandgap_ev"],
        c=panel_a["degradation_efficiency_percent"],
        cmap="plasma",
        s=58,
        alpha=0.95,
        edgecolors=DARK_EDGE,
        linewidths=0.45,
    )

    colorbar_a = fig.colorbar(scatter_a, ax=axs[0], pad=0.02)
    colorbar_a.set_label("Degradation efficiency (%)")

    axs[0].set_title(
        "a) Catalyst electronic and textural space",
        loc="left",
    )
    axs[0].set_xlabel(r"BET surface area, $S_{\mathrm{BET}}$ (m$^2$ g$^{-1}$)")
    axs[0].set_ylabel(r"Bandgap energy, $E_{\mathrm{g}}$ (eV)")
    axs[0].grid(True, linestyle="--")

    panel_b = df.dropna(
        subset=[
            "catalyst_dosage_gl",
            "initial_dye_conc_mgl",
            "rate_constant_k_min1",
        ]
    )

    scatter_b = axs[1].scatter(
        panel_b["catalyst_dosage_gl"],
        panel_b["initial_dye_conc_mgl"],
        c=panel_b["rate_constant_k_min1"],
        cmap="viridis",
        s=58,
        alpha=0.95,
        edgecolors=DARK_EDGE,
        linewidths=0.45,
    )

    colorbar_b = fig.colorbar(scatter_b, ax=axs[1], pad=0.02)
    colorbar_b.set_label(r"Rate constant, $k_{\mathrm{obs}}$ (min$^{-1}$)")

    axs[1].set_title(
        "b) Experimental operating space",
        loc="left",
    )
    axs[1].set_xlabel("Catalyst dosage (g L$^{-1}$)")
    axs[1].set_ylabel("Initial pollutant concentration (mg L$^{-1}$)")
    axs[1].grid(True, linestyle="--")

    fig.tight_layout(pad=1.5)
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------

def plot_figure_3_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Figure 3: Spearman descriptor and target correlation matrix."""
    feature_labels = {
        "bandgap_ev": "Bandgap",
        "surface_area_m2g": "BET area",
        "catalyst_dosage_gl": "Catalyst dose",
        "initial_dye_conc_mgl": "Initial concentration",
        "ph": "pH",
        "light_intensity_mwcm2": "Light intensity",
        "reaction_time_min": "Reaction time",
        "dye_molecular_weight": "Molecular weight",
        "degradation_efficiency_percent": "Efficiency",
        "rate_constant_k_min1": "Rate constant",
    }

    valid_columns = [
        column for column in feature_labels if column in df.columns
    ]

    numeric_data = df[valid_columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_data.corr(method="spearman")
    correlation = correlation.rename(
        index=feature_labels,
        columns=feature_labels,
    )

    upper_triangle = np.triu(
        np.ones_like(correlation, dtype=bool),
        k=1,
    )

    fig, ax = plt.subplots(figsize=(10.2, 8.5))

    sns.heatmap(
        correlation,
        mask=upper_triangle,
        cmap=VIVID_DIVERGING,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={
            "fontsize": 8.5,
            "fontweight": "bold",
        },
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "shrink": 0.82,
            "label": r"Spearman correlation, $\rho$",
        },
        ax=ax,
    )

    ax.set_title(
        "Spearman rank correlations among benchmark descriptors",
        loc="left",
        pad=14,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
    )

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------

def plot_figure_4_kinetics_overview(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Figure 4: rate constants and degradation response by light source."""
    figure_data = df.dropna(
        subset=[
            "light_source_type",
            "rate_constant_k_min1",
            "reaction_time_min",
            "degradation_efficiency_percent",
        ]
    ).copy()

    light_order = (
        figure_data.groupby("light_source_type")["rate_constant_k_min1"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    categorical_palette = {
        category: VIVID_COLORS[index % len(VIVID_COLORS)]
        for index, category in enumerate(light_order)
    }

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.2))

    sns.boxplot(
        data=figure_data,
        x="light_source_type",
        y="rate_constant_k_min1",
        order=light_order,
        palette=categorical_palette,
        saturation=1.0,
        width=0.62,
        linewidth=1.2,
        fliersize=3.5,
        boxprops={
            "edgecolor": DARK_EDGE,
            "linewidth": 1.2,
        },
        whiskerprops={
            "color": DARK_EDGE,
            "linewidth": 1.2,
        },
        capprops={
            "color": DARK_EDGE,
            "linewidth": 1.2,
        },
        medianprops={
            "color": "#FFFFFF",
            "linewidth": 2.0,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#FFFFFF",
            "markeredgecolor": DARK_EDGE,
            "markersize": 4,
            "alpha": 1.0,
        },
        ax=axs[0],
    )

    axs[0].set_title(
        "a) Apparent rate constants by light source",
        loc="left",
    )
    axs[0].set_xlabel("Irradiation regime")
    axs[0].set_ylabel(r"Rate constant, $k_{\mathrm{obs}}$ (min$^{-1}$)")
    axs[0].tick_params(axis="x", rotation=15)
    axs[0].grid(axis="y", linestyle="--")
    axs[0].grid(axis="x", visible=False)

    sns.lineplot(
        data=figure_data,
        x="reaction_time_min",
        y="degradation_efficiency_percent",
        hue="light_source_type",
        hue_order=light_order,
        palette=categorical_palette,
        estimator="mean",
        errorbar=("ci", 95),
        err_style="band",
        err_kws={
            "alpha": 0.14,
        },
        linewidth=2.6,
        marker="o",
        markersize=6,
        markeredgecolor=DARK_EDGE,
        markeredgewidth=0.45,
        ax=axs[1],
    )

    axs[1].set_title(
        "b) Degradation response with irradiation time",
        loc="left",
    )
    axs[1].set_xlabel("Irradiation duration (min)")
    axs[1].set_ylabel("Mean degradation efficiency (%)")
    axs[1].grid(True, linestyle="--")

    axs[1].legend(
        title="Light source",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor=DARK_EDGE,
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
    )

    fig.tight_layout(pad=1.6)
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------
# Complete EDA suite
# ---------------------------------------------------------------------

def generate_all_eda_visualizations(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Generate the complete publication EDA figure suite."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_figure_1_dataset_overview(
        df,
        figures_dir / "fig1_dataset_overview.png",
    )
    plot_figure_2_chemical_space(
        df,
        figures_dir / "fig2_chemical_space.png",
    )
    plot_figure_3_correlation_heatmap(
        df,
        figures_dir / "fig3_correlation_heatmap.png",
    )
    plot_figure_4_kinetics_overview(
        df,
        figures_dir / "fig4_kinetics_overview.png",
    )

    logger.info(
        "Generated four publication figures as 600-DPI PNG and vector PDF."
    )