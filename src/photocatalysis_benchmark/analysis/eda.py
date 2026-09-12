"""
src/photocatalysis_benchmark/analysis/eda.py
===========================================
Exploratory Data Analysis and publication-grade visualization suite for Scientific Data.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for CPU
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("eda_analysis")

# Configure Nature Portfolio publication aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4
})


def plot_figure_1_dataset_overview(df: pd.DataFrame, output_path: Path):
    """
    Figure 1: Dataset Composition & Provenance Overview
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Dye Distribution
    dye_counts = df["dye_name"].value_counts().head(8)
    sns.barplot(x=dye_counts.values, y=dye_counts.index, ax=axs[0, 0], palette="Blues_r")
    axs[0, 0].set_title("a) Model Organic Pollutant Representation", loc="left", fontweight="bold")
    axs[0, 0].set_xlabel("Number of Experiments")
    axs[0, 0].set_ylabel("Pollutant Name")

    # Panel B: Catalyst Formulation Breakdown
    cat_counts = df["catalyst_name"].value_counts().head(8)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axs[0, 1], palette="Greens_r")
    axs[0, 1].set_title("b) Photocatalyst System Distribution", loc="left", fontweight="bold")
    axs[0, 1].set_xlabel("Number of Experiments")
    axs[0, 1].set_ylabel("Photocatalyst Composition")

    # Panel C: Light Source Distribution
    light_counts = df["light_source_type"].value_counts()
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    axs[1, 0].pie(light_counts.values, labels=light_counts.index, autopct="%1.1f%%", startangle=140, colors=colors[:len(light_counts)])
    axs[1, 0].set_title("c) Irradiation Source Classification", loc="left", fontweight="bold")

    # Panel D: Degradation Efficiency Distribution
    sns.histplot(df["degradation_efficiency_percent"], kde=True, ax=axs[1, 1], color="#8172B2", bins=25)
    axs[1, 1].set_title("d) Degradation Efficiency Distribution", loc="left", fontweight="bold")
    axs[1, 1].set_xlabel("Degradation Efficiency (%)")
    axs[1, 1].set_ylabel("Count")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Figure 1 saved to {output_path}")


def plot_figure_2_chemical_space(df: pd.DataFrame, output_path: Path):
    """
    Figure 2: Experimental Feature Space & Catalyst Properties
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bandgap vs. Surface Area by Phase
    scatter = axs[0].scatter(
        df["surface_area_m2g"],
        df["bandgap_ev"],
        c=df["degradation_efficiency_percent"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="k",
        linewidth=0.5,
        s=50
    )
    cbar = plt.colorbar(scatter, ax=axs[0])
    cbar.set_label("Degradation Efficiency (%)")
    axs[0].set_title("a) Catalyst Electronic & Textural Space", loc="left", fontweight="bold")
    axs[0].set_xlabel(r"BET Surface Area ($S_{\mathrm{BET}}$, $\mathrm{m}^2/\mathrm{g}$)")
    axs[0].set_ylabel(r"Bandgap Energy ($E_g$, $\mathrm{eV}$)")
    axs[0].grid(True, linestyle="--")

    # Panel B: Dye Concentration vs. Catalyst Dosage
    scatter2 = axs[1].scatter(
        df["catalyst_dosage_gl"],
        df["initial_dye_conc_mgl"],
        c=df["rate_constant_k_min1"],
        cmap="plasma",
        alpha=0.75,
        edgecolors="k",
        linewidth=0.5,
        s=50
    )
    cbar2 = plt.colorbar(scatter2, ax=axs[1])
    cbar2.set_label(r"Rate Constant $k$ ($\mathrm{min}^{-1}$)")
    axs[1].set_title("b) Reaction Operating Space", loc="left", fontweight="bold")
    axs[1].set_xlabel(r"Catalyst Dosage ($C_{\mathrm{cat}}$, $\mathrm{g}/\mathrm{L}$)")
    axs[1].set_ylabel(r"Initial Dye Concentration ($C_0$, $\mathrm{mg}/\mathrm{L}$)")
    axs[1].grid(True, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Figure 2 saved to {output_path}")


def plot_figure_3_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Figure 3: Correlation Matrix of Experimental Descriptors & Targets
    """
    feature_cols = [
        "bandgap_ev", "surface_area_m2g", "catalyst_dosage_gl",
        "initial_dye_conc_mgl", "ph", "light_intensity_mwcm2",
        "reaction_time_min", "dye_molecular_weight",
        "degradation_efficiency_percent", "rate_constant_k_min1"
    ]
    valid_cols = [c for c in feature_cols if c in df.columns]
    corr_df = df[valid_cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation ($\\rho$)"},
        ax=ax
    )
    ax.set_title("Spearman Rank Correlation Across Benchmark Descriptors", loc="left", fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Figure 3 saved to {output_path}")


def plot_figure_4_kinetics_overview(df: pd.DataFrame, output_path: Path):
    """
    Figure 4: Reaction Kinetics and Degradation Trajectories
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Rate constant distribution by Light Source
    sns.boxplot(
        x="light_source_type",
        y="rate_constant_k_min1",
        data=df,
        ax=axs[0],
        palette="Set2"
    )
    axs[0].set_title("a) Apparent Rate Constants by Light Source", loc="left", fontweight="bold")
    axs[0].set_xlabel("Irradiation Regime")
    axs[0].set_ylabel(r"Rate Constant $k_{\mathrm{obs}}$ ($\mathrm{min}^{-1}$)")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: Degradation Efficiency vs Reaction Time
    sns.lineplot(
        x="reaction_time_min",
        y="degradation_efficiency_percent",
        hue="light_source_type",
        data=df,
        ax=axs[1],
        palette="Set2",
        errorbar="sd"
    )
    axs[1].set_title("b) Degradation Kinetics Time-Profile", loc="left", fontweight="bold")
    axs[1].set_xlabel("Irradiation Duration (min)")
    axs[1].set_ylabel("Mean Degradation Efficiency (%)")
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Figure 4 saved to {output_path}")


def generate_all_eda_visualizations(df: pd.DataFrame, figures_dir: Path):
    """
    Run complete EDA figure generation suite.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_figure_1_dataset_overview(df, figures_dir / "fig1_dataset_overview.png")
    plot_figure_2_chemical_space(df, figures_dir / "fig2_chemical_space.png")
    plot_figure_3_correlation_heatmap(df, figures_dir / "fig3_correlation_heatmap.png")
    plot_figure_4_kinetics_overview(df, figures_dir / "fig4_kinetics_overview.png")
    logger.info("All 4 publication figures successfully generated.")