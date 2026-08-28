"""Generate publication figures for the repositioned manuscript."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import FIGURES, RESULTS

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C = {"deep": "#2F5D8A", "rf": "#B4553C", "base": "#9AA5B1",
     "obs": "#3D7A5A", "imp": "#C1873B"}

TN = {"Attack": "Attack type\n(11 classes)", "pattern": "Pattern\n(11 classes)",
      "victim.country": "Country\n(189 classes)"}
ORDER = ["Attack", "pattern", "victim.country"]


# ---------------------------------------------------- Fig 2: representation
def fig_representation():
    old = {"Attack": (14.00, 1.59), "pattern": (81.51, 4.21), "victim.country": (1.79, 0.17)}
    new = {"Attack": (32.57, 2.62), "pattern": (88.95, 3.36), "victim.country": (4.95, 0.63)}
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.bar(x - w/2, [old[t][0] for t in ORDER], w, yerr=[old[t][1] for t in ORDER],
           label="Label encoding + scaling", color=C["base"], capsize=3, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, [new[t][0] for t in ORDER], w, yerr=[new[t][1] for t in ORDER],
           label="Entity embeddings", color=C["deep"], capsize=3, edgecolor="black", linewidth=0.4)
    for i, t in enumerate(ORDER):
        d = new[t][0] - old[t][0]
        ax.annotate(f"+{d:.1f}", (i, max(new[t][0], old[t][0]) + 5),
                    ha="center", fontsize=8.5, fontweight="bold", color=C["deep"])
    ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in ORDER])
    ax.set_ylabel("Macro F1 (%)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.savefig(str(FIGURES) + "/fig_representation.png"); plt.close(fig)


# ------------------------------------------------------- Fig 3: circularity
def fig_circularity():
    obs = {"Attack": (26.58, 1.32), "pattern": (84.24, 3.10)}
    imp = {"Attack": (70.41, 5.84), "pattern": (94.27, 5.05)}
    keys = ["Attack", "pattern"]
    lbl = ["Attack type", "Pattern"]
    x = np.arange(2); w = 0.34
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.bar(x - w/2, [obs[k][0] for k in keys], w, yerr=[obs[k][1] for k in keys],
           label="Observed labels", color=C["obs"], capsize=3, edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, [imp[k][0] for k in keys], w, yerr=[imp[k][1] for k in keys],
           label="RF-imputed labels", color=C["imp"], capsize=3, edgecolor="black", linewidth=0.4)
    for i, k in enumerate(keys):
        d = imp[k][0] - obs[k][0]
        ax.annotate(f"+{d:.1f}", (i, imp[k][0] + 7), ha="center",
                    fontsize=8.5, fontweight="bold", color=C["imp"])
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_ylabel("Macro F1 (%)"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.savefig(str(FIGURES) + "/fig_circularity.png"); plt.close(fig)


# --------------------------------------------------------- Fig 4: ablation
def fig_ablation():
    labs = ["All\nfeatures", "Minus\naction.*", "Minus action/actor/\nasset/attribute"]
    pat = [(87.05, 4.48), (81.46, 4.31), (43.30, 1.34)]
    cty = [(5.73, 0.76), (5.28, 1.01), (4.38, 0.56)]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for ax, dat, name, col in [(axes[0], pat, "Pattern (11 classes)", C["deep"]),
                               (axes[1], cty, "Country (189 classes)", C["rf"])]:
        ax.bar(x, [d[0] for d in dat], 0.55, yerr=[d[1] for d in dat],
               color=col, capsize=3, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.6)
        ax.set_title(name); ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_ylabel("Macro F1 (%)")
    axes[0].set_ylim(0, 100); axes[1].set_ylim(0, 10)
    axes[0].annotate("−43.8", (2, 50), ha="center", fontsize=9,
                     fontweight="bold", color=C["deep"])
    fig.savefig(str(FIGURES) + "/fig_ablation.png"); plt.close(fig)


# ------------------------------------------------------- Fig 5: cardinality
def fig_cardinality():
    acc = [60.94, 95.60, 66.61]
    mf1 = [32.57, 88.95, 4.95]
    maj = [32.87, 38.86, 48.89]
    x = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.bar(x - w, acc, w, label="Accuracy", color=C["deep"], edgecolor="black", linewidth=0.4)
    ax.bar(x, mf1, w, label="Macro F1", color=C["rf"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w, maj, w, label="Majority-class accuracy", color=C["base"],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([TN[t] for t in ORDER])
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.annotate("accuracy–macro F1 gap\nwidens with cardinality",
                xy=(2, 20), xytext=(1.35, 62), fontsize=7.8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))
    fig.savefig(str(FIGURES) + "/fig_cardinality.png"); plt.close(fig)


# ---------------------------------------------------- Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.35)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def box(x, y, w, h, txt, col, fs=7.4):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, alpha=.30,
                                   edgecolor="black", lw=.7))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=.9, color="black"))

    # sources (stacked left)
    box(0.00, 0.60, 0.19, 0.28, "VERIS\n1971–2017\n7,745 incidents", C["base"])
    box(0.00, 0.14, 0.19, 0.28, "Hackmageddon\n2018–2023\n12,534 incidents", C["base"])

    # harmonisation
    box(0.26, 0.37, 0.19, 0.28, "Schema\nharmonisation\n+ taxonomy\nrecovery", C["deep"])
    arrow(0.19, 0.74, 0.26, 0.58)
    arrow(0.19, 0.28, 0.26, 0.44)

    # corpus
    box(0.52, 0.37, 0.17, 0.28, "Unified corpus\n20,279 × 64", C["deep"])
    arrow(0.45, 0.51, 0.52, 0.51)

    # embeddings + models
    box(0.76, 0.37, 0.22, 0.28,
        "Entity embeddings\n→ MLP / CNN / LSTM\n+ RL weighting", C["rf"])
    arrow(0.69, 0.51, 0.76, 0.51)

    ax.text(0.87, 0.28, "attack type · country · pattern",
            ha="center", fontsize=7.2, style="italic")
    ax.text(0.605, 0.30, "evaluation protocol:\nper-source split\n+ feature ablation",
            ha="center", va="top", fontsize=6.9, style="italic", color="#444444")
    fig.savefig(str(FIGURES) + "/fig_pipeline.png"); plt.close(fig)



# ------------------------------------------- Fig 4: masked-imputation control
def fig_control():
    """Confounded (Section 7.3) vs controlled (Section 7.4) inflation."""
    src = pd.read_csv(RESULTS / "caveats_source.csv").drop_duplicates(
        subset=["target", "seed", "stratum"] if "stratum" in
        pd.read_csv(RESULTS / "caveats_source.csv").columns else None)
    ctl = pd.read_csv(RESULTS / "mask_impute_results.csv").drop_duplicates(
        subset=["target", "seed", "stratum"])

    def pair(df, target, obs_key, imp_key, col="label"):
        d = df[df.target == target]
        o = d[d[col] == obs_key].macro_f1
        i = d[d[col] == imp_key].macro_f1
        return (o.mean(), o.std()), (i.mean(), i.std())

    conf_a = pair(src, "Attack", "observed", "imputed")
    conf_p = pair(src, "pattern", "observed", "imputed")
    ctl_a = pair(ctl, "Attack", "observed", "reconstructed", col="stratum")
    ctl_p = pair(ctl, "pattern", "observed", "reconstructed", col="stratum")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, (name, conf, ct) in zip(axes, [("Attack type", conf_a, ctl_a),
                                           ("Pattern", conf_p, ctl_p)]):
        x = np.arange(2); w = 0.34
        ax.bar(x - w/2, [conf[0][0], ct[0][0]], w,
               yerr=[conf[0][1], ct[0][1]], label="Observed labels",
               color=C["obs"], capsize=3, edgecolor="black", linewidth=.4)
        ax.bar(x + w/2, [conf[1][0], ct[1][0]], w,
               yerr=[conf[1][1], ct[1][1]], label="Reconstructed labels",
               color=C["imp"], capsize=3, edgecolor="black", linewidth=.4)
        for i, (o, im) in enumerate([(conf[0][0], conf[1][0]), (ct[0][0], ct[1][0])]):
            ax.annotate(f"+{im - o:.1f}", (i, im + 8), ha="center",
                        fontsize=8.5, fontweight="bold", color=C["imp"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Confounded\n(era differs)", "Control\n(era constant)"],
                           fontsize=8)
        ax.set_title(name); ax.grid(axis="y", alpha=.25, linewidth=.5)
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("Macro F1 (%)")
    axes[0].legend(frameon=False, loc="upper center", ncol=2,
                   bbox_to_anchor=(1.06, 1.20), fontsize=8.5)
    fig.savefig(str(FIGURES) + "/fig_control.png"); plt.close(fig)


# --------------------------------------------- Fig 7: chronological evaluation
def fig_chrono():
    """Random-split (observed stratum) vs chronological attack-type macro F1."""
    src = pd.read_csv(RESULTS / "caveats_source.csv")
    chr_ = pd.read_csv(RESULTS / "chrono_results.csv").drop_duplicates(
        subset=["target", "config", "model", "seed"])
    d = src[(src.target == "Attack") & (src.label == "observed")].macro_f1
    ch = chr_[(chr_.target == "Attack") & (chr_.config == "observed")]

    deep = [(d.mean(), d.std()),
            (ch[ch.model == "EmbedMLP"].macro_f1.mean(),
             ch[ch.model == "EmbedMLP"].macro_f1.std())]
    rf_v = [(35.46, 1.07),
            (ch[ch.model == "RandomForest"].macro_f1.mean(),
             ch[ch.model == "RandomForest"].macro_f1.std())]
    maj = [(4.50, 0.03),
           (ch[ch.model == "MajorityClass"].macro_f1.mean(),
            ch[ch.model == "MajorityClass"].macro_f1.std())]

    x = np.arange(2); w = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w, [v[0] for v in deep], w, yerr=[v[1] for v in deep],
           label="Deep model", color=C["deep"], capsize=3, edgecolor="black", linewidth=.4)
    ax.bar(x, [v[0] for v in rf_v], w, yerr=[v[1] for v in rf_v],
           label="Random Forest", color=C["rf"], capsize=3, edgecolor="black", linewidth=.4)
    ax.bar(x + w, [v[0] for v in maj], w, yerr=[v[1] for v in maj],
           label="Majority class", color=C["base"], capsize=3, edgecolor="black", linewidth=.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random split\n(observed stratum)",
                        "Chronological\n(train\u22642020, test 2023)"], fontsize=8.5)
    ax.set_ylabel("Macro F1 (%)"); ax.set_ylim(0, 44)
    ax.legend(frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.16), fontsize=8.5)
    ax.grid(axis="y", alpha=.25, linewidth=.5)
    fig.savefig(str(FIGURES) + "/fig_chrono.png"); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); fig_representation(); fig_circularity()
    fig_ablation(); fig_cardinality()
    fig_control(); fig_chrono()
    print("figures written")
