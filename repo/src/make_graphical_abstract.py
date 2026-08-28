"""Graphical abstract for Elsevier submission.

Elsevier requires a single image, minimum 531 x 1328 px (h x w), sized for
display at 200 x 500 px. Output here is 1992 x 797 px, comfortably above the
minimum and at the required aspect ratio.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FIGURES

plt.rcParams.update({
    "font.family": "serif", "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

OBS, IMP, DEEP, GREY = "#3D7A5A", "#C1873B", "#2F5D8A", "#9AA5B1"

fig = plt.figure(figsize=(9.96, 3.98))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.95], wspace=0.42,
                      left=0.03, right=0.98, top=0.86, bottom=0.13)

# ---------------------------------------------------------------- panel 1
ax = fig.add_subplot(gs[0]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Merging incident repositories", fontsize=10.5, fontweight="bold", pad=10)


def box(x, y, w, h, txt, col, fs=7.6, alpha=.32):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=col, alpha=alpha,
                               edgecolor="black", lw=.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.1, color="black"))


box(0.02, 0.66, 0.30, 0.22, "VERIS\n1971–2017", GREY)
box(0.02, 0.34, 0.30, 0.22, "Hackmageddon\n2018–2023", GREY)
box(0.44, 0.50, 0.28, 0.22, "harmonise\n+ impute\nmissing fields", DEEP)
arr(0.32, 0.77, 0.44, 0.66)
arr(0.32, 0.45, 0.44, 0.56)
box(0.80, 0.50, 0.18, 0.22, "20,279\nincidents", DEEP)
arr(0.72, 0.61, 0.80, 0.61)
ax.text(0.50, 0.14, "≈ half of all labels become\nmodel outputs over the same\nfeatures used to predict them",
        ha="center", va="center", fontsize=7.4, style="italic", color="#8A2B12")

# ---------------------------------------------------------------- panel 2
ax2 = fig.add_subplot(gs[1])
ax2.set_title("Pooled metrics hide it", fontsize=10.5, fontweight="bold", pad=10)
x = np.arange(2); w = 0.36
ax2.bar(x - w / 2, [26.58, 24.03], w, yerr=[1.32, 2.07], color=OBS,
        capsize=3, edgecolor="black", lw=.5, label="observed labels")
ax2.bar(x + w / 2, [70.41, 40.89], w, yerr=[5.84, 7.16], color=IMP,
        capsize=3, edgecolor="black", lw=.5, label="imputed labels")
ax2.annotate("+43.8", (0, 79), ha="center", fontsize=9, fontweight="bold", color=IMP)
ax2.annotate("+16.9", (1, 50), ha="center", fontsize=9, fontweight="bold", color=IMP)
ax2.set_xticks(x)
ax2.set_xticklabels(["as merged\n(era differs)", "controlled\n(era constant)"], fontsize=8)
ax2.set_ylabel("Macro F1 (%)", fontsize=9)
ax2.set_ylim(0, 92)
ax2.tick_params(labelsize=8)
ax2.legend(frameon=False, fontsize=7.4, loc="upper right")
ax2.grid(axis="y", alpha=.25, lw=.5)

# ---------------------------------------------------------------- panel 3
ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_title("A protocol that surfaces it", fontsize=10.5, fontweight="bold", pad=10)
items = [
    ("1", "Provenance stratification", "report metrics separately for\nobserved and imputed labels"),
    ("2", "Feature-group ablation", "detect targets that are\nderived taxonomy rules"),
    ("3", "Masked-imputation control", "isolate imputation from\ncorrelated era effects"),
]
for i, (n, title, sub) in enumerate(items):
    y = 0.76 - i * 0.30
    ax3.add_patch(plt.Circle((0.08, y), 0.055, facecolor=DEEP, alpha=.35,
                             edgecolor="black", lw=.8))
    ax3.text(0.08, y, n, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax3.text(0.19, y + 0.045, title, ha="left", va="center", fontsize=8.4, fontweight="bold")
    ax3.text(0.19, y - 0.045, sub, ha="left", va="center", fontsize=7.2, color="#333333")

fig.savefig(str(FIGURES) + "/graphical_abstract.png")
plt.close(fig)

from PIL import Image
im = Image.open(str(FIGURES) + "/graphical_abstract.png")
print("graphical abstract:", im.size, "px  (Elsevier minimum 1328 x 531)")
