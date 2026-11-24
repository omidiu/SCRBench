"""Generate the cross-task comparison figure."""

from pathlib import Path

import matplotlib.pyplot as plt

from core.experiments.enums import LLMModel

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType instead of Type 3
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]  # match the paper's Times


MODELS = [
    LLMModel.QWEN_3_235B_A22B_2507,
    LLMModel.GPT_OSS_120B,
    LLMModel.MISTRAL_SMALL_24B_INSTRUCT_2501,
    LLMModel.LLAMA_3_1_8B_INSTRUCT,
    LLMModel.GPT_4_1_NANO,
]
SCORES = {
    "T1": [99.69, 98.16, 94.71, 67.93, 55.49],
    "T2": [93.47, 97.07, 54.44, 22.67, 61.07],
    "T3": [26.82, 39.27, 25.11, 3.09, 19.96],
}
TASK_LABELS = {"T1": "T1", "T2": "T2", "T3": "T3"}
MARKERS = {"T1": "o", "T2": "s", "T3": "^"}


def main() -> None:
    output = Path(__file__).with_name("fig_cross_task.pdf")
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    y_positions = list(range(len(MODELS)))
    offsets = {"T1": 0.3, "T2": 0.0, "T3": -0.3}
    colors = {"T1": "#4C78A8", "T2": "#F58518", "T3": "#54A24B"}

    for task, scores in SCORES.items():
        ax.scatter(
            scores,
            [y + offsets[task] for y in y_positions],
            label=TASK_LABELS[task],
            color=colors[task],
            marker=MARKERS[task],
            s=12,
            zorder=3,
        )
        for score, y in zip(scores, y_positions):
            ax.annotate(
                f"{score:.2f}",
                (score, y + offsets[task]),
                xytext=(2.5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=colors[task],
            )

    for boundary in range(len(MODELS) - 1):
        ax.axhline(boundary + 0.5, color="#D9D9D9", linewidth=0.6, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([model.display_name for model in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis="both", labelsize=8, length=2.5, pad=2)
    ax.set_xlabel("Overall accuracy (%)", fontsize=8, labelpad=2)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8,
        handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
