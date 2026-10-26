"""`rag report <run>`: markdown tables and a figure from one results file.

Writes results/<run>.md (headline table, slices, paired differences between
modes, per-question rows) and docs/figures/<run>-metrics.png with its table
view <run>-metrics.md. Every figure has a table view and a legend, so no
number is carried by colour alone.
"""

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.io import write_text_atomic  # noqa: E402
from rag.eval.metrics import paired_bootstrap_diff  # noqa: E402
from rag.eval.results import ModeResult, RunResult  # noqa: E402

# Validated categorical palette (dataviz), fixed per mode so a mode keeps its
# colour across every figure.
MODE_COLOURS = {"bm25": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"}
FALLBACK_COLOUR = "#eda100"
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HEADLINE_SLICE = "overall"


def load(path: Path) -> RunResult:
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def write_report(result: RunResult, results_dir: Path, figures_dir: Path) -> list[Path]:
    name = f"{result.created_at.date().isoformat()}-{result.run}"
    report = results_dir / f"{name}.md"
    figure = figures_dir / f"{name}-metrics.png"
    write_text_atomic(report, render_markdown(result, figure.name))
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot(result, figure)
    write_text_atomic(figure.with_suffix(".md"), headline_table(result, title=True))
    return [report, figure, figure.with_suffix(".md")]


def metrics_of(result: RunResult) -> list[str]:
    for mode in result.modes.values():
        if HEADLINE_SLICE in mode.aggregates:
            return list(mode.aggregates[HEADLINE_SLICE])
    return []


def _cell(mode: ModeResult, slice_: str, metric: str) -> str:
    s = mode.aggregates.get(slice_, {}).get(metric)
    return f"{s.mean:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}]" if s else "–"


def headline_table(result: RunResult, title: bool = False) -> str:
    metrics = metrics_of(result)
    n = next(iter(result.modes.values())).aggregates.get(HEADLINE_SLICE, {})
    count = next(iter(n.values())).n if n else 0
    lines = [f"# {result.run}: retrieval metrics", ""] if title else []
    lines += [
        f"Answerable questions as asked (n = {count}); mean [95% bootstrap interval].",
        "",
        "| Mode | " + " | ".join(metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    for mode_name, mode in result.modes.items():
        cells = " | ".join(_cell(mode, HEADLINE_SLICE, m) for m in metrics)
        lines.append(f"| {mode_name} | {cells} |")
    return "\n".join(lines) + "\n"


def paired_table(result: RunResult) -> str:
    """Differences between modes on the same questions, with paired bootstrap intervals."""
    metrics = metrics_of(result)
    b = result.config["eval"]["bootstrap"]
    lines = [
        "| Comparison | " + " | ".join(metrics) + " |",
        "|---" * (len(metrics) + 1) + "|",
    ]
    for (a_name, a), (b_name, b_mode) in combinations(result.modes.items(), 2):
        cells = []
        for metric in metrics:
            xs, ys = _paired(a, b_mode, metric)
            if not xs:
                cells.append("–")
                continue
            mean, low, high = paired_bootstrap_diff(ys, xs, b["samples"], b["seed"])
            flag = "" if low <= 0 <= high else " *"
            cells.append(f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]{flag}")
        lines.append(f"| {b_name} − {a_name} | " + " | ".join(cells) + " |")
    lines += ["", "\\* the interval excludes 0."]
    return "\n".join(lines) + "\n"


def _paired(a: ModeResult, b: ModeResult, metric: str) -> tuple[list[float], list[float]]:
    """Per-question values of `metric` in both modes, headline slice only, aligned by question."""

    def by_key(mode: ModeResult) -> dict[tuple[str, str], float]:
        return {
            (r.question_id, r.variant): r.metrics[metric]
            for r in mode.questions
            if r.scored and r.variant == "raw" and metric in r.metrics
        }

    xa, xb = by_key(a), by_key(b)
    keys = sorted(xa.keys() & xb.keys())
    return [xa[k] for k in keys], [xb[k] for k in keys]


def render_markdown(result: RunResult, figure_name: str) -> str:
    metrics = metrics_of(result)
    slices = sorted({s for m in result.modes.values() for s in m.aggregates})
    lines = [
        f"# Run {result.run}",
        "",
        f"- Created: {result.created_at.isoformat(timespec='seconds')}",
        f"- Commit: `{result.git_commit}` · config hash `{result.config_hash[:12]}`",
        f"- Questions: {result.n_questions} from `{Path(result.questions_file).name}` "
        f"(sha256 `{result.questions_sha256[:12]}`)",
    ]
    for kind, meta in result.index_meta.items():
        limit = f", --limit {meta.limit}" if meta.limit else ""
        lines.append(
            f"- {kind} index: {meta.n_chunks} chunks{limit}, built {meta.built_at:%Y-%m-%d}"
        )
    lines += [
        "",
        "## Headline",
        "",
        headline_table(result),
        f"![metrics](../docs/figures/{figure_name})",
        "",
        "## Paired differences",
        "",
        paired_table(result),
        "## By slice",
        "",
    ]
    for metric in metrics:
        lines += [f"### {metric}", "", "| Slice | " + " | ".join(result.modes) + " | n |"]
        lines.append("|---" * (len(result.modes) + 2) + "|")
        for s in slices:
            cells = " | ".join(_cell(m, s, metric) for m in result.modes.values())
            n = max(
                (
                    m.aggregates.get(s, {}).get(metric).n
                    for m in result.modes.values()
                    if metric in m.aggregates.get(s, {})
                ),
                default=0,
            )
            lines.append(f"| {s} | {cells} | {n} |")
        lines.append("")
    lines += ["## Per question", ""]
    lines += [
        "| Question | Type | Variant | "
        + " | ".join(f"{mode} {metric}" for mode in result.modes for metric in metrics[:2])
        + " |",
        "|---" * (3 + 2 * len(result.modes)) + "|",
    ]
    first = next(iter(result.modes.values()))
    for i, row in enumerate(first.questions):
        cells = []
        for mode in result.modes.values():
            r = mode.questions[i]
            cells += [f"{r.metrics[m]:.2f}" if m in r.metrics else "–" for m in metrics[:2]]
        lines.append(
            f"| {row.question_id} | {row.type} | {row.variant} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def plot(result: RunResult, path: Path) -> None:
    """Grouped bars: one group per metric, one bar per mode, 95% interval whiskers."""
    metrics = metrics_of(result)
    modes = list(result.modes)
    width = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for j, mode_name in enumerate(modes):
        agg = result.modes[mode_name].aggregates.get(HEADLINE_SLICE, {})
        xs = [i + (j - (len(modes) - 1) / 2) * width for i in range(len(metrics))]
        means = [agg[m].mean if m in agg else 0 for m in metrics]
        errs = [
            [max(0.0, agg[m].mean - agg[m].ci_low) if m in agg else 0 for m in metrics],
            [max(0.0, agg[m].ci_high - agg[m].mean) if m in agg else 0 for m in metrics],
        ]
        ax.bar(
            xs,
            means,
            width * 0.92,
            color=MODE_COLOURS.get(mode_name, FALLBACK_COLOUR),
            label=mode_name,
            edgecolor=SURFACE,
            linewidth=2,
        )
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=TEXT_MUTED, elinewidth=1, capsize=3)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1.05)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{result.run}: retrieval on answerable questions", loc="left", color=TEXT, fontsize=12
    )
    ax.set_ylabel("mean over questions (95% interval)", color=TEXT_MUTED, fontsize=9)
    # Below the axis: at the top it would sit on the whiskers that reach 1.0.
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        ncols=len(modes),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
    )
    ax.text(
        1,
        1.02,
        "dup: lower is better",
        transform=ax.transAxes,
        ha="right",
        color=TEXT_MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
