"""`rag eval answer-report`: what the chat's answers add up to (Phase 8, D-046).

Reads the answers (`rag eval answers`) and Sharad's labels (`rag eval label`).
Without labels it reports what code can count: abstention, uncited claims,
invalid citations, the threshold sweep. With labels it adds groundedness,
correctness and risk.

The retrieval gate was off during the run, so any threshold t is simulated:
a question is answered at t unless its top dense cosine is below t or the
model said NOT_SUPPORTED. That is exactly what the engine does, except that in
a conversation a refusal would also change the next turn's history.

A false-premise question counts as handled when the chat abstains, or when it
answers and Sharad labels the answer correct (it corrected the premise).
"""

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from rag.common.schemas import StrictModel
from rag.eval.answers import AnswerRecord, AnswersMeta
from rag.eval.labels import Label
from rag.eval.metrics import bootstrap_ci
from rag.eval.results import MetricSummary, RunResult

BAD_ANSWERS = {"wrong"}
CLAIM_VALUES = ("supported", "partial", "unsupported")


class Labels:
    """Sharad's labels, looked up by question."""

    def __init__(self, labels: Sequence[Label]) -> None:
        self.claims: dict[str, dict[int, str]] = {}
        self.answer: dict[str, str] = {}
        self.rewrite: dict[str, str] = {}
        for lb in labels:
            if lb.kind == "claim" and lb.claim is not None:
                self.claims.setdefault(lb.question_id, {})[lb.claim] = lb.value
            elif lb.kind == "answer":
                self.answer[lb.question_id] = lb.value
            else:
                self.rewrite[lb.question_id] = lb.value

    def __bool__(self) -> bool:
        return bool(self.claims or self.answer or self.rewrite)


def abstains(r: AnswerRecord, threshold: float) -> str | None:
    """Which gate refuses this question at the threshold: "gate", "model" or None."""
    if r.top_dense_cosine is None or r.top_dense_cosine < threshold:
        return "gate"
    return "model" if r.model_abstained else None


def handled_unanswerable(r: AnswerRecord, threshold: float, labels: Labels) -> bool | None:
    """Did the chat handle a question the corpus can't answer? None: needs a label."""
    if abstains(r, threshold):
        return True
    if r.unanswerable_kind == "false_premise":
        value = labels.answer.get(r.question_id)
        return None if value is None else value == "correct"
    return False


def bad_answer(r: AnswerRecord, labels: Labels) -> bool | None:
    """An answered question counts against risk if it's wrong or has an unsupported claim."""
    if not r.answerable:
        return None  # judged by handled_unanswerable instead
    answer = labels.answer.get(r.question_id)
    claims = labels.claims.get(r.question_id, {})
    if answer is None and not claims:
        return None
    return answer in BAD_ANSWERS or "unsupported" in claims.values()


class SweepPoint(StrictModel):
    threshold: float
    coverage: float  # share of all questions answered
    unanswerable_handled: float | None  # share of unanswerable ones handled (None: unlabelled)
    false_abstention: float  # share of answerable ones refused
    risk: float | None  # share of answered answerable questions judged bad
    risk_n: int  # answered answerable questions with labels


def sweep(
    records: Sequence[AnswerRecord], thresholds: Sequence[float], labels: Labels
) -> list[SweepPoint]:
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    out = []
    for t in thresholds:
        answered = [r for r in records if not abstains(r, t)]
        handled = [handled_unanswerable(r, t, labels) for r in unanswerable]
        judged = [b for r in answered if (b := bad_answer(r, labels)) is not None]
        out.append(
            SweepPoint(
                threshold=round(t, 4),
                coverage=len(answered) / len(records) if records else 0.0,
                unanswerable_handled=(
                    sum(map(bool, handled)) / len(handled)
                    if handled and None not in handled
                    else None
                ),
                false_abstention=(
                    sum(bool(abstains(r, t)) for r in answerable) / len(answerable)
                    if answerable
                    else 0.0
                ),
                risk=sum(judged) / len(judged) if judged else None,
                risk_n=len(judged),
            )
        )
    return out


class Failure(StrictModel):
    question_id: str
    what: str  # "false abstention", "missed abstention", "wrong answer", "unsupported claim", ...
    stage: str  # suggested: retrieval | rewrite | gate | generation (Sharad decides)
    evidence: list[str]
    inspect: str


def failures(records: Sequence[AnswerRecord], threshold: float, labels: Labels) -> list[Failure]:
    out = []
    for r in records:
        refused = abstains(r, threshold)
        problems = []
        if r.answerable and refused:
            problems.append(f"false abstention ({refused})")
        if not r.answerable and handled_unanswerable(r, threshold, labels) is False:
            problems.append("missed abstention")
        if r.answerable and not refused:
            value = labels.answer.get(r.question_id)
            if value in ("wrong", "partial"):
                problems.append(f"{value} answer")
            n = Counter(labels.claims.get(r.question_id, {}).values())["unsupported"]
            if n:
                problems.append(f"{n} unsupported claim(s)")
            if r.invalid_citations:
                problems.append(f"invalid citations {r.invalid_citations}")
        if not problems:
            continue
        out.append(
            Failure(
                question_id=r.question_id,
                what="; ".join(problems),
                stage=_stage(r, refused, labels),
                evidence=_evidence(r, labels),
                inspect=f'uv run rag inspect query "{r.rewrite}"',
            )
        )
    return out


def _stage(r: AnswerRecord, refused: str | None, labels: Labels) -> str:
    # A follow-up sent on unrewritten ("they", "that approach") fails downstream,
    # whichever gate then refuses it: the rewrite is the first stage that went wrong.
    if _unrewritten(r) or labels.rewrite.get(r.question_id) == "different":
        return "rewrite"
    if refused == "gate":
        return "gate"
    if r.answerable and not r.gold_papers_in_context:
        return "retrieval"
    return "generation"


def _evidence(r: AnswerRecord, labels: Labels) -> list[str]:
    cosine = "n/a" if r.top_dense_cosine is None else f"{r.top_dense_cosine:.3f}"
    lines = [f"top dense cosine {cosine}"]
    if r.gold_passages_in_context is not None:
        lines.append(
            f"gold passages in context {r.gold_passages_in_context:.2f}, "
            f"gold papers {r.gold_papers_in_context:.2f}"
        )
    if _unrewritten(r):
        lines.append("follow-up retrieved with the user's words: the rewrite returned it unchanged")
    elif r.rewrite != r.user:
        rewrite = labels.rewrite.get(r.question_id, "unlabelled")
        lines.append(f"rewrite ({rewrite}): {r.rewrite!r}; oracle: {r.oracle_rewrite!r}")
    if r.model_abstained:
        lines.append("the model answered NOT_SUPPORTED")
    return lines


def _unrewritten(r: AnswerRecord) -> bool:
    return r.type == "followup" and r.rewrite.strip() == r.user.strip()


class Rate(StrictModel):
    count: int
    total: int
    rate: float | None


def rate(count: int, total: int) -> Rate:
    return Rate(count=count, total=total, rate=count / total if total else None)


class AnswerReport(StrictModel):
    created_at: datetime
    answers_file: str
    answers_commit: str
    model: str
    labelled: bool
    deployed_threshold: float
    n_questions: int
    abstention: dict[str, Rate]
    gates: dict[str, int]  # which gate refused, at the deployed threshold
    uncited_claims: Rate
    invalid_citations: int
    claims: dict[str, Rate] = {}  # supported / partial / unsupported, of labelled claims
    correct: MetricSummary | None = None  # answered answerable questions labelled correct
    rewrites: dict[str, int] = {}  # equivalent / different
    followup_retrieval: dict[str, dict[str, float]] = {}  # variant -> metric -> mean
    followup_run: str | None = None
    sweep: list[SweepPoint]
    failures: list[Failure]


def build(
    records: Sequence[AnswerRecord],
    meta: AnswersMeta,
    answers_file: str,
    labels: Labels,
    thresholds: Sequence[float],
    bootstrap: tuple[int, int],
    followups: RunResult | None = None,
    followup_run: str | None = None,
) -> AnswerReport:
    t = meta.deployed_threshold
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    answered = [r for r in records if not abstains(r, t)]
    handled = [handled_unanswerable(r, t, labels) for r in unanswerable]
    claims = [(r, c) for r in answered for c in r.claims]
    values = [labels.claims.get(r.question_id, {}).get(c.index) for r, c in claims]
    labelled_claims = [v for v in values if v]
    correct_values = [
        float(labels.answer[r.question_id] == "correct")
        for r in answered
        if r.answerable and r.question_id in labels.answer
    ]
    correct = None
    if correct_values:
        mean, low, high = bootstrap_ci(correct_values, *bootstrap)
        correct = MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(correct_values))
    return AnswerReport(
        created_at=datetime.now(UTC),
        answers_file=answers_file,
        answers_commit=meta.git_commit,
        model=meta.model,
        labelled=bool(labels),
        deployed_threshold=t,
        n_questions=len(records),
        abstention={
            "unanswerable handled": rate(sum(h is True for h in handled), len(handled)),
            "unanswerable awaiting a label": rate(sum(h is None for h in handled), len(handled)),
            "false abstention (answerable)": rate(
                sum(bool(abstains(r, t)) for r in answerable), len(answerable)
            ),
        },
        gates=dict(Counter(g for r in records if (g := abstains(r, t)))),
        uncited_claims=rate(sum(not c.cites for _, c in claims), len(claims)),
        invalid_citations=sum(len(r.invalid_citations) for r in answered),
        claims={v: rate(labelled_claims.count(v), len(labelled_claims)) for v in CLAIM_VALUES}
        if labelled_claims
        else {},
        correct=correct,
        rewrites=dict(Counter(labels.rewrite.values())),
        followup_retrieval=_followup_retrieval(followups),
        followup_run=followup_run,
        sweep=sweep(records, thresholds, labels),
        failures=failures(records, t, labels),
    )


def _followup_retrieval(run: RunResult | None) -> dict[str, dict[str, float]]:
    if run is None:
        return {}
    mode = next(iter(run.modes.values()))
    return {
        name.split(":", 1)[1]: {m: s.mean for m, s in metrics.items() if "recall" in m}
        for name, metrics in mode.aggregates.items()
        if name.startswith("followup:")
    }


def _pct(r: Rate) -> str:
    return f"{r.count}/{r.total}" + (f" ({r.rate:.0%})" if r.rate is not None else "")


def markdown(report: AnswerReport) -> str:
    t = report.deployed_threshold
    lines = [
        "# Answer evaluation",
        "",
        f"Answers: `{report.answers_file}` (commit {report.answers_commit[:12]}, {report.model}),"
        f" {report.n_questions} questions. Judge: Sharad, by hand (D-046)."
        + ("" if report.labelled else " **No labels yet**: code-only numbers below."),
        "",
        f"## At the deployed threshold ({t:.2f})",
        "",
        "| Measure | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {_pct(value)} |" for name, value in report.abstention.items()]
    lines += [
        f"| refused by | {', '.join(f'{k} {v}' for k, v in report.gates.items()) or 'none'} |",
        f"| uncited claims (code) | {_pct(report.uncited_claims)} |",
        f"| invalid citations (code) | {report.invalid_citations} |",
    ]
    lines += [f"| {k} claims (labelled) | {_pct(v)} |" for k, v in report.claims.items()]
    if report.correct:
        c = report.correct
        lines.append(
            f"| answers correct (labelled) | {c.mean:.2f} [{c.ci_low:.2f}, {c.ci_high:.2f}],"
            f" n={c.n} |"
        )
    if report.rewrites:
        lines.append(
            "| follow-up rewrites | "
            + ", ".join(f"{k} {v}" for k, v in report.rewrites.items())
            + " |"
        )
    if report.followup_retrieval:
        lines += [
            "",
            f"## Follow-up retrieval: raw vs oracle vs system rewrite (`{report.followup_run}`)",
            "",
            "| Variant | " + " | ".join(next(iter(report.followup_retrieval.values()))) + " |",
            "|---|" + "---|" * len(next(iter(report.followup_retrieval.values()))),
        ]
        for variant, metrics in report.followup_retrieval.items():
            lines.append(
                f"| {variant} | " + " | ".join(f"{v:.3f}" for v in metrics.values()) + " |"
            )
    lines += [
        "",
        "## Threshold sweep (simulated; gate off during the run)",
        "",
        "| threshold | coverage | unanswerable handled | false abstention | risk (n) |",
        "|---|---|---|---|---|",
    ]
    for p in report.sweep:
        handled = "–" if p.unanswerable_handled is None else f"{p.unanswerable_handled:.2f}"
        risk = "–" if p.risk is None else f"{p.risk:.2f} ({p.risk_n})"
        mark = " ←" if abs(p.threshold - t) < 1e-9 else ""
        lines.append(
            f"| {p.threshold:.2f}{mark} | {p.coverage:.2f} | {handled} |"
            f" {p.false_abstention:.2f} | {risk} |"
        )
    n_unanswerable = report.abstention["unanswerable handled"].total
    lines += [
        "",
        f"Only {n_unanswerable} unanswerable questions: the sweep shows the trade-off, it can't"
        " tune a threshold (any choice would be fitted to a handful of questions). Risk = share"
        " of answered, answerable questions labelled wrong or with an unsupported claim.",
        "",
        "## Failure candidates",
        "",
        "Suggested stage only: the analysis is Sharad's (docs/phase8-failures.md).",
        "",
    ]
    for f in report.failures:
        lines.append(f"- **{f.question_id}**: {f.what}. Suggested stage: {f.stage}.")
        lines += [f"  - {e}" for e in f.evidence]
        lines.append(f"  - `{f.inspect}`")
    if not report.failures:
        lines.append("None at the deployed threshold" + ("." if report.labelled else " (yet)."))
    lines += ["", "Generated by `rag eval answer-report`."]
    return "\n".join(lines) + "\n"


def plot(report: AnswerReport, path) -> None:
    """Threshold vs coverage and abstention; risk vs coverage once labels exist."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, text, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, green, orange = "#2a78d6", "#1baf7a", "#eb6834"
    xs = [p.threshold for p in report.sweep]
    panels = 2 if report.labelled else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4.2), facecolor=surface)
    axes = list(axes) if panels > 1 else [axes]
    left = axes[0]
    left.plot(xs, [p.coverage for p in report.sweep], color=blue, lw=2, label="coverage")
    left.plot(
        xs,
        [p.false_abstention for p in report.sweep],
        color=orange,
        lw=2,
        label="false abstention (answerable)",
    )
    handled = [p.unanswerable_handled for p in report.sweep]
    if all(h is not None for h in handled):
        left.plot(xs, handled, color=green, lw=2, label="unanswerable handled")
    left.axvline(report.deployed_threshold, color=muted, ls="--", lw=1)
    left.set_xlabel("retrieval gate threshold (top dense cosine)", color=muted)
    left.set_title("What the gate threshold trades", loc="left", color=text, fontsize=11)
    left.legend(frameon=False, fontsize=8, labelcolor=text)
    if panels > 1:
        right = axes[1]
        points = [p for p in report.sweep if p.risk is not None]
        right.plot([p.coverage for p in points], [p.risk for p in points], color=blue, lw=2)
        right.scatter([p.coverage for p in points], [p.risk for p in points], color=blue, s=12)
        right.set_xlabel("coverage (share answered)", color=muted)
        right.set_ylabel("risk (answered, judged bad)", color=muted)
        right.set_title("Risk–coverage", loc="left", color=text, fontsize=11)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.set_ylim(0, 1.02)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
