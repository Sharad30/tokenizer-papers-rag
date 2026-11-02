"""`rag eval ...`: build and check the eval set (Phase 3), score runs (Phase 4)."""

from pathlib import Path
from typing import Annotated

import typer

from rag.collect.manifest import read_manifest
from rag.common.cli import LimitOption, get_config
from rag.common.config import Config
from rag.eval import draft as draft_mod
from rag.eval import find as find_mod
from rag.eval import validate as validate_mod
from rag.eval.ablation import run_ablation
from rag.eval.spans import ChunkTexts
from rag.parse.pipeline import CHUNKS, step_dir

app = typer.Typer(
    help="Eval set tooling (find spans, validate questions) and eval runs.",
    no_args_is_help=True,
)


def _chunk_texts(config: Config) -> ChunkTexts:
    path = step_dir(config, "chunk") / CHUNKS
    if not path.is_file():
        typer.echo(f"No chunks at {path}: run `rag parse all` first.", err=True)
        raise typer.Exit(1)
    return ChunkTexts.from_file(path)


@app.command()
def find(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Text to look for (case and quotes ignored)")],
    paper: Annotated[str | None, typer.Option(help="Only this record id")] = None,
    all_words: Annotated[
        bool, typer.Option("--all-words", help="Match chunks containing every word, any order")
    ] = False,
) -> None:
    """Grep the chunks for text to copy a gold span from. Unranked, in corpus order."""
    config = get_config(ctx)
    result = find_mod.find(
        _chunk_texts(config),
        text,
        max_hits=config.eval.find.max_hits,
        paper=paper,
        all_words=all_words,
    )
    typer.echo(find_mod.render(result, text))


def _corpus(config: Config) -> draft_mod.Corpus:
    """Manifest + chunks, with each work's canonical record when the works exist."""
    from rag.retrieve.collapse import WorkMap, WorksUnavailable

    try:
        canonical = WorkMap.load(config.paths.works, step_dir(config, "chunk") / CHUNKS).canonical
    except WorksUnavailable:
        canonical = set()
    return draft_mod.Corpus(read_manifest(config.paths.manifest), _chunk_texts(config), canonical)


def _draft_path(config: Config, file: Path | None) -> Path:
    path = file or config.paths.eval / config.eval.draft.file
    if not path.is_file():
        typer.echo(f"No draft at {path}.", err=True)
        raise typer.Exit(1)
    return path


DraftFile = Annotated[Path | None, typer.Option(help="Draft file (default: eval.draft.file)")]


@app.command("check-papers")
def check_papers(ctx: typer.Context, file: DraftFile = None) -> None:
    """Match every drafted gold paper to the corpus: by arXiv id, other id, then title."""
    config = get_config(ctx)
    questions = draft_mod.read_draft(_draft_path(config, file))
    corpus = _corpus(config)
    rows = draft_mod.check_papers(questions, corpus, config.eval.draft.title_min_ratio)
    typer.echo(draft_mod.render_check(rows, corpus))
    if any(m.status in ("missing", "not_indexed") for _, _, m in rows):
        raise typer.Exit(1)


@app.command()
def resolve(
    ctx: typer.Context,
    file: DraftFile = None,
    id: Annotated[  # noqa: A002 - the question id, as the draft names it
        str | None, typer.Option("--id", help="Only this question (also redoes resolved spans)")
    ] = None,
) -> None:
    """Pick a verbatim gold span for each drafted gold entry, interactively.

    Lists the paper's chunks containing the entry's span_search terms, in
    corpus order; you open one, pick sentences or type the span, and it is
    written only if it maps to a chunk. Saves after every change (.bak kept).
    """
    config = get_config(ctx)
    path = _draft_path(config, file)
    questions = draft_mod.read_draft(path)
    changed = draft_mod.resolve(
        questions,
        _corpus(config),
        config.eval.draft,
        ask=lambda prompt: typer.prompt(prompt, default="", show_default=False),
        say=typer.echo,
        save=lambda qs: draft_mod.write_draft(path, qs),
        only=id,
    )
    typer.echo(f"\n{changed} change(s) saved to {path}")


@app.command()
def absent(
    ctx: typer.Context,
    file: DraftFile = None,
    id: Annotated[str | None, typer.Option("--id", help="Only this question")] = None,  # noqa: A002
    paper: Annotated[
        str | None, typer.Option(help="Search only this record (a question about one paper)")
    ] = None,
    write: Annotated[
        bool, typer.Option("--write/--no-write", help="Ask to confirm and record each check")
    ] = True,
) -> None:
    """Search the corpus for each unanswerable question's verify_absent terms."""
    config = get_config(ctx)
    path = _draft_path(config, file)
    questions = draft_mod.read_draft(path)
    texts = _chunk_texts(config)
    for qi, q in enumerate(questions):
        if q.answerable or (id and q.id != id):
            continue
        if not write:
            hits = draft_mod.absent_hits(q.verify_absent, texts, config.eval.draft, paper)
            typer.echo(draft_mod.render_absent(q, hits, paper))
            continue
        checked = draft_mod.check_absent(
            q,
            texts,
            config.eval.draft,
            ask=lambda prompt: typer.prompt(prompt, default="", show_default=False),
            say=typer.echo,
            paper=paper,
        )
        if checked is not None:
            questions[qi] = checked
            draft_mod.write_draft(path, questions)


@app.command()
def finalize(
    ctx: typer.Context,
    file: DraftFile = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing eval set")] = False,
) -> None:
    """Write eval/gold/questions.jsonl from draft entries marked status=verified, then validate."""
    from rag.common.io import write_jsonl

    config = get_config(ctx)
    out = config.paths.eval / config.eval.questions_file
    if out.exists() and not force:
        typer.echo(f"{out} exists: rerun with --force to replace it.", err=True)
        raise typer.Exit(1)
    result = draft_mod.finalize(draft_mod.read_draft(_draft_path(config, file)))
    if not result.questions:
        typer.echo(draft_mod.render_finalized(result, out))
        typer.echo("Nothing verified yet: set status to verified in the draft first.", err=True)
        raise typer.Exit(1)
    write_jsonl(out, result.questions)
    typer.echo(draft_mod.render_finalized(result, out))
    indexed = {r.record_id: bool(r.included_in_index) for r in read_manifest(config.paths.manifest)}
    report = validate_mod.validate_file(out, _chunk_texts(config), indexed, config.eval)
    typer.echo(validate_mod.render(report, out))
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def answers(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Run every eval question through the chat (gate off) and save the answers (Phase 8)."""
    from rag.chat.engine import ChatEngine
    from rag.chat.llm import make_llm
    from rag.chat.prompts import load_prompt
    from rag.common.io import file_sha256, write_json, write_jsonl
    from rag.common.provenance import config_hash, git_commit
    from rag.eval import answers as answers_mod
    from rag.eval.identity import identity_groups, load_pairs
    from rag.eval.run import results_path
    from rag.retrieve.retriever import Retriever

    config = get_config(ctx)
    qpath = config.paths.eval / config.eval.questions_file
    questions, errors = validate_mod.load_questions(qpath)
    if errors:
        typer.echo(f"{qpath} has invalid lines; run `rag eval validate`.", err=True)
        raise typer.Exit(1)
    if limit:
        questions = questions[:limit]
    chat = config.chat
    deployed = chat.abstain.min_dense_cosine
    # Gate off for the run: the report simulates every threshold (D-046).
    chat = chat.model_copy(
        update={"abstain": chat.abstain.model_copy(update={"min_dense_cosine": -1.0})}
    )
    prompts = [load_prompt(config.paths.prompts, chat.prompts.answer)]
    prompts.append(load_prompt(config.paths.prompts, chat.prompts.rewrite))
    llm = make_llm(chat)
    manifest = read_manifest(config.paths.manifest)
    engine = ChatEngine(
        Retriever.load(config, {"hybrid"}),
        llm,
        {r.record_id: r for r in manifest},
        chat,
        *prompts,
    )
    groups = identity_groups(manifest, load_pairs(config.paths.eval / config.eval.pairs_file))
    out = results_path(config, "answers").with_suffix(".jsonl")
    done: list = []

    def keep(record) -> None:  # saved after every answer: a crash loses nothing
        done.append(record)
        write_jsonl(out, done)
        state = "abstained" if record.model_abstained else f"{len(record.claims)} claims"
        cosine = record.top_dense_cosine
        shown = "n/a" if cosine is None else f"{cosine:.3f}"
        typer.echo(f"{record.question_id}: {state} · cosine {shown}")

    answers_mod.run(questions, engine, _chunk_texts(config), groups, on_answer=keep)
    write_json(
        out.with_suffix(".json"),
        answers_mod.meta(
            questions,
            deployed,
            git_commit=git_commit(),
            config_hash=config_hash(config),
            model=llm.model,
            prompts={p.name: p.sha256 for p in prompts},
            questions_file=str(qpath.relative_to(config.paths.eval.parent)),
            questions_sha256=file_sha256(qpath),
        ),
    )
    typer.echo(f"{len(done)} answers -> {out}")


def _answers_path(config: Config, answers: Path | None) -> Path:
    if answers:
        return answers
    found = sorted(config.paths.results.glob("*-answers.jsonl"))
    if not found:
        typer.echo("No answers yet: run `rag eval answers` first.", err=True)
        raise typer.Exit(1)
    return found[-1]


AnswersFile = Annotated[
    Path | None, typer.Option("--answers", help="Answers file (default: the newest)")
]


@app.command()
def label(
    ctx: typer.Context,
    answers: AnswersFile = None,
    id: Annotated[str | None, typer.Option("--id", help="Only this question")] = None,  # noqa: A002
) -> None:
    """Judge the chat's answers by hand: each claim, each answer, each follow-up rewrite."""
    from rag.common.io import read_jsonl
    from rag.eval import labels as labels_mod
    from rag.eval.answers import AnswerRecord

    config = get_config(ctx)
    path = _answers_path(config, answers)
    records = read_jsonl(path, AnswerRecord)
    questions, _ = validate_mod.load_questions(config.paths.eval / config.eval.questions_file)
    gists = {q.id: [g.answer_gist for g in q.gold if g.answer_gist] for q in questions}
    out = labels_mod.labels_path(path)
    labels = labels_mod.read_labels(out)
    added = labels_mod.label_all(
        records,
        gists,
        _chunk_texts(config),
        labels,
        ask=lambda prompt: typer.prompt(prompt, default="", show_default=False),
        say=typer.echo,
        save=labels_mod.save_to(out),
        only=id,
    )
    typer.echo(f"\n{added} label(s) added; {len(labels)} in {out}")


@app.command()
def validate(
    ctx: typer.Context,
    file: Annotated[
        Path | None, typer.Option(help="Questions file (default: eval.questions_file)")
    ] = None,
) -> None:
    """Check the eval set: schema, papers indexed, spans map to chunks, conversations, mix."""
    config = get_config(ctx)
    path = file or config.paths.eval / config.eval.questions_file
    indexed = {r.record_id: bool(r.included_in_index) for r in read_manifest(config.paths.manifest)}
    report = validate_mod.validate_file(path, _chunk_texts(config), indexed, config.eval)
    typer.echo(validate_mod.render(report, path))
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def run(
    ctx: typer.Context,
    name: Annotated[str, typer.Option(help="Run name: results/<date>-<name>.json")],
    modes: Annotated[str, typer.Option(help="Comma-separated: bm25,dense,hybrid")] = (
        "bm25,dense,hybrid"
    ),
    file: Annotated[
        Path | None, typer.Option(help="Questions file (default: eval.questions_file)")
    ] = None,
    limit: LimitOption = None,
    rewrite: Annotated[
        bool,
        typer.Option("--rewrite", help="Also score follow-ups rewritten by the chat's LLM"),
    ] = False,
) -> None:
    """Score retrieval modes on the eval set; write a results file for `rag report`."""
    from rag.eval.run import run_eval

    chosen = [m.strip() for m in modes.split(",") if m.strip()]
    unknown = set(chosen) - {"bm25", "dense", "hybrid"}
    if unknown or not chosen:
        raise typer.BadParameter(f"unknown modes: {sorted(unknown)}", param_hint="--modes")
    config = get_config(ctx)
    rewriter = None
    if rewrite:
        from rag.chat.llm import make_llm
        from rag.chat.prompts import load_prompt

        prompt = load_prompt(config.paths.prompts, config.chat.prompts.rewrite)
        rewriter = (make_llm(config.chat), prompt)
    out = run_eval(config, name, chosen, file, limit, rewriter=rewriter)
    typer.echo(f"Wrote {out}" if out else "No questions yet: nothing scored, nothing written.")


@app.command()
def ablate(
    ctx: typer.Context,
    name: Annotated[str, typer.Option(help="Which ablation in eval.ablations")] = "dedup",
) -> None:
    """Score each configured condition on the probe queries (and the eval set, if any)."""
    config = get_config(ctx)
    if name not in config.eval.ablations:
        typer.echo(f"No ablation {name!r} in config eval.ablations", err=True)
        raise typer.Exit(1)
    out = run_ablation(config, name)
    typer.echo(out.with_suffix(".md").read_text(encoding="utf-8"))
    typer.echo(f"Results: {out}")


@app.command()
def silver(ctx: typer.Context) -> None:
    """LLM-written questions from sampled chunks -> eval/silver/ (not gold; D-044)."""
    from rag.chat.llm import make_llm
    from rag.chat.prompts import load_prompt
    from rag.common.io import read_jsonl
    from rag.dedup.files import SURVEYS, WORKS
    from rag.dedup.schemas import Survey, Work
    from rag.eval import silver as silver_mod

    config = get_config(ctx)
    cfg = config.eval.silver
    works_path = config.paths.works / WORKS
    if not works_path.is_file():
        typer.echo(
            "No works: run `rag dedup run` first (silver samples canonical records).", err=True
        )
        raise typer.Exit(1)
    canonical = {w.canonical for w in read_jsonl(works_path, Work)}
    surveys = {s.record_id for s in read_jsonl(config.paths.works / SURVEYS, Survey)}
    texts = _chunk_texts(config)
    chunks = [ic.chunk for ic in texts]
    sampled = silver_mod.sample_chunks(chunks, canonical, surveys, cfg)
    llm = make_llm(config.chat)
    prompt = load_prompt(config.paths.prompts, cfg.prompt)
    questions, reasons = silver_mod.generate(sampled, llm, prompt, texts, cfg)
    path = config.paths.eval / cfg.file
    silver_mod.write(path, questions, reasons, len(sampled), llm, prompt, cfg.seed)
    typer.echo(f"{len(questions)} of {len(sampled)} kept -> {path}; rejected: {reasons}")
