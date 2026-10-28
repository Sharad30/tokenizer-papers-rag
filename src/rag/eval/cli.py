"""`rag eval ...`: build and check the eval set (Phase 3), score runs (Phase 4)."""

from pathlib import Path
from typing import Annotated

import typer

from rag.collect.manifest import read_manifest
from rag.common.cli import LimitOption, get_config
from rag.common.config import Config
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
