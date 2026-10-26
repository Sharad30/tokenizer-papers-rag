"""`rag chat`: grounded multi-turn chat over the corpus, with citations."""

from typing import Annotated

import typer

from rag.common.cli import get_config

EXCERPT_CHARS = 300  # of each cited chunk, with --show-sources
COMMANDS_HELP = "Commands: /reset (new conversation), /quit. Ctrl-D also quits."


def chat(
    ctx: typer.Context,
    provider: Annotated[
        str | None, typer.Option(help="ollama | anthropic (default: chat.provider)")
    ] = None,
    mode: Annotated[
        str | None, typer.Option(help="bm25 | dense | hybrid (default: chat.retrieval.mode)")
    ] = None,
    show_sources: Annotated[
        bool, typer.Option("--show-sources", help="Print an excerpt of each cited chunk")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Show the rewrite, retrieval table, gates, timings")
    ] = False,
    once: Annotated[
        str | None, typer.Option(help="Answer one question and exit (no conversation)")
    ] = None,
) -> None:
    """Interactive grounded chat over the corpus, with citations."""
    from rag.chat.engine import ChatEngine, format_sources
    from rag.chat.llm import LLMError, make_llm
    from rag.chat.prompts import load_prompt
    from rag.chat.transcript import Transcript
    from rag.collect.manifest import read_manifest
    from rag.common.provenance import config_hash, git_commit
    from rag.retrieve.retriever import Retriever

    config = get_config(ctx)
    if provider not in (None, "ollama", "anthropic"):
        raise typer.BadParameter(f"unknown provider {provider!r}", param_hint="--provider")
    if mode not in (None, "bm25", "dense", "hybrid"):
        raise typer.BadParameter(f"unknown mode {mode!r}", param_hint="--mode")
    chat_config = config.chat
    if mode:
        chat_config = chat_config.model_copy(
            update={"retrieval": chat_config.retrieval.model_copy(update={"mode": mode})}
        )
    try:
        llm = make_llm(chat_config, provider)  # type: ignore[arg-type]
    except LLMError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    prompts = [
        load_prompt(config.paths.prompts, chat_config.prompts.answer),
        load_prompt(config.paths.prompts, chat_config.prompts.rewrite),
    ]
    # TODO(sharad): Phase 7 "You do": add a --year-min option and pass
    # allow=lambda chunk: <the chunk's paper year> >= year_min. The year is
    # on the manifest record (records[chunk.record_id].year), not the chunk.
    allow = None
    records = {r.record_id: r for r in read_manifest(config.paths.manifest)}
    typer.echo("Loading indexes and models…", err=True)
    engine = ChatEngine(
        Retriever.load(config, {"hybrid"}), llm, records, chat_config, *prompts, allow=allow
    )
    transcript = Transcript(config.paths.chat)
    transcript.write_header(
        git_commit(), config_hash(config), llm.model, prompts, chat_config.model_dump(mode="json")
    )

    def ask(text: str) -> None:
        try:
            result = engine.turn(text)
        except LLMError as exc:
            typer.echo(f"LLM error: {exc}", err=True)
            return
        transcript.write_turn(result)
        if debug:
            typer.echo(_debug_view(result))
        typer.echo(f"\n{result.reply}")
        sources = format_sources(result)
        if sources:
            typer.echo("\nSources:")
            for line, (_, cc) in zip(sources, result.sources, strict=True):
                typer.echo(f"  {line}")
                if show_sources:
                    excerpt = cc.hit.chunk.text.replace("\n", " ")[:EXCERPT_CHARS]
                    typer.echo(f"      {excerpt}…")
        typer.echo("")

    if once:
        ask(once)
        return
    typer.echo(f"Chat with {llm.model} over the corpus. {COMMANDS_HELP}")
    typer.echo(f"Transcript: {transcript.path}\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            engine.reset()
            transcript.write_reset()
            typer.echo("New conversation.\n")
            continue
        ask(text)


def _debug_view(result) -> str:
    from rag.retrieve.show import render_hits

    rw = result.rewrite
    if rw.rewritten:
        rewrite_line = f"rewrite: {rw.query!r}"
    elif rw.fallback:
        rewrite_line = f"rewrite FAILED, using the raw message ({rw.error or 'empty reply'})"
    else:
        rewrite_line = "rewrite: not needed (first turn)"
    top = result.retrieval.top_scores
    lines = [
        "",
        f"── turn {result.turn} ── {rewrite_line}",
        "best scores in corpus: "
        + ", ".join(f"{name} {score:.3f}" for name, score in sorted(top.items())),
        f"abstained: {result.abstained or 'no'}",
        render_hits(result.rewrite.query, "context", [cc.hit for cc in result.context]),
    ]
    if result.completion:
        c = result.completion
        lines.append(f"model {c.model}: {c.input_tokens} in / {c.output_tokens} out tokens")
        lines.append(f"raw answer: {c.text!r}")
    if result.cited:
        lines.append(
            f"invalid citations: {result.cited.invalid or 'none'}; "
            f"uncited sentences: {result.cited.uncited_sentences}"
        )
    lines.append("seconds: " + ", ".join(f"{k} {v:.2f}" for k, v in result.seconds.items()))
    return "\n".join(lines)
