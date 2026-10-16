# Phase 0 teaching note: Scaffold

Phase 0 has no RAG concepts in it. What it does have is the machinery every later
phase relies on: how code is packaged and run, how config reaches the code, how
outputs are written safely, and how a number gets traced back to the code and
config that produced it. Each section below gives the concept, where it lives,
and the pitfall it guards against.

---

## 1. A uv project with a src layout

**Where:** `pyproject.toml`, `uv.lock`, `.python-version`, `src/rag/`.

- `pyproject.toml` declares *what* the project needs (`typer`, `pydantic`, `pyyaml`;
  dev group `pytest`, `ruff`) with lower bounds. `uv.lock` records *exactly which*
  versions were resolved, for every package, on every platform. `uv sync` installs
  from the lock, so a fresh clone gets the same environment. The lock is committed
  because this is an application, not a library.
- `[dependency-groups] dev` holds tools needed only for development. `uv sync`
  and `uv run` include the `dev` group by default.
- **src layout:** the package is in `src/rag/`, not `./rag/`. Python can't
  import it by accident from the repo root. It has to be *installed* into `.venv`
  (uv does this), so tests exercise the same import path as the real command.
- **Entry point:** `[project.scripts] rag = "rag.cli:main"` is what makes `rag`
  a command. `uv run rag ...` runs it inside the project environment.
- **Name vs import name:** the distribution is `tokenizer-papers-rag`, but
  `[tool.uv.build-backend] module-name = "rag"` tells the `uv_build` backend that
  the module is `src/rag/`.
- **uv version note:** local uv is 0.11.7. From uv 0.12 the docs say `uv init`
  adds a build system to apps by default. On 0.11 it needed `uv init --package`.

**Pitfall:** running `python -m pytest` or a bare `python` outside `uv run`
uses whatever interpreter is on PATH (here, Anaconda), not `.venv`. Always use
`uv run` or the Makefile.

## 2. The CLI: Typer callback, context, and lazy config

**Where:** `src/rag/cli.py` (wiring), `src/rag/common/cli.py` (`AppState`,
`get_config`, `LimitOption`, `print_stub`), `src/rag/<stage>/cli.py` (each command).

- Typer builds a Click command tree from Python functions. Parameters typed with
  `Annotated[int | None, typer.Option(min=1)]` become options. Typer reads the type
  for parsing and validation, so `--limit 0` is rejected before your code runs (exit code 2).
- `@app.callback()` on the root (`cli.py:root`) runs before any subcommand. It
  handles the *global* options `--config` and `--log-level`, so those go
  **before** the subcommand: `rag --config other.yaml parse`, not
  `rag parse --config ...`.
- The callback puts an `AppState` on `ctx.obj`. Each command receives
  `ctx: typer.Context` and calls `get_config(ctx)`, which finds the `AppState`
  with `ctx.find_object(AppState)`. This is Click's way of passing shared state
  down the tree without globals.
- **Why lazy:** Click runs the root callback even for `rag parse --help`. If the
  callback loaded `config.yaml`, help would fail in any directory without one.
  So the callback only stores the path, and `AppState.config()` loads and caches
  the config on first use. It also calls `setup_logging`, so the log level comes
  from the config unless `--log-level` overrides it.
- `app.add_typer(inspection.app, name="inspect")` attaches a sub-app as a
  command group, so you get `rag inspect paper|work|query`.
- Stage commands are plain functions registered in `cli.py`
  (`app.command("eval")(run_eval)`). The function is named `run_eval` so it
  doesn't shadow the builtin `eval`. Only the CLI name is `eval`.
- **Stubs** (`print_stub`) print `[stub] rag parse (Phase 2): <reads> -> <writes>, limit=N`
  and exit 0. That line *is* the stage contract from PLAN.md, now with real resolved
  paths. Each phase replaces its stub's body. See D-003 for why they exit 0.

**Pitfall:** Typer 0.27 no longer depends on the `click` package. It ships
a private copy at `typer._click`, so `import click` fails in this environment.
Use Typer's public names (`typer.core.TyperGroup`, `typer.testing.CliRunner`).

## 3. Typed, strict configuration

**Where:** `src/rag/common/config.py` (`Config`, `PathsConfig`, `load_config`,
`ConfigError`), `src/rag/common/schemas.py` (`StrictModel`), `config.yaml`.

- `load_config` does four things, each with its own readable error: the file
  exists; it parses as YAML (`yaml.safe_load`, which never builds arbitrary Python
  objects); the top level is a mapping; and pydantic validates it against `Config`.
  The CLI turns `ConfigError` into one `Error: ...` line on stderr and exit code 1.
- `StrictModel` sets `extra="forbid"`. Without it, pydantic *ignores* unknown keys,
  so `chunk_szie: 300` would load without complaint and the run would quietly use
  the default. The config file would then disagree with what actually ran.
- Types do the validating: `level: Literal["DEBUG", ...]` rejects `LOUD`, and
  `max_date: date` takes YAML's `2026-10-15` as a `datetime.date`.
- **Path resolution:** `PathsConfig.resolved_against(base)` makes every path
  absolute *relative to the config file's directory*, not the shell's current
  directory, so `rag` writes to the same place whichever directory you run it from.
- Only Phase 0's sections exist. Each later phase adds a section model (e.g.
  `ChunkingConfig`) and a field on `Config` when it makes its decisions (D-002).

**Your task:** `TODO(sharad)` in `config.py` (add a field) and in
`src/rag/collect/cli.py` (use it). To do it you need a new `StrictModel`
subclass or a field on an existing one, a matching key in `config.yaml`, and
`config.<section>.<field>` in the command. Then check whether
`tests/fixtures/config_minimal.yaml` needs the field too, and think about why.

## 4. Safe outputs and stats.json

**Where:** `src/rag/common/io.py`, `StageStats`/`StageFailure` in `schemas.py`.

- **Atomic writes** (`_write_text_atomic`): write to a temp file in the *same
  directory*, then `os.replace` it over the target. A rename within one filesystem
  is atomic, so readers see the old file or the new one, never half of one. If a
  stage is killed mid-write, the next stage won't read a truncated JSONL as if it
  were complete.
- `read_jsonl(path, Model)` validates every line, so a schema mismatch between
  stages fails at the boundary, not three stages later.
- `StageStats` is what every stage's `stats.json` will contain: counts, a list of
  failures (each with an id and reason, so failures are counted rather than lost),
  the `--limit` used, and provenance.

## 5. Provenance: which code and config produced this?

**Where:** `src/rag/common/provenance.py`.

- `git_commit()` returns HEAD plus `-dirty` if the working tree has uncommitted
  changes. A result produced by uncommitted code can't be reproduced from the hash
  alone, so the marker matters. Outside a git repo it returns `"unknown"` rather
  than crashing.
- `config_hash(config)` is the sha256 of the config serialised with sorted keys,
  so reordering YAML sections doesn't change it. It **excludes `paths`**: they're
  absolute and differ between clones, but where files live doesn't change what a
  run computes. Everything else changes the hash.

## 6. Tests that grow with the project

**Where:** `tests/`.

- `test_cli.py::_command_paths` walks the *real* command tree
  (`typer.main.get_command(app)`), and `test_help_works_for_every_command` is
  parametrized over every command it finds. A command added in Phase 4 gets a
  `--help` test with no edit to the test file. It runs from an empty temp directory
  (`monkeypatch.chdir`) to prove help doesn't need a config.
- `CliRunner().invoke(app, [...])` runs the CLI in-process. `result.stdout` and
  `result.stderr` are separate, which is how `test_missing_config_fails_with_clear_message`
  checks that errors go to stderr.
- `test_provenance.py` builds a throwaway git repo in `tmp_path` and passes the git
  identity inline (`-c user.name=...`), so it doesn't depend on your machine's git config.
  The outside-repo test sets `GIT_CEILING_DIRECTORIES` so git can't find a parent repo.

## 7. Makefile

`make check` (lint + format check + tests) is the gate before every commit.
`make smoke` runs every stage in pipeline order on `--limit 2`. It's trivial
now, and by Phase 7 it becomes the five-minute end-to-end run. `make clean`
removes caches only. It never touches `data/`, `results/` or `eval/`, because
those are expensive or hand-made.

---

## What to look at

- `uv run rag --help`, then `uv run rag inspect --help`: the shape of the whole project.
- `uv run rag config show`: the config *after* validation and path resolution.
  Compare it with `config.yaml`: relative paths are now absolute, and the date is normalised.
- `make smoke`: each line is one stage's contract.
- Try to break it: add `colour: true` under `logging:` in `config.yaml` and run
  `uv run rag parse`. Then try `--limit 0`, then `--config nope.yaml`.

## Common pitfalls (summary)

| Pitfall | Guard here |
|---|---|
| Typo in config silently ignored | `StrictModel` `extra="forbid"` |
| Output paths depend on where you ran the command | `resolved_against` the config dir |
| Truncated output read as complete | atomic writes in `io.py` |
| Result can't be reproduced | `git_commit` with `-dirty`, `config_hash` |
| `--help` needs a config | lazy `AppState.config()` |
| Running outside the venv | always `uv run` / `make` |
| New command never tested | tree-walking `--help` test |
