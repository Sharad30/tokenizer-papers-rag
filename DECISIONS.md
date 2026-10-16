# Decisions

One entry per significant choice. Claude Code drafts entries as `proposed`;
Sharad reviews and marks them `accepted` (or rewrites them). Entries are never
deleted: a changed decision gets a new entry, and the old one is marked
`superseded by D-NNN`.

## Template

```
## D-NNN: <title>
- Date / Phase:
- Status: proposed | accepted (Sharad, YYYY-MM-DD) | superseded by D-NNN
- Context: what forced a choice
- Options considered:
  1. <option>: pros / cons
  2. <option>: pros / cons
  3. <option>: pros / cons
- Choice:
- Why:
- Evidence: results file / commit / probe, or "none yet: judgement call"
- Tuned on eval set?: yes / no
- Revisit if:
```

---

## D-001: Package and CLI layout
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Every phase needs a place for its code and a command to run it; PLAN.md asks for one Typer CLI with a subcommand per stage.
- Options considered:
  1. One package `rag` (src layout); each stage's command lives in `rag/<stage>/cli.py`, and `rag/cli.py` only registers them. Pros: stage code and its CLI are side by side; the root file stays a readable table of contents. Cons: one extra file per stage.
  2. All commands in one `cli.py`. Pros: one file to read. Cons: grows with every phase and mixes all stages' imports.
  3. Distribution-named package `tokenizer_papers_rag`. Pros: uv's default. Cons: long imports, and the name differs from the `rag` command.
- Choice: 1. Distribution name `tokenizer-papers-rag`, import name and command `rag` (via `tool.uv.build-backend.module-name`).
- Why: Matches PLAN.md's stage list one to one; each phase touches mostly its own folder.
- Sub-choice: the config is loaded **lazily** (the root callback stores the path; `AppState.config()` loads it on first use). Click runs the root callback even for `rag <cmd> --help`, so loading eagerly would break help outside the repo root. A bad config still fails at the start of any real command. Alternatives were eager loading in the callback, or each command calling `load_config` itself (repeats the `--config` handling in every command).
- Evidence: none yet: judgement call. `tests/test_cli.py::test_help_works_for_every_command` runs help from an empty directory.
- Tuned on eval set?: no
- Revisit if: a stage needs more than one command (then give it a sub-Typer like `inspect`).

## D-002: Config format and strict validation
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Every tunable value must live in `config.yaml`, and every results file must record the config that produced it.
- Options considered:
  1. YAML + pydantic models with `extra="forbid"`; sections added per phase when decided. Pros: typos fail loudly; no invented values. Cons: config.yaml grows phase by phase, so the final shape isn't visible yet.
  2. Full skeleton for all phases now. Pros: the whole shape is visible. Cons: full of placeholder numbers chosen before any evidence, which defeats "decisions backed by measurements".
  3. pydantic-settings with env-var overrides. Pros: easy overrides. Cons: a second source of truth that results files would also need to record.
- Choice: 1.
- Why: A silently ignored key (`chunk_szie: 300`) would make a run use the default while the config claims otherwise, which is the worst kind of eval bug.
- Related: `config_hash` hashes everything except `paths`. Resolved paths are absolute and differ between clones, but where files live doesn't change what a run computes.
- Evidence: `tests/test_config.py`, `tests/test_provenance.py`.
- Tuned on eval set?: no
- Revisit if: we need per-run overrides for sweeps (then add a `--set key=value` that is recorded in the results file).

## D-003: Stub commands exit 0 and print their contract
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Commands exist before their stages are built.
- Options considered:
  1. Print `[stub] rag <cmd> (Phase N): reads -> writes, limit=N` and exit 0. Pros: `make smoke` chains the full pipeline now and shows the stage contracts. Cons: a stub "succeeds" without doing anything.
  2. Exit non-zero "not implemented". Pros: can't be mistaken for success. Cons: smoke can't run past the first stub.
- Choice: 1. The `[stub]` prefix makes it obvious in output.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: never needed after Phase 7, when every stub is gone.

## D-004: Tooling: ruff now, type checker later
- Date / Phase: 2026-10-16 / Phase 0
- Status: proposed
- Context: Type hints are required everywhere; the question is whether to enforce them yet.
- Options considered:
  1. ruff (lint + format) only; add mypy when real logic lands. Pros: little friction while there is almost no logic. Cons: hints are unchecked for now.
  2. ruff + mypy from day one. Pros: hints checked from the start. Cons: setup cost and stub-package noise before there's logic worth checking.
  3. No tooling. Cons: style drift, and the commit history gets noisy.
- Choice: 1. Python pinned to 3.11 (`.python-version`), the declared floor, so 3.12-only features can't slip in.
- Evidence: none yet: judgement call.
- Tuned on eval set?: no
- Revisit if: Phase 1 adds API clients and schemas. That's a good point to add `mypy --strict` on `src/`.
