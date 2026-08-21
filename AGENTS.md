# Agent Guidelines & Operating Principles

This document outlines the operational rules, architectural boundaries, and coding standards for agents working within this repository.

---

## 1. Operating Mindset & Workflow

- **Plan First**: For non-trivial tasks, outline the steps and affected files before modifying code.
- **Minimal Diffs**: Make focused, surgical changes. Never reformat, reorganize, or rewrite unrelated files or functions.
- **Root-Cause Fixes**: Fix underlying logic issues; do not patch errors with quick workarounds or suppression flags.
- **Verification**: Always run builds, type checks, or test suites after modifying code to verify changes.

---

## 2. Layer Boundaries & Responsibilities

- **`src/data/`**: Data ingestion, source connectors, and dataset loading.
- **`src/features/`**: Feature transformations, indicators, and preprocessing pipelines.
- **`src/models/`**: Model architecture definitions and forward logic.
- **`src/training/`**: Loss functions, optimization loops, and evaluation metrics.
- **`src/backtesting/`**: Simulation engine, execution logic, and strategy evaluation.
- **`src/utils/`**: Shared helper functions, logging, and configuration parsers.
- **`configs/`**: Declarative configuration files for experiments and pipelines.
- **`tests/`**: Unit and integration tests for repository components.

---

## 3. Engineering & Code Standards

- **Strict Typing**: Enforce explicit type annotations on all function signatures.
- **Clean Code**: Write self-documenting code with descriptive names. Comment only complex or non-obvious logic.
- **Testing**: Ensure all new modules have corresponding unit tests in `tests/`.

---

## 4. Documentation Protocol

- Update `docs/ARCHITECTURE.md` when introducing new architectural components or mathematical formulations.
- Log new benchmark runs and quantitative results in `docs/EXPERIMENTS.md`.
- Document non-obvious domain nuances, data caveats, or design discoveries in `docs/LEARNINGS.md`.
