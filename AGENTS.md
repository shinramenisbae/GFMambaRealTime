# Repository Guidelines

## Project Structure & Module Organization

- `core/`: data loader (`dataset.py` expects aligned_50.pkl), losses, metrics, optimizer, scheduler, utils.
- `models/`: `GFMamba` and building blocks (`Intramodel`, `graph_fusion`, `mamba/*`).
- `configs/`: YAML configs (default: `configs/mosi_train.yaml`).
- `ckpt/<dataset>/`: saved checkpoints and best models.
- Entrypoints: `train.py` (training/validation/test), `intra_infer.py` (intra‑modal enhancer demo).

## Build, Test, and Development Commands

- Create env (conda recommended):
  ```bash
  conda env create -f environment.yml && conda activate gfmamba
  # Or legacy: conda env create -f require.yml && conda activate misa-code
  ```
- Train (edit config first):
  ```bash
  python train.py --config_file configs/mosi_train.yaml --seed 42
  ```
- Quick inference example:
  ```bash
  python intra_infer.py
  ```
- Tests (if added under `tests/`): `pytest -q`.

## Coding Style & Naming Conventions

- Python 3.10+ preferred; 4‑space indentation; PEP8.
- Files/functions: `snake_case`; classes: `PascalCase`.
- Keep modules small and focused; place training utilities in `core/`, model logic in `models/`.
- Format/lint: `autopep8 -i -a -r core models` and `pylint core models`.

## Testing Guidelines

- Framework: `pytest`. Name files `tests/test_*.py` and functions `test_*`.
- Include at least a smoke test: construct `GFMamba` with a tiny config and check forward shapes on dummy tensors.
- When modifying loss/metrics, add tests that validate MAE/accuracy calculations.

## Commit & Pull Request Guidelines

- Commit style: short imperative subject (≤72 chars). Recommended prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.
- PRs must include: brief description, what changed in `configs/*`, reproducible command(s), and before/after metrics (or sample outputs) when applicable. Link related issues.

## Security & Configuration Tips

- Update `configs/mosi_train.yaml: dataset.dataPath` to your local `.pkl` (expects keys `train/valid/test → text, vision, audio, id, regression_labels`). Avoid committing datasets or large checkpoints; prefer Git LFS or ignore.
- Device selection is configurable. Examples: `--device cuda`, `--device cpu`, or `--gpu 0` to set `CUDA_VISIBLE_DEVICES`.
