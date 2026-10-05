# Changelog

All notable changes. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [1.0.0] - unreleased

First public release: the pre-registered evaluation `f9-v1` and everything needed to
replay it. Not tagged yet; `release.yml` publishes to PyPI and GHCR when `v1.0.0` is pushed.

### Added
- README: Gazebo episode GIF/MP4, results-page replay GIF and screenshot (`docs/media/`), with
  `scripts/media/record_episode.py` and `scripts/media/capture_site.py` to regenerate them.
- F10: `data/f9-v1.tar.gz` (core dataset: 400 episode records, programs, sandbox traces, LLM
  response cache; CC BY 4.0) and `scripts/export_dataset.py` (deterministic bundle builder,
  `--full` adds the 100 Hz joint samples).
- F10: `scripts/demo.py` and `docker-compose.yml`: replay any F9 episode without GPU, API key
  or ROS.
- F10: static results page (`scripts/build_site.py` → `site/index.html`) with success
  tables, energy, hypotheses, success-vs-Wh plot and 3D replays; GitHub Pages workflow.
- F10: `docs/benchmark.md`, `docs/report.md`, `SECURITY.md`, `THIRD_PARTY_LICENSES.md`,
  `examples/llm.ollama.yaml`; release workflow (PyPI Trusted Publishing, GHCR); coverage
  gate (≥ 80 %) and link check in CI.
- F9: pre-registered protocol (`docs/protocol.md`), `armbench protocol`, `armbench analyze`,
  results `reports/f9_eval.{json,md}`: H1 supported, H2 and H3 not supported, scenario null.
- F0–F8: simulation (UR5e, gripper, RGB-D camera in Gazebo Harmonic), energy model A/B,
  perception, typed primitives, tasks and baseline A, LLM agents B/B+S/C/C+S, sandbox,
  cache/replay, frozen skill library, energy reference. See `docs/PROJECT_STATE.md`.
