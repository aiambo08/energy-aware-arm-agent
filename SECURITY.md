# Security policy

## Scope

armbench executes Python programs written by a language model. The sandbox
(`src/armbench/sandbox`, ADR-007) is the security-relevant part of the repository:

1. **AST whitelist**: imports, attribute names starting with `_`, `gi_*`/`f_*` frame
   attributes, `exec`/`eval`/`open` and every name outside the published primitive API are
   rejected before anything runs (`max_source_chars`, `max_nodes` cap the input).
2. **Restricted builtins**: only a fixed list (`len`, `range`, `min`, `max`, `sorted`, `next`, …).
3. **Child process with rlimits**: CPU seconds, address space, wall clock, number of primitive
   calls and stdout size are capped (`configs/llm.yaml` → `sandbox`). The program talks to the
   robot only through a typed JSON-RPC channel.
4. **Container**: in simulation the worker runs inside the simulation container started with
   `--network none`.

`unshare -rn` (user + network namespace) is attempted and recorded in each episode's
`sandbox_isolation`; it is refused inside Docker on hosts without unprivileged user
namespaces, so there the container is the network boundary. The attack battery
(`tests/unit/test_sandbox.py`, 86 cases) runs in CI.

The sandbox is designed for **benchmark programs from a fixed set of LLMs**, not for
hostile code from the internet. Do not expose `armbench run` to untrusted users.

## Secrets

API keys are read from environment variables named in the LLM profile (`api_key_env`, e.g.
`ARMBENCH_NEBIUS_API_KEY`) at call time and are never written to logs, caches or `run.json`.
The cache key contains provider, model and endpoint, not the key. `detect-private-key` runs
in pre-commit.

## Reporting a vulnerability

Please open a private report through GitHub: *Security → Report a vulnerability* on
<https://github.com/aiambo08/energy-aware-arm-agent/security/advisories/new>, or e-mail
aibo.ni@alumnos.upm.es. Include the program or input that escapes the sandbox and the
`sandbox_isolation` value printed by `armbench run`. Expect an answer within 7 days.

Only the latest release on `main` is supported.
