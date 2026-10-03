# ADR-004: Joint torque source for the energy model (decision D3) and η without copper (D4)

- Status: accepted
- Date: 2026-10-03
- Phase: F2

## Context

The energy model (`src/armbench/energy`) needs the torque τᵢ and the velocity ωᵢ of each arm
joint at every sample. Decision D3 of `docs/plan.es.md` left two candidates:

1. **Gazebo effort** — the `effort` field that `joint_state_broadcaster` publishes on
   `/joint_states` from `gz_ros2_control` (the force/torque the physics engine applies at
   the joint, which for position-commanded joints is the PID output).
2. **Inverse dynamics** — τ = RNEA(q, q̇, q̈) computed from the URDF with Pinocchio, with q̇
   and q̈ from filtered differentiation of the recorded q.

The plan required measuring, not assuming, that (1) is physically meaningful before any
Wh figure is reported. `scripts/torque_source.py` runs the experiment end to end
(container with `--network none`, `torque_probe`, `torque_compare`, host-side analysis) and
writes `reports/f2_torque_source.json`.

Decision D4 (η excludes copper losses) is settled in the same phase because the model
equation depends on it:

    P_el = P_mech / η + Σᵢ Rᵢ (τᵢ / k_t,i)² + P₀

If η already contained the winding losses they would be counted twice; η is therefore the
efficiency of gearbox + drive electronics only (0.70 nominal, sensitivity 0.60 / 0.80) and
copper is modelled explicitly from the per-joint k_t and R in `configs/energy.yaml`.

## Experiment (seed 0, `reports/f2_torque_source.json`)

- 5 static holds (home, stretched, half, pregrasp, elbow_up), 2 s of sim time each.
- 20 scripted multi-waypoint trajectories (3–5 waypoints, 2.4–3.8 s), waypoints checked
  against the UR5e kinematics (table, camera post, reach) before execution.
- The first trajectory repeated 10 times for repeatability.
- Recording windows are delimited in **simulation time** (goal time + 1 s settle) so that
  the P₀·T term is identical across repeats; `/joint_states` arrives at ≈ 500 Hz.
- CPU of the `energy_meter` process and of the whole container from cgroup v2 `cpu.stat`.

## Results

| Check | Threshold | Measured |
|---|---|---|
| Static: Gazebo effort vs RNEA gravity torque, joints with τ ≥ 1 N·m | rel. diff < 10 % | **1.6 % max** (others: abs. diff < 0.5 N·m) |
| Motion: NRMSE of effort vs inverse dynamics (per joint, span-normalised) | median < 0.15 | **≈ 0.06** |
| Repeatability of Wh over 10 repeats (variant A, Gazebo effort) | CV < 2 % | **1.1 %** (inverse dynamics 0.07 %) |
| Energy Gazebo vs inverse dynamics, 20 trajectories | informative | Gazebo 0–7 % higher (PID effort includes friction-like terms RNEA ignores) |
| Meter CPU (100 Hz stream) | < 5 % of container CPU time | **2.3 %** (5.2 % of one core; container ≈ 200 % of a core) |

During motion R² is low on joints whose torque span is small (wrist 3 carries ≈ 0 N·m) and
on the fast joints because Gazebo's effort is a 500 Hz PID output with high-frequency
content that the Savitzky–Golay q̈ estimate cannot reproduce; the NRMSE and the integrated
energies show the two sources agree where it matters.

### Meter cost

A Python (`rclpy`) subscriber costs ≈ 10 % of one core at 500 Hz *before doing anything*
(measured in the image: no-op callback 12.7 %, raw-bytes callback 10.5 %, full meter
13.5 %), and the headless simulation uses ≈ 220 % of a core, so a 500 Hz Python meter is
≈ 6–9 % of the container and cannot meet the 5 % budget whatever it computes. The
decision is to feed the meter from a second `joint_state_broadcaster`
(`energy_state_broadcaster`, `update_rate: 100`, `use_local_topics: true` →
`/energy_state_broadcaster/joint_states`) and keep the 500 Hz `/joint_states` untouched for
control and for the probe. Decimating the 500 Hz logs to 100 Hz offline changes the
integrated Wh (variant A) by ≤ 2.6 % over the 20 trajectories
(`decimation_100hz_rel_err_max`), of the same order as the run-to-run CV, so the online
figure is reported as the 100 Hz value and offline recomputation from a 500 Hz log is the
reference when both exist.

## Decision

1. **Torque source = Gazebo effort** (`torque_source: gazebo_effort` in the report). It
   matches gravity torque within 1.6 % at rest, tracks inverse dynamics during motion
   (NRMSE 0.06), is repeatable (CV 1.1 %) and is available online at no cost. Inverse
   dynamics stays implemented (`torque_compare`) as the cross-check and as the fallback if
   a future Gazebo/`gz_ros2_control` upgrade breaks the effort field (re-run
   `scripts/torque_source.py`; the gate must pass before results are trusted).
2. **η excludes copper losses** (D4); k_t and R per joint are explicit assumptions in
   `configs/energy.yaml` and every report must state them. Absolute Wh are therefore
   model-dependent; comparisons between methods on the same episodes are the claim.
3. The online meter integrates the 100 Hz `energy_state_broadcaster` stream; offline
   recomputation (`armbench energy <log.jsonl>`) uses whatever rate the log has.

## Consequences

- F4–F9 read τ from the joint states (`effort`) and never call Pinocchio online.
- `reports/f2_torque_source.json` is the evidence for critical gate 2; the thresholds are
  in `scripts/torque_source.py::THRESHOLDS` and must not be changed without a new ADR.
- Open: the gripper finger effort is recorded but not in the arm energy; add it when F4
  defines `grasp()` (one extra pair of joints in `configs/energy.yaml`).
