from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from armbench.kinematics import (
    DEFAULT_KINEMATICS_FILE,
    HOME_Q,
    JOINT_NAMES,
    N_JOINTS,
    IKResult,
    UR5eModel,
    is_rotation_matrix,
    load_ur5e_kinematics,
    rotation_angle,
    rotation_vector,
    rpy_to_matrix,
    top_down_pose,
)
from armbench.kinematics.ur5e import ORIGIN_NAMES

MODEL = UR5eModel()
LOWER = MODEL.lower_limits
UPPER = MODEL.upper_limits

joint_vectors = hnp.arrays(
    np.float64,
    (N_JOINTS,),
    elements=st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False),
)
angles = st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False)


def rotvec_fd(model: UR5eModel, q: np.ndarray, h: float = 1e-6) -> np.ndarray:
    jac = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = h
        plus, minus = model.fk(q + dq), model.fk(q - dq)
        jac[:3, i] = (plus[:3, 3] - minus[:3, 3]) / (2 * h)
        jac[3:, i] = rotation_vector(plus[:3, :3] @ minus[:3, :3].T) / (2 * h)
    return jac


# --- config -------------------------------------------------------------------------------


def test_default_yaml_matches_ur_description() -> None:
    params = load_ur5e_kinematics(DEFAULT_KINEMATICS_FILE)
    assert tuple(params.kinematics) == ORIGIN_NAMES
    assert tuple(params.joint_limits) == JOINT_NAMES
    assert params.kinematics["shoulder"].z == pytest.approx(0.1625)
    assert params.kinematics["forearm"].x == pytest.approx(-0.425)
    assert params.kinematics["wrist_1"].x == pytest.approx(-0.3922)
    assert params.kinematics["wrist_1"].z == pytest.approx(0.1333)
    assert params.kinematics["wrist_2"].y == pytest.approx(-0.0997)
    assert params.kinematics["wrist_3"].y == pytest.approx(0.0996)
    assert params.joint_limits["elbow_joint"].max_position == pytest.approx(np.pi)
    for name in JOINT_NAMES:
        lim = params.joint_limits[name]
        assert lim.max_velocity == pytest.approx(np.pi)
        assert lim.min_position == -lim.max_position
        if name != "elbow_joint":
            assert lim.max_position == pytest.approx(2 * np.pi)
    assert params.joint_limits["shoulder_pan_joint"].max_effort == 150.0
    assert params.joint_limits["wrist_3_joint"].max_effort == 28.0


def test_unknown_key_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_KINEMATICS_FILE.read_text())
    raw["kinematics"]["shoulder"]["extra"] = 1.0
    p = tmp_path / "k.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="extra"):
        load_ur5e_kinematics(p)


def test_missing_joint_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_KINEMATICS_FILE.read_text())
    del raw["joint_limits"]["wrist_3_joint"]
    p = tmp_path / "k.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="joint_limits keys"):
        load_ur5e_kinematics(p)


# --- SE(3) helpers ------------------------------------------------------------------------


def test_rpy_convention_is_zyx() -> None:
    rot = rpy_to_matrix(0.0, 0.0, np.pi / 2)
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    rot = rpy_to_matrix(np.pi / 2, 0.0, 0.0)
    assert np.allclose(rot @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])


@given(angles, angles, angles)
def test_rotation_vector_round_trip(roll: float, pitch: float, yaw: float) -> None:
    rot = rpy_to_matrix(roll, pitch, yaw)
    vec = rotation_vector(rot)
    angle = float(np.linalg.norm(vec))
    assert angle == pytest.approx(rotation_angle(rot), abs=1e-9)
    assert 0.0 <= angle <= np.pi + 1e-12
    if angle > 1e-9:
        axis = vec / angle
        assert np.allclose(rot @ axis, axis, atol=1e-7)


def test_rotation_vector_near_pi_is_finite() -> None:
    vec = rotation_vector(rpy_to_matrix(np.pi, 0.0, 0.0))
    assert np.all(np.isfinite(vec))
    assert np.linalg.norm(vec) == pytest.approx(np.pi)
    assert abs(vec[0]) == pytest.approx(np.pi)


# --- forward kinematics -------------------------------------------------------------------


def test_fk_zero_configuration_reference() -> None:
    pose = MODEL.fk(np.zeros(N_JOINTS))
    assert np.allclose(pose[:3, 3], [0.8172, 0.2329, 0.0628], atol=1e-4)
    assert pose[0, 3] == pytest.approx(0.425 + 0.3922, abs=1e-4)
    assert pose[1, 3] == pytest.approx(0.1333 + 0.0996, abs=1e-4)
    assert pose[2, 3] == pytest.approx(0.1625 - 0.0997, abs=1e-4)
    expected_rot = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(pose[:3, :3], expected_rot, atol=1e-6)


def test_fk_home_configuration() -> None:
    pose = MODEL.fk(HOME_Q)
    # Arm straight up: z = 0.1625 + 0.425 + 0.3922 + 0.0997 = 1.0794 m (tool0, not the TCP).
    assert np.allclose(pose[:3, 3], [0.0008, 0.2329, 1.0794], atol=2e-3)
    assert np.allclose(pose[:3, 2], [0.0, 1.0, 0.0], atol=2e-3)


def test_fk_accepts_sequences_and_rejects_wrong_length() -> None:
    assert np.allclose(MODEL.fk(list(HOME_Q)), MODEL.fk(np.array(HOME_Q)))
    with pytest.raises(ValueError, match="6 joint values"):
        MODEL.fk([0.0, 0.0, 0.0])


def test_fk_all_frames() -> None:
    frames = MODEL.fk_all(HOME_Q)
    assert len(frames) == N_JOINTS + 1
    assert np.allclose(frames[-1], MODEL.fk(HOME_Q))
    assert np.allclose(frames[0][:3, 3], [0.0, 0.0, 0.1625])
    for frame in frames:
        assert is_rotation_matrix(frame[:3, :3])
        assert np.allclose(frame[3], [0.0, 0.0, 0.0, 1.0])


def test_fk_matches_sim_check_fixture() -> None:
    """Joint-space fixture from armbench_bringup/config/sim_check.yaml (TCP = tool0 + 0.125 z)."""
    fixture = {
        (0.2699, -1.6612, -1.9823, -1.0690, 1.5708, 1.8407): (-0.50, 0.00, 0.1725),
        (0.2699, -1.9135, -2.1424, -0.6565, 1.5708, 1.8407): (-0.50, 0.00, 0.0275),
        (-0.0920, -1.7334, -1.8989, -1.0802, 1.5708, 1.4788): (-0.50, 0.18, 0.1725),
        (-0.0920, -1.9387, -2.0426, -0.7311, 1.5708, 1.4788): (-0.50, 0.18, 0.0425),
    }
    for q, tcp in fixture.items():
        pose = MODEL.fk(np.array(q))
        tcp_from_fk = pose[:3, 3] + 0.125 * pose[:3, 2]
        assert np.allclose(tcp_from_fk, tcp, atol=1e-3)
        assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)


@settings(max_examples=1000, deadline=None)
@given(joint_vectors)
def test_fk_rotation_is_orthonormal(q: np.ndarray) -> None:
    rot = MODEL.fk(q)[:3, :3]
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)


@given(joint_vectors)
def test_fk_shoulder_pan_rotates_about_base_z(q: np.ndarray) -> None:
    pose = MODEL.fk(q)
    q2 = q.copy()
    q2[0] += 0.3
    pose2 = MODEL.fk(q2)
    assert np.linalg.norm(pose[:3, 3][:2]) == pytest.approx(np.linalg.norm(pose2[:3, 3][:2]))
    assert pose[2, 3] == pytest.approx(pose2[2, 3])


# --- Jacobian -----------------------------------------------------------------------------


@settings(deadline=None)
@given(joint_vectors)
def test_jacobian_matches_finite_differences(q: np.ndarray) -> None:
    jac = MODEL.jacobian(q)
    assert jac.shape == (6, N_JOINTS)
    assert np.allclose(jac, rotvec_fd(MODEL, q), atol=1e-5)


def test_jacobian_first_column_is_base_z() -> None:
    jac = MODEL.jacobian(HOME_Q)
    assert np.allclose(jac[3:, 0], [0.0, 0.0, 1.0])


# --- limits -------------------------------------------------------------------------------


def test_within_limits() -> None:
    assert MODEL.within_limits(HOME_Q)
    assert MODEL.within_limits(UPPER)
    assert not MODEL.within_limits(np.array([0.0, 0.0, np.pi + 0.1, 0.0, 0.0, 0.0]))
    assert not MODEL.within_limits(np.array([2 * np.pi + 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_wrap_to_limits() -> None:
    q = np.array([7.0, -7.0, 3.5, 0.0, 0.0, 0.0])
    wrapped = MODEL.wrap_to_limits(q)
    assert MODEL.within_limits(wrapped)
    assert wrapped[0] == pytest.approx(7.0 - 2 * np.pi)
    assert wrapped[1] == pytest.approx(-7.0 + 2 * np.pi)
    assert wrapped[2] == pytest.approx(3.5 - 2 * np.pi)
    narrow = np.array([0.0, 0.0, 3.5, 0.0, 0.0, 0.0])
    assert MODEL.within_limits(MODEL.wrap_to_limits(narrow))
    assert np.allclose(MODEL.fk(wrapped), MODEL.fk(q))
    assert np.allclose(MODEL.wrap_to_limits(HOME_Q), HOME_Q)


def test_clamp() -> None:
    q = np.array([10.0, -10.0, 4.0, 0.0, 0.0, 0.0])
    clamped = MODEL.clamp(q)
    assert np.allclose(clamped, [UPPER[0], LOWER[1], UPPER[2], 0.0, 0.0, 0.0])


# --- inverse kinematics -------------------------------------------------------------------


def test_ik_result_is_frozen_and_validated() -> None:
    res = IKResult(success=True, q=np.zeros(6), iters=0, position_error=0.0, orientation_error=0.0)
    with pytest.raises(ValueError, match="frozen"):
        res.success = False  # type: ignore[misc]
    assert not res.q.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        IKResult(success=True, q=np.zeros(5), iters=0, position_error=0.0, orientation_error=0.0)


def test_ik_from_exact_seed_converges_immediately() -> None:
    target = MODEL.fk(HOME_Q)
    res = MODEL.ik(target, HOME_Q)
    assert res.success
    assert res.iters == 0
    assert np.allclose(res.q, HOME_Q)


ELBOW_MARGIN = 0.05  # at |elbow| = pi the fold singularity coincides with the joint limit


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    q=st.tuples(
        angles,
        angles,
        st.floats(-np.pi + ELBOW_MARGIN, np.pi - ELBOW_MARGIN, allow_nan=False),
        angles,
        angles,
        angles,
    ).map(np.array),
    noise=hnp.arrays(
        np.float64,
        (N_JOINTS,),
        elements=st.floats(-0.3, 0.3, allow_nan=False, allow_infinity=False),
    ),
)
def test_ik_round_trip(q: np.ndarray, noise: np.ndarray) -> None:
    """fk(ik(fk(q), q0 = q + noise).q) == fk(q) for every q inside the limits.

    The elbow stays ``ELBOW_MARGIN`` away from +-pi: exactly there the arm is folded (singular)
    and clamped at the same time, and convergence becomes sublinear.
    """
    target = MODEL.fk(q)
    res = MODEL.ik(target, q + noise)
    assert res.success, (res.iters, res.position_error, res.orientation_error)
    assert MODEL.within_limits(res.q)
    pose = MODEL.fk(res.q)
    assert np.linalg.norm(pose[:3, 3] - target[:3, 3]) < 1e-4
    assert rotation_angle(target[:3, :3] @ pose[:3, :3].T) < 1e-3


def test_ik_unreachable_target_fails_gracefully() -> None:
    target = np.eye(4)
    target[:3, 3] = [2.0, 0.0, 0.0]
    res = MODEL.ik(target, HOME_Q)
    assert not res.success
    assert res.iters == 200
    assert res.position_error > 0.5
    assert MODEL.within_limits(res.q)
    assert np.all(np.isfinite(res.q))


def test_ik_rejects_bad_target_shape() -> None:
    with pytest.raises(ValueError, match="4x4"):
        MODEL.ik(np.eye(3), HOME_Q)


def test_top_down_pose_convention() -> None:
    pose = top_down_pose((-0.5, 0.0, 0.2), 0.0)
    assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(pose[:3, 0], [1.0, 0.0, 0.0])
    assert np.allclose(pose[:3, 3], [-0.5, 0.0, 0.2])
    yawed = top_down_pose((-0.5, 0.0, 0.2), np.pi / 2)
    assert np.allclose(yawed[:3, 2], [0.0, 0.0, -1.0])
    assert np.allclose(yawed[:3, 0], [0.0, 1.0, 0.0])


def test_ik_top_down_tool0_points_down() -> None:
    res = MODEL.ik_top_down((-0.50, 0.00, 0.2975), 0.0, HOME_Q)
    assert res.success
    pose = MODEL.fk(res.q)
    assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
    assert np.allclose(pose[:3, 3], [-0.50, 0.00, 0.2975], atol=1e-4)


def test_ik_top_down_tcp_offset_matches_sim_check_fixture() -> None:
    """Fixture `pregrasp` pose: TCP (-0.50, 0.00, 0.1725), yaw 0, TCP = tool0 + 0.125 m along z."""
    res = MODEL.ik_top_down((-0.50, 0.00, 0.1725), 0.0, HOME_Q, tcp_offset_m=0.125)
    assert res.success
    assert MODEL.within_limits(res.q)
    pose = MODEL.fk(res.q)
    tcp = pose[:3, 3] + 0.125 * pose[:3, 2]
    assert np.allclose(tcp, [-0.50, 0.00, 0.1725], atol=1e-4)
    assert np.allclose(pose[:3, 3], [-0.50, 0.00, 0.2975], atol=1e-4)
    assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)


@settings(deadline=None)
@given(
    x=st.floats(-0.65, -0.35),
    y=st.floats(-0.22, 0.22),
    yaw=st.floats(-np.pi / 4, np.pi / 4),
)
def test_ik_top_down_over_workspace(x: float, y: float, yaw: float) -> None:
    res = MODEL.ik_top_down((x, y, 0.1725), yaw, HOME_Q, tcp_offset_m=0.125)
    assert res.success
    pose = MODEL.fk(res.q)
    assert np.allclose(pose[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
    assert np.allclose(pose[:3, 0], [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-3)
