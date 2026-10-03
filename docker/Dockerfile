# Simulation image: ROS 2 Jazzy + Gazebo Harmonic + UR5e packages.
# Phase F0 only needs an empty headless world; the UR/Robotiq/Pinocchio
# packages are installed now so later phases reuse the cached layer.
# ur_simulation_gz pulls rviz2/MoveIt (+0.21 GB measured); kept because the F1
# gate is precisely that its launch files boot the UR5e headless (ADR-002).
FROM ros:jazzy-ros-base@sha256:066420e07f60aa18262f2479981def87ebcfcec42eefb0c0c57c4a46098348ca

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    ROS_DISTRO=jazzy \
    GZ_VERSION=harmonic \
    PYTHONUNBUFFERED=1

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-gz-ros2-control \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-ur-description \
        ros-jazzy-ur-simulation-gz \
        ros-jazzy-robotiq-description \
        ros-jazzy-pinocchio \
        ros-jazzy-cv-bridge \
        ros-jazzy-image-transport \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-xacro \
        python3-colcon-common-extensions \
        python3-opencv \
        python3-numpy \
        libgl1 \
        libegl1 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        mesa-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

WORKDIR /work

# ROS 2 workspace (armbench_description, armbench_bringup): launch files, xacro, worlds, rclpy
# nodes only, so the build takes seconds and the layer is small.
COPY ros_ws/src /work/ros_ws/src
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cd /work/ros_ws \
    && colcon build --event-handlers console_direct+ \
    && rm -rf build log"

# Pure-Python `armbench` package (energy meter now, primitives later) importable from the
# rclpy nodes. --no-deps keeps the apt numpy 1.26 that rclpy/cv_bridge/pinocchio are built
# against; the two missing runtime deps are added explicitly.
COPY pyproject.toml README.md LICENSE /work/armbench/
COPY src /work/armbench/src
COPY configs /work/armbench/configs
ENV ARMBENCH_CONFIG_DIR=/work/armbench/configs
RUN uv pip install --system --break-system-packages --no-cache --no-deps /work/armbench \
    && uv pip install --system --break-system-packages --no-cache "pydantic>=2.7,<3" "typer>=0.12,<1" \
    && python3 -c "import armbench.energy, pydantic, numpy; print(armbench.__version__, numpy.__version__)"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
