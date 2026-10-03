#!/usr/bin/env bash
set -eo pipefail
# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash"
if [ -f /work/ros_ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  source /work/ros_ws/install/setup.bash
fi
exec "$@"
