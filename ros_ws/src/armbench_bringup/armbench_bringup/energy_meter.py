"""``energy_meter`` node: integrates the energy model on the 100 Hz energy_state_broadcaster.

Publishes ``/armbench/energy`` (``std_msgs/Float64MultiArray``:
``[duration_s, wh_A_nominal, wh_B_nominal]``) and ``/armbench/energy_table``
(``std_msgs/String`` JSON with the full variant x eta table) at ``publish_hz``.
``/armbench/energy/reset`` (``std_srvs/Trigger``) restarts the integration (one call per
episode). With ``--log`` every sample is appended as JSONL so the Wh can be recomputed offline.
The model itself lives in the pure-Python package ``armbench.energy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

from armbench.energy import EnergyMeter, Variant, load_energy_params
from armbench.energy.params import DEFAULT_ENERGY_FILE

DEFAULT_TOPIC = "/energy_state_broadcaster/joint_states"


class EnergyMeterNode(Node):
    def __init__(self, config: Path, log: Path | None, publish_hz: float, topic: str) -> None:
        super().__init__("armbench_energy_meter")
        self.params = load_energy_params(config)
        self.joints = list(self.params.joint_order)
        self.meter = EnergyMeter(self.params)
        self.log_fh = log.open("a") if log else None
        self.n_dropped = 0
        self.create_subscription(JointState, topic, self._on_js, 200)
        self.pub = self.create_publisher(Float64MultiArray, "/armbench/energy", 10)
        self.pub_table = self.create_publisher(String, "/armbench/energy_table", 10)
        self.create_service(Trigger, "/armbench/energy/reset", self._on_reset)
        self.create_timer(1.0 / publish_hz, self._publish)

    def _on_js(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if any(j not in idx for j in self.joints) or len(msg.effort) != len(msg.name):
            self.n_dropped += 1
            return
        sel = [idx[j] for j in self.joints]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        qd = [msg.velocity[i] for i in sel]
        tau = [msg.effort[i] for i in sel]
        try:
            self.meter.update(t, qd, tau)
        except ValueError as exc:  # time jump backwards (sim reset) -> start over
            self.get_logger().warning(f"meter reset: {exc}")
            self.meter.reset()
            self.meter.update(t, qd, tau)
        if self.log_fh is not None:
            q = [msg.position[i] for i in sel]
            self.log_fh.write(json.dumps({"t": t, "q": q, "qd": qd, "tau": tau}) + "\n")

    def _on_reset(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        self.meter.reset()
        res.success = True
        res.message = "energy meter reset"
        return res

    def _publish(self) -> None:
        a = self.meter.breakdown(Variant.A)
        b = self.meter.breakdown(Variant.B)
        msg = Float64MultiArray()
        msg.data = [a.duration_s, a.total_wh, b.total_wh]
        self.pub.publish(msg)
        table = [
            {**row.model_dump(mode="json"), "total_wh": row.total_wh} for row in self.meter.table()
        ]
        self.pub_table.publish(String(data=json.dumps({"rows": table, "dropped": self.n_dropped})))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENERGY_FILE)
    parser.add_argument("--log", type=Path, default=None, help="append samples as JSONL")
    parser.add_argument("--publish-hz", type=float, default=10.0)
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="JointState topic to integrate (default: the 100 Hz energy_state_broadcaster)",
    )
    args = parser.parse_args()
    rclpy.init()
    node = EnergyMeterNode(args.config, args.log, args.publish_hz, args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.log_fh is not None:
            node.log_fh.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
