"""Save each /camera/image frame as a JPEG until a stop file appears (runs in the sim image)."""

import argparse
import pathlib
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, out: pathlib.Path) -> None:
        super().__init__("media_recorder")
        self.out = out
        self.n = 0
        self.create_subscription(Image, "/camera/image", self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        cv2.imwrite(str(self.out / f"{self.n:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.n += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--stop", type=pathlib.Path, required=True)
    ap.add_argument("--max-s", type=float, default=300.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Recorder(a.out)
    t0 = time.time()
    while not a.stop.exists() and time.time() - t0 < a.max_s:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
