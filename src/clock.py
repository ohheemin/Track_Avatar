#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Int32
import time

BUFFER_SIZE = 1000000

class TimingComparator(Node):
    def __init__(self):
        super().__init__("timing_comparator")
        self._buf_clock: dict[int, float] = {}
        self._buf_index: dict[int, float] = {}

        self._sub_clock = self.create_subscription(
            Int32, "/vision_clock", self._cb_clock, 10)
        self._sub_index = self.create_subscription(
            Int32, "/index", self._cb_index, 10)

        self._clock_pub = self.create_publisher(Float64, "/calculated_clock", 10)
        self._timer = self.create_timer(1.0, self._report)

        self.get_logger().info(
            f"TimingComparator ready  (buffer={BUFFER_SIZE})\n"
            "  Subscribing: /vision_clock  /index\n"
            "  Publishing:  /calculated_clock"
        )

    def _cb_clock(self, msg: Int32):
        frame_idx = msg.data
        self._buf_clock[frame_idx] = time.time()
        self._trim(self._buf_clock)
        self._try_match(frame_idx)

    def _cb_index(self, msg: Int32):
        frame_idx = msg.data
        print(frame_idx)
        self._buf_index[frame_idx] = time.time()
        self._trim(self._buf_index)
        self._try_match(frame_idx)

    def _trim(self, buf: dict):
        while len(buf) > BUFFER_SIZE:
            del buf[min(buf.keys())]

    def _try_match(self, frame_idx: int):
        if frame_idx in self._buf_clock and frame_idx in self._buf_index:
            t_clock = self._buf_clock.pop(frame_idx)
            t_index = self._buf_index.pop(frame_idx)
            diff_ms = (t_index - t_clock) * 1000.0
            self.get_logger().info(
                f"[frame {frame_idx:05d}]  "
                f"diff(index - vision_clock) = {diff_ms:+.2f} ms"
            )
            msg = Float64()
            msg.data = diff_ms
            self._clock_pub.publish(msg)

    def _report(self):
        self.get_logger().info(
            f"subscribing... /vision_clock and /index"
        )

def main():
    rclpy.init()
    node = TimingComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()