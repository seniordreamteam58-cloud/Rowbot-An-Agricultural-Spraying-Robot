import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class FixRotation(Node):
    def __init__(self):
        super().__init__('fix_rotation')

        self.declare_parameter('min_ang_vel', 1.0)
        self.declare_parameter('lin_eps', 0.02)
        self.declare_parameter('ang_eps', 0.01)

        self.min_ang_vel = float(self.get_parameter('min_ang_vel').value)
        self.lin_eps = float(self.get_parameter('lin_eps').value)
        self.ang_eps = float(self.get_parameter('ang_eps').value)

        self.sub = self.create_subscription(Twist, '/cmd_vel_nav', self.cmd_vel_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel_fixed', 10)

        self.get_logger().info(f'Rotate guard active | min_ang_vel={self.min_ang_vel}')

    def cmd_vel_cb(self, msg: Twist):
        out = Twist()
        out.linear = msg.linear
        out.angular = msg.angular

        lin_zero = abs(msg.linear.x) < self.lin_eps and abs(msg.linear.y) < self.lin_eps
        ang_cmd = abs(msg.angular.z) > self.ang_eps
        ang_too_slow = abs(msg.angular.z) < self.min_ang_vel

        if lin_zero and ang_cmd and ang_too_slow:
            out.angular.z = self.min_ang_vel if msg.angular.z > 0.0 else -(self.min_ang_vel)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FixRotation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

