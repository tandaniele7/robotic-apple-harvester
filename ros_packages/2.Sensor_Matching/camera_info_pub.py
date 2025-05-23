import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

from std_msgs.msg import String


class CameraInfoPublisher(Node):

    def __init__(self):
        super().__init__("camera_info_publisher")
        rclpy.logging.set_logger_level("CameraInfo", 10)

        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/camera/color/camera_info", 10
        )

        self.depth_info_pub = self.create_publisher(
            CameraInfo, "/camera/aligned_depth_to_color/camera_info", 10
        )

        timer_period = 0.066666  # 15 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        print('\n\n')
        self.get_logger().info("Publishing...")        

    def timer_callback(self):

        msg = CameraInfo()
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.height = 480
        msg.width = 640
        #msg.height = 720
        #msg.width = 1280
        msg.distortion_model = "plumb_bob"  #this correspond to the brown_corrady model
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [
            633.724426269531,
            0.0,
            919.856811523438,
            0.0,
            919.407348632812,
            364.202056884766,
            0.0,
            0.0,
            1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            617.4224243164062,
            0.0,
            316.72235107421875,
            0.0,
            0.0,
            617.7899780273438,
            244.21875,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        msg.binning_x = 0
        msg.binning_y = 0
        msg.roi.x_offset = 0
        msg.roi.y_offset = 0
        msg.roi.height = 0
        msg.roi.width = 0
        msg.roi.do_rectify = False

        self.camera_info_pub.publish(msg)

        msg.header.frame_id = "camera_color_optical_frame"
        msg.k = [
            387.34246826171875,
            0.0,
            321.9106750488281,
            0.0,
            387.34246826171875,
            236.75909423828125,
            0.0,
            0.0,
            1.0,
        ]
        msg.p = [
            387.34246826171875,
            0.0,
            321.9106750488281,
            0.0,
            0.0,
            387.34246826171875,
            236.75909423828125,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        self.depth_info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = CameraInfoPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
