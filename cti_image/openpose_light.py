import numpy as np
import cv2 as cv
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from .lightweight_human_modules.models.with_mobilenet import PoseEstimationWithMobileNet
from .lightweight_human_modules.keypoints import extract_keypoints, group_keypoints
from .lightweight_human_modules.load_state import load_state
from .lightweight_human_modules.pose import Pose, track_poses
from .lightweight_human_modules.image_tools import normalize, pad_width

class OpenPoseLight(Node):
    def __init__(self):
        super().__init__('openpose_light')

        self.subImg = self.create_subscription(Image, "image_raw/image", self.imageCallback, 10)

    
    def imageCallback(self, msg):
        img = np.array(msg.data, dtype=np.uint8)


def main(args=None):
    rclpy.init(args=args)

    node = OpenPoseLight()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

