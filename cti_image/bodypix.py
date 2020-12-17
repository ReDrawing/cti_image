import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import tf_bodypix.api
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, CameraInfo

class BodyPixROS(Node):
    '''!
        Utiliza o modelo "BodyPix" para segmentar o corpo
        de uma pessoa na imagem de um topico
    '''

    def __init__(self):
        super().__init__('bodypix')
        self.pubMask = self.create_publisher(Image, "image_raw/body_mask", 10)
        self.colMask = self.create_publisher(Image, "image_raw/colored_body_mask", 10)

        self.subImg = self.create_subscription(Image, "image_raw/image", self.imageCallback, 10)

        self.bodypix_model = load_model(download_model(
            BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_16
        ))

    def imageCallback(self, msg):
        '''!
            Recebe a imagem, seg,emta e publica
        '''

        img = np.array(msg.data, dtype=np.uint8)
        img = np.array(np.split(img, msg.height))
        img = np.array(np.split(img, msg.width, axis=1))

        img = np.rot90(img)

        img = tf.keras.preprocessing.image.array_to_img(img)
        image_array = tf.keras.preprocessing.image.img_to_array(img)
        
        result = self.bodypix_model.predict_single(image_array)
        mask = result.get_mask(threshold=0.75)
        colored_mask = result.get_colored_part_mask(mask)
        
        mask = mask.numpy().astype(np.uint8)
        colored_mask = colored_mask.astype(np.uint8)

        maskMsg = Image()
        maskMsg._data = mask.flatten().tolist()
        maskMsg.height = mask.shape[0]
        maskMsg.width = mask.shape[1]
        maskMsg.encoding = "8UC1"
        maskMsg.is_bigendian = 0
        maskMsg.step = maskMsg.width

        colMsg = Image()
        colMsg._data = colored_mask.flatten().tolist()
        colMsg.height = colored_mask.shape[0]
        colMsg.width = colored_mask.shape[1]
        colMsg.encoding = "8UC3"
        colMsg.is_bigendian = 0
        colMsg.step = colMsg.width*3


        maskMsg.header = msg.header
        colMsg.header = msg.header

        self.pubMask.publish(maskMsg)
        self.colMask.publish(colMsg)

def main(args=None):
    rclpy.init(args=args)

    node = BodyPixROS()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
