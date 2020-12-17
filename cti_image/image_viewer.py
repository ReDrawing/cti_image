import cv2 as cv

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, CameraInfo

class ImageViewer(Node):
    '''!
        Cria uma janela de visualização da imagem em um topico
    '''

    def __init__(self):
        super().__init__('bodypix')

        self.subImg = self.create_subscription(Image, "image_raw/image", self.imageCallback, 10)


    def imageCallback(self, msg):
        '''!
            Recebe a imagem e mostra na tela
        '''
        
        img = np.array(msg.data, dtype=np.uint8)
        img = np.array(np.split(img, msg.height))
        img = np.array(np.split(img, msg.width, axis=1))

        img = np.rot90(img)

        cv.imshow("camera", img)

        if cv.waitKey(1) == ord('q'):
            pass

        
def main(args=None):
    rclpy.init(args=args)

    node = ImageViewer()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
