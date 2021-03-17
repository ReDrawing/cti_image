import cv2 as cv

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from cti_interfaces.msg import BodyPose

class BodyViewer(Node):
    '''!
        Exibe uma imagem e poses corporais recebidas via mensagem.

        Possui um histórico das últimas 5 poses para mostrar na tela
    '''

    def __init__(self):
        super().__init__('body_viewer')

        self.img = None
        self.kp = []

        self.updateTimer = self.create_timer(0.033, self.updateCallback)

        self.subImg = self.create_subscription(Image, "camera/image", self.imageCallback, 10)
        self.subBodyPose = self.create_subscription(BodyPose,"user/body_pose", self.poseCallback, 10)

        
    def updateCallback(self):
        """!
            Atualiza a imagem na tela.

            Utiliza a última imagem recebida e o histórico de poses
        """

        if self.img is None or self.kp is None:
            return

        imgDraw = self.img.copy()

        for keypoints in self.kp:
            for point in keypoints:
                cv.circle(imgDraw, tuple([int(point[0]), int(point[1])]), 10, (0,0,255), -1)

        cv.imshow("body_pose", imgDraw)

        if cv.waitKey(1) == ord('q'):
            pass


    def imageCallback(self, msg):
        '''!
            Recebe a imagem e armazena.

            Parâmetros:
                @param msg (Imagem) - Imagem
        '''
        
        img = np.array(msg.data, dtype=np.uint8)
        img = np.array(np.split(img, msg.height))
        img = np.array(np.split(img, msg.width, axis=1))

        img = np.rot90(img)

        self.img = img

    def poseCallback(self, msg):
        """!
            Recebe a pose corporal e armazena.

            Ignora keypoints que não estejam no pixel space

            Parâmetros:
                @param msg (cti_interfaces.BodyPose) - Pose corporal
        """

        kp = []

        for msgKp in msg.keypoints:
            if(msgKp.pixel_space == False):
                self._logger.error("Can't show keypoints that are not in pixel space")

            kp.append([msgKp.pose.pose.position.x, msgKp.pose.pose.position.y])

        self.kp.append(kp)

        if len(self.kp) > 5:
            self.kp.pop(0)


        
def main(args=None):
    rclpy.init(args=args)

    node = BodyViewer()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()