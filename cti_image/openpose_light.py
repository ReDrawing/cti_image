import os

import numpy as np
import cv2 as cv
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

from cti_interfaces.msg import BodyPose, BodyKeypoint

from cti_image import lightweight_human_modules as lhm
from .lightweight_human_modules.models.with_mobilenet import PoseEstimationWithMobileNet
from .lightweight_human_modules.keypoints import extract_keypoints, group_keypoints
from .lightweight_human_modules.load_state import load_state
from .lightweight_human_modules.pose import Pose, track_poses
from .lightweight_human_modules.image_tools import normalize, pad_width

import inspect


class OpenPoseLight(Node):
    '''!
        Estima poses corporais em imagens."
        
        Utiliza o modelo disponível em https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
    '''

    ## Dicionário que mapeia os keypoints do modelo para a mensagem
    keypointDict = {'nose'  : BodyKeypoint.NOSE, 
                    'neck'  : BodyKeypoint.NECK,
                    'r_sho' : BodyKeypoint.SHOULDER_R, 
                    'r_elb' : BodyKeypoint.ELBOW_R, 
                    'r_wri' : BodyKeypoint.WRIST_R,   
                    'l_sho' : BodyKeypoint.SHOULDER_L, 
                    'l_elb' : BodyKeypoint.ELBOW_L, 
                    'l_wri' : BodyKeypoint.WRIST_L,
                    'r_hip' : BodyKeypoint.HIP_R, 
                    'r_knee': BodyKeypoint.KNEE_R, 
                    'r_ank' : BodyKeypoint.ANKLE_R, 
                    'l_hip' : BodyKeypoint.HIP_L, 
                    'l_knee': BodyKeypoint.KNEE_L, 
                    'l_ank' : BodyKeypoint.ANKLE_L,
                    'r_eye' : BodyKeypoint.EYE_R, 
                    'l_eye' : BodyKeypoint.EYE_L,
                    'r_ear' : BodyKeypoint.EAR_R, 
                    'l_ear' : BodyKeypoint.EAR_L}

    ## Lista que mapeia os keypoints do modelo para a mensagem.
    keypointList = list(keypointDict.items())

    def __init__(self):
        super().__init__('openpose_light')

        self.subImg  = self.create_subscription(Image, "camera/image", self.imageCallback, 10)
        self.subInfo = self.create_subscription(CameraInfo, "camera/camera_info", self.infoCallback, 10)

        self.pubPose = self.create_publisher(BodyPose, "user/body_pose", 10)

        self.gpu = False

        self.header = None

        lhmPath = os.path.abspath(lhm.__file__)
        lhmPath = lhmPath[:-11]
        checkpointPath = lhmPath + "openpose_light.pth"
        checkpoint = torch.load(checkpointPath, map_location='cpu')

        self.net = PoseEstimationWithMobileNet()
        load_state(self.net, checkpoint)

        self.net = self.net.eval()
        if self.gpu :
            self.net = self.net.cuda()

    
    def infoCallback(self, msg):
        '''!
            Recebe o info da câmera, e guarda o header.

            Parâmetros:
                @param nome (CameraInfo) - imagem com a info recebida 
        '''
        self.header = msg.header

    def imageCallback(self, msg):
        '''!
            Recebe a imagem, processa e publica as poses.

            Parâmetros:
                @param msg (Image) - mensagem com a imagem recebida 
        '''
        img = np.array(msg.data, dtype=np.uint8)
        img = np.array(np.split(img, msg.height))
        img = np.array(np.split(img, msg.width, axis=1))

        img = np.rot90(img)

        poses = self.getPose(img)

        for pose in poses:
            self.publishPose(pose)



    
    
    
    def imageFormat(self, img, net_input_height_size, stride, pad_value, img_mean, img_scale):
        '''!
            Formata a imagem para o formato necessário para a rede
            
        '''

        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
        
        return padded_img, scale, pad
    
    def do_inference(self, upsample_ratio, padded_img):
        '''!
            Realiza a inferência.

            Paramêtros:
                @param upsample_ratio - Tava de redimensionamento
                @param padded_img (numpy array) - Imagem Formatada

            Retorno:
                @return heatmap - Heatmap de onde estão keypoints
                @return pafs - 
                
        '''

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        
        if self.gpu:
            tensor_img = tensor_img.cuda()
        
        stages_output = self.net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv.INTER_CUBIC)
        
        return heatmaps, pafs
    
    def inference(self, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        '''!
            Formata a imagem original para realizar a infêrencia.

            Paramêtros:
                @param img (numpy array) - Imagem Original
                @param net_input_height_size - Dimensão de altura do input da rede
                @param stride - 
                @param upsample_ratio - 

            Retorno:
                @return heatmap - Heatmap de onde estão keypoints
                @return pafs -
                @return scale - Escala da Imagem modificada
                @return pad - 
                
        '''

        padded_img, scale, pad = self.imageFormat(img, net_input_height_size, stride, pad_value, img_mean, img_scale)
        
        heatmaps, pafs = self.do_inference(img, upsample_ratio, padded_img)  
        return heatmaps, pafs, scale, pad

    def getPose(self, img, height_size=256):
        '''!
            Realiza a inferência e gera o vetor de poses corporais.

            Parâmetros
                @param img (numpy array) - imagem para realizar a inferência
                @param height_size (int)

            Retorno
                @return poses (list) - vetor de poses
        '''
           
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        heatmaps, pafs, scale, pad = self.inference(img, height_size, stride, upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        all_poses = []
        
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            all_poses.append(pose)

        return all_poses
    
    

    def publishPose(self, pose):
        '''!
            Gera a mensagem e publica a pose do corpo.

            Parâmetros:
                @param pose (lhm.Pose) - Pose a ser publicada
        '''

        msg = BodyPose()
        msg.user_id = BodyPose.UNKNOWN

        for i in range(18):
            
            if pose.keypoints[i][0] == -1:
                continue

            bKeypoint = BodyKeypoint()

            bKeypoint.joint_type = OpenPoseLight.keypointList[i][1]
            bKeypoint.pixel_space = True

            if self.header != None:
                bKeypoint.pose.header = self.header
            
            bKeypoint.pose.pose.orientation.x = 0.0
            bKeypoint.pose.pose.orientation.y = 0.0
            bKeypoint.pose.pose.orientation.z = 0.0
            bKeypoint.pose.pose.orientation.w = 1.0

            #Colocar posições dos (x,y) keypoints na imagem
            bKeypoint.pose.pose.position.x = float(pose.keypoints[i][0])
            bKeypoint.pose.pose.position.y = float(pose.keypoints[i][1])
            bKeypoint.pose.pose.position.z = 1.0
            
            msg.keypoints.append(bKeypoint)

        self.pubPose.publish(msg)


def main(args=None):
    rclpy.init(args=args)


    node = OpenPoseLight()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

