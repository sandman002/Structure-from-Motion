#Author: Sandeep Manandhar, PhD.
#ENS PARIS
import numpy as np


def vecnorm(vec):
    return np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
class Camera:
    def __init__(self, focal_length=3, location=np.array([0,0,0]), orientation=np.identity(3)):
        self.t = location #in world coordinate
        self.R = orientation #in world coordinate
        self.R[2,2] = -1
        self.setPoseMatrix(self.R, self.t)
        self.lookat = self.R[:,2]
        self.f = focal_length
        self.intrinsic = np.array([[self.f,0,0],[0,self.f,0],[0,0,1]])
        self.setImPlane()
    def setPoseMatrix(self, R, C):
        # t = -np.matmul(R,C)
        self.pose = np.c_[R, C]
        self.pose = np.vstack([self.pose, np.array([0,0,0,1])])

    def rotate(self, newR):
        self.R = np.matmul(newR, self.R)
        self.setPoseMatrix(self.R, self.t)

    def translate(self, newt):
        self.t = self.t + newt
        self.setPoseMatrix(self.R, self.t)

    def computeLookatvec(self, object_center):
        vec = self.t- object_center 
        self.lookat =  vec/vecnorm(vec)


    def computeRotationFromLookAT(self):
        caz = self.lookat
        proj_lookat = np.array([caz[0], caz[1], caz[2]])
        
        proj_lookat[2] = 0 #projected to xy plane

        cam_z_axis = self.R[:,2]
        if np.linalg.norm(proj_lookat) != 0: #if projection is not 0
            perp_vec = np.cross(proj_lookat, cam_z_axis) #find perpendicual vector to both proj_lookat and xyplane's normal

            n3 = self.lookat/np.linalg.norm(self.lookat)
            n2 = np.array([perp_vec[0], perp_vec[1], perp_vec[2]])
            n2 = n2/np.linalg.norm(n2)

            n1 = np.cross(n3,n2)
            n1 = n1/np.linalg.norm(n1)
            R = np.eye(3)
            R[:,0] = np.transpose(n1)
            R[:,1] = np.transpose(n2)
            R[:,2] = np.transpose(n3)
        else:   
            R = np.eye(3)

        return R

    def setImPlane(self, l = 5):
        focus = self.lookat*self.f
        self.implane = self.t - focus
        self.tr = self.implane + (self.R[:3,1]+ self.R[:3,0])*l
        self.tl = self.implane - (self.R[:3,1]+ self.R[:3,0])*l
        self.br = self.implane + (self.R[:3,1]- self.R[:3,0])*l
        self.bl = self.implane - (self.R[:3,1]- self.R[:3,0])*l

    def snapshot(self, point3d): #point3d from world frame  
        pointIm = []
        camcoord = []
        for p3d in point3d:
            cam_coordinate = (np.matmul(np.linalg.inv(self.R),p3d - self.t))
          
            im_coordinate = np.matmul(self.intrinsic, cam_coordinate)
            im_coordinate[2] = 0
            pointIm.append(im_coordinate)
            camcoord.append(cam_coordinate)
     
        return np.array(pointIm), np.array(camcoord)
        
