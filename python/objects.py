#Author: Sandeep Manandhar, PhD.
#ENS PARIS
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

tx = 0
ty=0
tz=0
s = 5
vertices = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])*s +np.array([tx,ty,tz])


# Define the edges connecting the vertices
edges = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5)
]
colors = cm.rainbow(np.linspace(0, 1, len(edges)))



def computeRotationFromLookAT(campose, lookAtvec):
    n3 = campose[0:3,2] #current lookat of cam
    print(n3.shape, lookAtvec.shape)
    lookAtvec /= np.linalg.norm(lookAtvec) 
    n3 /= np.linalg.norm(n3) 


    rotation_axis = np.cross(lookAtvec, n3)
    
    angle = np.arccos(np.dot(lookAtvec, n3))
    costheta = np.cos(angle)
    sintheta = np.sin(angle)
    kx, ky, kz = rotation_axis
    K = np.array([[0, -kz, ky],[kz, 0, -kx], [-ky, kx, 0]])
    R = np.eye(3) + sintheta*K + (1-costheta) * np.dot(K, K)
    return R

def computeTransformation(R,t):
    R = np.c_[R, t]
    l =np.array([0,0,0,1])
    R = np.vstack([R, l])
    return R

def plotcam(ax, campose, mag=5):

    n1 = campose[0:3,0]*mag
    n2 = campose[0:3,1]*mag
    n3 = campose[0:3,2]*mag
    pos= campose[0:3,3]
    print(pos.shape)
    ax.plot(pos[0], pos[1], pos[2], 'or')
    ax.plot([pos[0],pos[0]+n1[0]], [pos[1], pos[1]+n1[1]], [pos[2],pos[2]+n1[2]], 'r')
    ax.plot([pos[0],pos[0]+n2[0]], [pos[1], pos[1]+n2[1]], [pos[2],pos[2]+n2[2]], 'g')
    ax.plot([pos[0],pos[0]+n3[0]], [pos[1], pos[1]+n3[1]], [pos[2],pos[2]+n3[2]], 'b')

def plotImPlane(ax, cam):
    x = [cam.tl[0], cam.br[0], cam.tr[0], cam.bl[0]]
    y = [cam.tl[1], cam.br[1], cam.tr[1], cam.bl[1]]
    z = [cam.tl[2], cam.br[2], cam.tr[2], cam.bl[2]]
    vertices = [list(zip(x,y,z))]
    rectangle = Poly3DCollection(vertices)
    rectangle.set_facecolor('red')
    rectangle.set_edgecolor('black')
    rectangle.set_alpha(0.4)
    ax.add_collection3d(rectangle)

def plotImages(ax, im_pts):
    ax.scatter(im_pts[:,1], im_pts[:,0])
    for edge,c in zip(edges,colors):
        x = [im_pts[edge[0]][0], im_pts[edge[1]][0]]
        y = [im_pts[edge[0]][1], im_pts[edge[1]][1]]

        ax.plot(y, x, color=c )