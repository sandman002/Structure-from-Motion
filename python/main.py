#Author: Sandeep Manandhar, PhD.
#ENS PARIS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from objects import * #these are coordinates of object in scene
from camera import Camera
def makeCamera(initpose):
    R1 = np.identity(3)
    R1 = np.c_[R1, initpose]
    R1[2,2] = -1
    l =np.array([0,0,0,1])
    R1 = np.vstack([R1, l])
    return R1


cam =  Camera(focal_length = 3,location=np.array([10,0,8]))

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
imaxes = []
imaxes.append(fig.add_subplot(222))
imaxes.append(fig.add_subplot(223))
imaxes.append(fig.add_subplot(224))
ax.set_xlabel('X',fontsize=15)
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Diamond')
lims=25
ax.axes.set_xlim3d(left  =-lims, right=lims)
ax.axes.set_ylim3d(bottom=-lims, top=lims) 
ax.axes.set_zlim3d(bottom=-lims, top=lims) 
for vertex in vertices:
    ax.scatter(*vertex)

obj_center = np.mean(vertices, axis=0)
print("object center", obj_center)
ax.scatter(*obj_center)
# Plot the edges
for edge,c in zip(edges,colors):
    x = [vertices[edge[0]][0], vertices[edge[1]][0]]
    y = [vertices[edge[0]][1], vertices[edge[1]][1]]
    z = [vertices[edge[0]][2], vertices[edge[1]][2]]
    ax.plot(x, y, z, color=c)


cams = []
im_pts = []
camcoord = []
cam_init = np.array([[20,-23, 8], [18,4,-4], [7.5, 11, 2]])
for i in range(3):
    cam =  Camera(focal_length = 5,location=cam_init[i])
    cam.computeLookatvec(obj_center)
    R = cam.computeRotationFromLookAT()
    cam.rotate((R))
    cam.setImPlane()
    plotcam(ax, cam.pose, 3)
    ax.scatter(*cam.implane, marker='s', color='gray')
    cams.append(cam)
    plotImPlane(ax, cam)
    pts, camcoords = cam.snapshot(vertices)
    im_pts.append(pts)
    camcoord.append(camcoords)
    plotImages(imaxes[i], pts)


# plt.show()
##
#######
#SFM begins
#Known = im_pts only
#build obseration matrix
obs_mat_u = np.empty((0,im_pts[0].shape[0]))
obs_mat_v = np.empty((0,im_pts[0].shape[0]))
for i in range(len(im_pts)):
    points = im_pts[i]
    u_mean = np.mean(points[:,0])
    v_mean = np.mean(points[:,1])
    row_u = []
    row_v = []
    for j in range(points.shape[0]): #all im points
        points[j,0] = points[j,0] - u_mean
        points[j,1] = points[j,1] - v_mean
        row_u.append(points[j,0])
        row_v.append(points[j,1])
    obs_mat_u = np.vstack((obs_mat_u,np.array(row_u)))
    obs_mat_v = np.vstack((obs_mat_v,np.array(row_v)))
print("Observation matrix")
print(obs_mat_u)

obsmat = np.vstack((obs_mat_u, obs_mat_v))
print(obsmat)
rank_obs_mat = np.linalg.matrix_rank(obsmat)
print("RANK of observation matrix: ",rank_obs_mat)

'''
W = MxS
W(2FXN) = M(2FX3) x S(3xN)
F = number of observation
M = [i.T, j.T]
'''

U,S,V = np.linalg.svd(obsmat)
U = U[:,:3]
Sm = np.diag(S[:3])
V = V[:3,:]

re_obsmat= (U@Sm@V)
##Tomasi-kanade factorization
M_ = U@np.diag(np.sqrt(S[:3]))
S_ = np.diag(np.sqrt(S[:3]))@V


i = M_[:3,:]
j = M_[3:,:]


def objective(Q):
    residuals = []
    Q = Q.reshape(3,3)
    for idx in range(i.shape[0]):
        qi = np.dot(Q.T, i[idx])
        qj = np.dot(Q.T, j[idx])
        residuals.append(np.dot(qi,qi) -1)
        residuals.append( np.dot(qj, qj) - 1)
        residuals.append(np.dot(qi, qj))
    return residuals



from scipy.optimize import least_squares
init_Q = np.eye(3)

# res = objective(init_Q)

# result = least_squares(objective, init_Q.flatten())
Q = init_Q#result.x.reshape(3, 3)
print("Matrix Q:")
print(Q)

M = np.matmul(M_,Q)
Ss = np.matmul(np.linalg.inv(Q), S_)

print("M: \n", M)
print("S:\n", Ss)
# print("Vertices:\n", camcoord)
# plt.show()

ax3 = fig.add_subplot(222, projection='3d')


re_vertices = []
for jj in range(Ss[0].shape[0]):
    re_vertices.append(np.array([Ss[0][jj], Ss[1][jj] , Ss[2][jj]]))

re_vertices = np.array(re_vertices)
ax.scatter(*obj_center)
# Plot the edges
for edge,c in zip(edges,colors):
    x = [re_vertices[edge[0]][0], re_vertices[edge[1]][0]]
    y = [re_vertices[edge[0]][1], re_vertices[edge[1]][1]]
    z = [re_vertices[edge[0]][2], re_vertices[edge[1]][2]]
    ax3.plot(x, y, z, color=c)

pos = np.array([0,0,0])
for jj in range(1):
    n1 = M[0+jj,:]*3
    n2 = M[3+jj,:]*3
    n3 = np.cross(n1,n2)
    ax.plot([pos[0],pos[0]+n1[0]], [pos[1], pos[1]+n1[1]], [pos[2],pos[2]+n1[2]], 'r')
    ax.plot([pos[0],pos[0]+n2[0]], [pos[1], pos[1]+n2[1]], [pos[2],pos[2]+n2[2]], 'g')
    ax.plot([pos[0],pos[0]+n3[0]], [pos[1], pos[1]+n3[1]], [pos[2],pos[2]+n3[2]], 'b')


plt.show()