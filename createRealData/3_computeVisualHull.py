import numpy as np
import os
import cv2
from skimage import measure
import trimesh as trm
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import argparse
# code for reconstruct visual hull mesh
from scipy.io import loadmat
import open3d as o3d


def quaternionToRotation(Q):
    # q = a + bi + cj + dk
    a = float(Q[0])
    b = float(Q[1])
    c = float(Q[2])
    d = float(Q[3])

    R = np.array([[2*a**2-1+2*b**2, 2*b*c+2*a*d,     2*b*d-2*a*c],
                  [2*b*c-2*a*d,     2*a**2-1+2*c**2, 2*c*d+2*a*b],
                  [2*b*d+2*a*c,     2*c*d-2*a*b,     2*a**2-1+2*d**2]])
    return np.transpose(R)


parser = argparse.ArgumentParser()
parser.add_argument('--scene', default='Scene')
parser.add_argument('--shapeName', default='Mouse')
parser.add_argument('--nViews', type=int, default=10)
parser.add_argument('--asDataset', action='store_true', help='use this tag to change output path to real data folder')
parser.add_argument('--shapeId', type=int, default=0, help='activate when asDataset is true, indicate shape id for the shape')   
# use ClearPose data to provide segmentation, camera pose
parser.add_argument('--data_src', type=str, default='/media/cxt/6358C6357FEBD1E6/clearpose_downsample_100/set4/scene6')
parser.add_argument('--img_idx_start', type=int, default=100)
parser.add_argument('--img_idx_end',   type=int, default=1800)
parser.add_argument('--img_idx_step',  type=int, default=100, help='(end - start) / step should be integer')

parser.add_argument('--mask_suffix', type=str, default='label.png', help='switch this for gt/pred mask for label.png/label-predict.png')
parser.add_argument('--meta_file', type=str, default='metadata.mat')

parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--xmin', type=float, default=-1.7)
parser.add_argument('--xmax', type=float, default=1.7)
parser.add_argument('--ymin', type=float, default=-1.7)
parser.add_argument('--ymax', type=float, default=1.7)
parser.add_argument('--zmin', type=float, default=-1.7)
parser.add_argument('--zmax', type=float, default=1.7)

parser.add_argument('--output_dir', type=str, default='./visualhull_output')
parser.add_argument('--save_intermediate',   type=bool, default=False)
parser.add_argument('--point_num',   type=int, default=10000)

opt = parser.parse_args()

numMask = (opt.img_idx_end - opt.img_idx_start) // opt.img_idx_step + 1
setname, scenename = opt.data_src.split('/')[-2], opt.data_src.split('/')[-1]
if opt.mask_suffix == 'label.png': # gt
    mask_savelabel = 'gt'
else:
    mask_savelabel = 'pred'
meshName = f'{opt.output_dir}/{setname}_{scenename}_{opt.img_idx_start}_{opt.img_idx_end}_{opt.img_idx_step}_{mask_savelabel}_visualhull.obj'
pointcloudName = meshName.replace('.obj', '.ply')

os.system(f'mkdir -p {opt.output_dir}')

buildVisualHull = True
plot3Dcam = False

metadata = loadmat(f'{opt.data_src}/{opt.meta_file}')
cam_intr = metadata['000000']['intrinsic_matrix'][0][0]
f, cxp, cyp = cam_intr[0, 0], cam_intr[0, 2], cam_intr[1, 2]
imgH, imgW = cv2.imread(f'{opt.data_src}/{opt.img_idx_start:06d}-depth.png', -1).shape


camDict = {}
for i in range(opt.img_idx_start, opt.img_idx_end+1, opt.img_idx_step):
    paramDict = {}
    rt = metadata[f'{i:06d}']['rotation_translation_matrix'][0][0]
    R, T = rt[:, :3], rt[:, 3]
    R, T = R.T, -R.T @ T
    paramDict['Rot'] = R
    paramDict['Trans'] = T
    paramDict['Origin'] = -R.T @ T
    paramDict['Target'] = R[2] + paramDict['Origin']
    paramDict['Up'] = -R[1]
    paramDict['cId'] = i
    paramDict['imgName'] = f'{i:06d}'
    camDict[i] = paramDict

# initialize visual hull voxels
resolution = opt.resolution

minX, maxX = opt.xmin, opt.xmax
minY, maxY = opt.ymin, opt.ymax
minZ, maxZ = opt.zmin, opt.zmax

y, x, z = np.meshgrid(
        np.linspace(minX, maxX, resolution),
        np.linspace(minY, maxY, resolution),
        np.linspace(minZ, maxZ, resolution)
        )

x = x[:, :, :, np.newaxis]
y = y[:, :, :, np.newaxis]
z = z[:, :, :, np.newaxis]
coord = np.concatenate([x, y, z], axis=3)
volume = -np.ones(x.shape).squeeze()

############################
# Start to build the voxel #
############################
if plot3Dcam == True:
    centers = []
    lookups = []
    ups = []
    ids = []


for i in range(opt.img_idx_start, opt.img_idx_end+1, opt.img_idx_step):
    print('Processing mask idx:'.format(i))
    mask = cv2.imread(f'{opt.data_src}/{i:06d}-{opt.mask_suffix}', -1).reshape(imgH * imgW)
    mask[mask != 0] = 255
    imgId = i

    '''
    # If given extrinsic
    Rot = camDict[imgId]['Rot']
    Trans = camDict[imgId]['Trans']
    Trans = Trans.reshape([1, 1, 1, 3, 1])
    coordCam = np.matmul(Rot, np.expand_dims(coord, axis=4)) + Trans
    '''
    # If given Origin, Target, Up
    origin = camDict[imgId]['Origin']
    C = origin

    target = camDict[imgId]['Target']
    up = camDict[imgId]['Up']

    if plot3Dcam == True:
        centers.append(origin)
        lookups.append((target-origin))
        ups.append(up)
        ids.append(imgId)

    if buildVisualHull == True:

        yAxis = up / np.sqrt(np.sum(up * up))
        zAxis = target - origin
        zAxis = zAxis / np.sqrt(np.sum(zAxis * zAxis))
        xAxis = np.cross(zAxis, yAxis)
        xAxis = xAxis / np.sqrt(np.sum(xAxis * xAxis))
        Rot = np.stack([xAxis, yAxis, zAxis], axis=0)
        coordCam = np.matmul(Rot, np.expand_dims(coord - C, axis=4))


        coordCam = coordCam.squeeze(4)
        xCam = coordCam[:, :, :, 0] / coordCam[:, :, :, 2]
        yCam = coordCam[:, :, :, 1] / coordCam[:, :, :, 2]

        xId = xCam * f + cxp
        yId = -yCam * f + cyp

        xInd = np.logical_and(xId >= 0, xId < imgW-0.5)
        yInd = np.logical_and(yId >= 0, yId < imgH-0.5)
        imInd = np.logical_and(xInd, yInd)

        xImId = np.round(xId[imInd]).astype(np.int32)
        yImId = np.round(yId[imInd]).astype(np.int32)

        maskInd = mask[yImId * imgW + xImId]

        volumeInd = imInd.copy()
        volumeInd[imInd == 1] = maskInd

        volume[volumeInd == 0] = 1

        print('Occupied voxel: %d' % np.sum((volume > 0).astype(np.float32)))

        # verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0)
        verts, faces, normals, _ = measure._marching_cubes_lewiner.marching_cubes(volume, 0)

        print('Vertices Num: %d' % verts.shape[0])
        print('Normals Num: %d' % normals.shape[0])
        print('Faces Num: %d' % faces.shape[0])

        if opt.save_intermediate:
            axisLen = float(resolution-1) / 2.0
            verts = (verts - axisLen) / axisLen * 1.7
            mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces)
            points = trm.sample.sample_surface(mesh, opt.point_num)[0]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            mesh.export(meshName.replace('.obj', f'{i}.obj'))
            o3d.io.write_point_cloud(pointcloudName.replace('.ply', f'{i}.ply'), pcd)

if buildVisualHull:
    axisLen = float(resolution-1) / 2.0
    verts = (verts - axisLen) / axisLen * 1.7
    mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces)
    points = trm.sample.sample_surface(mesh, opt.point_num)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh.export(meshName)
    o3d.io.write_point_cloud(pointcloudName, pcd)

if plot3Dcam == True:
    #code for visualize lookup and up vectors
    centers = np.stack(centers, axis=0)
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]
    lookups = np.stack(lookups, axis=0)
    lookU = lookups[:, 0]
    lookV = lookups[:, 1]
    lookW = lookups[:, 2]
    ups = np.stack(ups, axis=0)
    upsU = ups[:, 0]
    upsV = ups[:, 1]
    upsW = ups[:, 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, lookU, lookV, lookW, cmap='Reds')
    #ax.quiver(x, y, z, upsU, upsV, upsW, cmap='Blues')
    ax.scatter(x, y, z, c='r', marker='o')
    for i in range(len(ids)):
        ax.text(x[i], y[i], z[i], ids[i])

    plt.show()

# cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i %s -o %s -om vn -s remeshVisualHullNew.mlx' % \
#         ((opt.meshName), (opt.meshSubdName))

# os.system(cmd)
