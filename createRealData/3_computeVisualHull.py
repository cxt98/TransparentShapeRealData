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
parser.add_argument('--point_n',   type=int, default=10000)

opt = parser.parse_args()

numMask = (opt.img_idx_end - opt.img_idx_start) // opt.img_idx_step + 1
setname, scenename = opt.data_src.split('/')[-2], opt.data_src.split('/')[-1]
meshName = f'{setname}_{scenename}_{opt.img_idx_start}_{opt.img_idx_end}_{opt.img_idx_step}__visualhull.obj'
pointcloudName = meshName.replace('.obj', '.ply')

os.system(f'mkdir -p {opt.output_dir}')

buildVisualHull = False
plot3Dcam = True

metadata = loadmat(f'{opt.data_src}/{opt.meta_file}')
cam_intr = metadata['000000']['intrinsic_matrix'][0][0]
f, cxp, cyp = cam_intr[0, 0], cam_intr[0, 2], cam_intr[1, 2]
imgH, imgW = cv2.imread(f'{opt.data_src}/{opt.img_idx_start:06d}-depth.png', -1).shape


camDict = {}
for i in range(opt.img_idx_start, opt.img_idx_end+1, opt.img_idx_step):
    paramDict = {}
    rt = metadata[f'{i:06d}']['rotation_translation_matrix'][0][0]
    R, T = rt[:, :3].T, rt[:, 3]
    paramDict['Rot'] = R
    paramDict['Trans'] = T
    paramDict['Origin'] = -R.T @ T
    paramDict['Target'] = R[2] + paramDict['Origin']
    paramDict['Up'] = -R[1]
    paramDict['cId'] = i
    paramDict['imgName'] = f'{i:06d}'
    camDict[i] = paramDict
# Read images file
# c = 0
# camDict = {}
# with open(imgFile, 'r') as camPoses:
#     for cam in camPoses.readlines():
#         c += 1
#         if c <= 3: # skip comments
#             continue
#         elif c == 4:
#             numImg = int(cam.strip().split(',')[0].split(':')[1])
#             print('Number of images:', numImg)
#         else:
#             if c % 2 == 1:
#                 line = cam.strip().split(' ')
#                 R = quaternionToRotation(line[1:5])
#                 paramDict = {}
#                 paramDict['Rot'] = R
#                 paramDict['Trans'] = np.array([float(line[5]), float(line[6]), float(line[7])])
#                 paramDict['Origin'] = -np.matmul(np.transpose(R), paramDict['Trans'])
#                 paramDict['Target'] = R[2, :] + paramDict['Origin']
#                 paramDict['Up'] = -R[1, :]
#                 paramDict['cId'] = int(line[0])
#                 name = line[9]
#                 paramDict['imgName'] = name
#                 nameId = name.split('_')[2].split('.')[0]
#                 camDict[nameId] = paramDict

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

# if writeOutputCamFile == True:
#     if os.path.exists(outputCamFile) == True:
#         print("{} exists!!! Insert y if want to overwrite:".format(outputCamFile))
#         overwrite = input()
#         if overwrite == 'y':
#             os.remove(outputCamFile)
#         else:
#             assert False, "{} exists!!!".format(outputCamFile)

#     with open(outputCamFile, 'w') as cam:
#         cam.write('{}\n'.format(numMask))

for i in range(opt.img_idx_start, opt.img_idx_end+1, opt.img_idx_step):
    print('Processing mask idx:'.format(i))
    #print(os.path.join(maskFolder, maskFiles[i]))
    #seg = cv2.imread(os.path.join(maskFolder, maskFiles[i]), cv2.IMREAD_UNCHANGED)[:,:,3]
    # seg = cv2.imread(maskFiles[i], cv2.IMREAD_UNCHANGED)[:,:,3]
    # baseName = os.path.basename(maskFiles[i])
    # if saveBinaryMask == True:
        # print(os.path.join(outputMaskFolder, baseName))
        # segResize = cv2.resize(seg, (resizeRGBimW, resizeRGBimH), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(os.path.join(outputMaskFolder, 'seg_{}.png'.format(i+1)), segResize)
    # mask = seg.reshape(imgH * imgW)

    # imgId = baseName.split('_')[2].split('.')[0]

    # if saveRGB == True:
    #     rgbName = os.path.join(sourceRGBFolder, baseName).replace('png', 'jpg')
    #     assert os.path.exists(rgbName), 'RGB image {} does not exist!'.format(rgbName)
    #     rgbImg = cv2.imread(rgbName)
    #     rgbImg = cv2.resize(rgbImg, (resizeRGBimW, resizeRGBimH), interpolation=cv2.INTER_LINEAR)
    #     cv2.imwrite(os.path.join(outputRGBFolder, 'im_{}.png'.format(i+1)), rgbImg)
    mask = cv2.imread(f'{opt.data_src}/{i:06d}-label.png', -1).reshape(imgH * imgW)
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

    #print('Origin:',origin, ' Target:', target, ' Up:', up)

    if plot3Dcam == True:
        centers.append(origin)
        lookups.append((target-origin))
        ups.append(up)
        ids.append(imgId)

    # if writeOutputCamFile == True:
    #     with open(outputCamFile, 'a') as cam:
    #         cam.write('{} {} {}\n'.format(origin[0], origin[1], origin[2]))
    #         cam.write('{} {} {}\n'.format(target[0], target[1], target[2]))
    #         cam.write('{} {} {}\n'.format(up[0], up[1], up[2]))

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

if buildVisualHull:
    axisLen = float(resolution-1) / 2.0
    verts = (verts - axisLen) / axisLen * 1.7
    mesh = trm.Trimesh(vertices = verts, vertex_normals = normals, faces = faces)
    pointcloud = trm.sample.sample_surface(mesh, opt.point_num)
    print('Export final mesh !')
    mesh.export(meshName)
    pointcloud.export(pointcloudName)

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
