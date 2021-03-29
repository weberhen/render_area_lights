import os
import numpy as np
from utils import read_3light_prediction, create_mitsuba_command
import utilsParams
import torch
import hdrio
import configInfos

configInfos.device = torch.device('cpu')

pickle_filename = '/Users/henriqueweber/datasets/LavalIndoor/output_deepparametric/test/predictedParams/AG8A1520-others-200-1.37354-1.02082.pkl'
obj_pos = np.array([0,-.3,2])
obj_ply = 'armadillo.ply'
obj_scale = '3'
scene = 'scene_3light_with_ALS'

posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
posCenters*=0
posCenters+=[[2,1,0],[2,1,0],[2,1,0]]
ambients*=0
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
print(command)
os.system(command)
os.system('mtsutil tonemap -m 10 -o scene_3light_with_ALS.png scene_3light_with_ALS.exr')

posCenters_torch, depths_torch, radius_torch = utilsParams.translateParamsTorch(
    torch.from_numpy(posCenters), 
    torch.from_numpy(depths), 
    torch.from_numpy(radius), 
    torch.Tensor(obj_pos*-1))

ibl = utilsParams.convertSGToIBLTorch(
    posCenters_torch,
    depths_torch,
    radius_torch,
    torch.from_numpy(intensities), 
    torch.from_numpy(ambients), 
    size=(300, 600, 3), 
    convertFromXYZ=True)

iblN = ibl.cpu().numpy()
hdrio.imsave("envmap.exr", iblN.astype('float16'))

scene = '-Denvmap=envmap.exr -Dscale=.1 scene_3light_with_pano' #TODO remove this ugly hack
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
print(command)
os.system(command)
os.system('mtsutil tonemap -m .1 -o scene_3light_with_pano.png scene_3light_with_pano.exr')