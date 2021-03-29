import os
import numpy as np
from read_3light_prediction import read_3light_prediction
from create_mitsuba_command import create_mitsuba_command
import utilsParams
import torch
import hdrio
import configInfos

configInfos.device = torch.device('cpu')

pickle_filename = '/Users/henriqueweber/datasets/LavalIndoor/output_deepparametric/test/predictedParams/AG8A1520-others-200-1.37354-1.02082.pkl'
obj_pos = np.array([0,-.3,2])
obj_ply = 'debug_final/armadillo.ply'
obj_scale = '3'
scene = 'debug_final/scene_3light'

posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
posCenters*=0
posCenters+=[[2,1,0],[2,1,0],[2,1,0]]
ambients*=0
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
print(command)
os.system(command)
os.system('mtsutil tonemap -m 10 -o debug_final/scene_3light.png debug_final/scene_3light.exr')

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
hdrio.imsave("debug_final/envmap.exr", iblN.astype('float16'))

scene = '-Denvmap=debug/envmap.exr debug_final/scene_pano' #TODO remove this ugly hack
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
print(command)
os.system(command)
os.system('mtsutil tonemap -m .1 -o debug_final/scene_pano.png debug_final/scene_pano.exr')