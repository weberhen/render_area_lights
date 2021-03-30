import utilsParams
import torch
import os
import numpy as np
from utils import read_3light_prediction, create_mitsuba_command, corners_to_xyz, read_cor_id, gen_polygon_from_layout, xyz_from_depth, normalize, gen_mitsuba_xml
from envmap.projections import world2latlong
import hdrio
import cv2
import configInfos

configInfos.device = torch.device('cpu')
obj_positions = [[i/100,0,.5] for i in range(-50,51,50)]
for i, obj_pos in enumerate(obj_positions):
    # obj_pos = np.array([-.5,0,.5])
    obj_pos = np.array(obj_pos)
    pickle_filename = '/Users/henriqueweber/datasets/LavalIndoor/output_deepparametric/test/predictedParams/AG8A1520-others-200-1.37354-1.02082.pkl'
    filename = '/Users/henriqueweber/codes/ldrestimator/test_data/crops/latlong.png'
    layout = '/Users/henriqueweber/codes/ldrestimator/test_data/latlong_layout.txt'

    ######
    # generate parametric pano
    ######
    posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
    posCenters*=0
    posCenters+=np.array([[0,  1.46136999, 2.51528728],[0,  1.46136999, 2.51528728],[0,  1.46136999, 2.51528728]])
    posCenters*=np.array([1,1,-1])
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
        torch.from_numpy(ambients*0), 
        size=(300, 600, 3), 
        convertFromXYZ=True)

    iblN = ibl.cpu().numpy()
    # hdrio.imsave("envmap.exr", iblN.astype('float16'))

    # command = create_mitsuba_command(obj_pos, posCenters, radius, intensities, ambients, depths, 'scene_3light_with_ALS')
    # print(command)
    # os.system(command)
    # os.system('mtsutil tonemap -m 10 -o scene_3light_with_ALS.png scene_3light_with_ALS.exr')


    # generate non-parametric pano
    # read pano
    texture = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    texture = cv2.resize(texture,(512,256))
    # read layout
    cor_id = read_cor_id(layout, texture.shape)

    # get 3d points from layout
    rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([0,0,0]), 1)
    gen_polygon_from_layout(rp)
    xyzp = xyz_from_depth(texture.shape[1], texture.shape[0], 'depth.png')
    xyz = np.array(xyzp)
    xyzn = normalize(xyz)

    # get correspondency between 3d and pixel colors
    uv_S = np.array(world2latlong(xyzn[:,0], xyzn[:,1], xyzn[:,2]))
    assert not np.isnan(np.sum(uv_S))

    # create xml with area light sources
    gen_mitsuba_xml(xyz, obj_pos, uv_S, texture, 'area_light_scene.xml')

    command = 'mitsuba -o non_parametric.exr area_light_scene.xml'
    print(command)
    os.system(command)

    warped_texture = hdrio.imread('non_parametric.exr')
    # for i in range(3):
        # warped_texture[:,:,i]/=(warped_texture[:,:,i].max()/ambients[i])

    # combine them
    final_prediction = warped_texture[:,:,0:3] + iblN
    hdrio.imwrite(final_prediction, "combined_envmap.exr")

    # render
    command = 'mitsuba -o final_render.exr '+' -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' -Denvmap=combined_envmap.exr render_scene.xml'
    print(command)
    os.system(command)
    os.system("mtsutil tonemap -m .1 -o final_render_"+str(i).zfill(3)+"_"+str(obj_pos[0])+"_"+str(obj_pos[1])+"_"+str(obj_pos[2])+".png final_render.exr")