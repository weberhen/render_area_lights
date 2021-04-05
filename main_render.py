import utilsParams
import torch
import os
import numpy as np
from utils import read_3light_prediction, corners_to_xyz, read_cor_id, gen_polygon_from_layout, xyz_from_depth, normalize, gen_mitsuba_xml, compose
from envmap.projections import world2latlong
import hdrio
import imageio
import configInfos
import cv2

inputImNames = [
        'AG8A1520-others-200-1.37354-1.02082.png',
        'AG8A4786-others-160-1.63987-1.10975.png',
        '9C4A7811-others-120-1.44844-1.01319.png',
        'AG8A8710-others-00-1.73133-1.00330.png',
        '9C4A8295-others-00-1.73210-1.06243.png',
        '9C4A0632-others-00-1.84997-0.94982.png',
        'AG8A9772-others-280-1.66504-1.06817.png',
        'AG8A9746-others-160-1.61411-1.05738.png',
        'AG8A9666-others-280-1.61196-1.10387.png',
        'AG8A9171-others-200-1.72190-0.87783.png',
        # 'AG8A9100-others-280-2.07701-1.08082.png',
        # 'AG8A8687-others-00-2.16029-1.13881.png',
    ]

for inputImName in sorted(inputImNames):
    background_filename = '/Users/henriqueweber/liszt/LavalIndoor/ldrInputs/test/'+inputImName
    pickle_filename = '/Users/henriqueweber/liszt/LavalIndoor/output_deepparametric/test/predictedParams/'+inputImName[:-3]+'pkl'
    filename = '/Users/henriqueweber/liszt/LavalIndoor/output_ldrestimator/test/'+inputImName[:-3]+'_texture.png'
    layout = '/Users/henriqueweber/liszt/LavalIndoor/pred_layout/test/'+inputImName[:-4]+'_layout.txt'

    configInfos.device = torch.device('cpu')
    # obj_positions = [[0,-.1,i/50+2] for i in range(-50,150+1,5)]
    obj_positions = [[0,-.5,3]]
    imInput = imageio.imread(background_filename).astype('float32')[:,:,0:3] / 255.
    for i, obj_pos in enumerate(obj_positions):
        # obj_pos = np.array([-.5,0,.5])
        obj_pos = np.array(obj_pos)
        
        ######
        # generate parametric pano
        ######
        posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
        # posCenters*=0
        # posCenters+=np.array([[1,1,0],[1,1,0],[1,1,0]])
        posCenters*=np.array([-1,1,-1])
        posCenters_torch, depths_torch, radius_torch = utilsParams.translateParamsTorch(
            torch.from_numpy(posCenters), 
            torch.from_numpy(depths), 
            torch.from_numpy(radius), 
            torch.Tensor(np.array([0,0,0])))

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
        texture = cv2.resize(texture,(128,64))
        # read layout
        cor_id = read_cor_id(layout, texture.shape)

        # get 3d points from layout
        rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([0,0,0]), 10)
        gen_polygon_from_layout(rp)
        xyzp = xyz_from_depth(texture.shape[1], texture.shape[0], 'depth.png')
        xyz = np.array(xyzp)
        xyzn = normalize(xyz)

        # get correspondency between 3d and pixel colors
        uv_S = np.array(world2latlong(xyzn[:,0], xyzn[:,1], xyzn[:,2]))
        assert not np.isnan(np.sum(uv_S))

        # create xml with area light sources
        gen_mitsuba_xml(xyz, np.array([0,0,0]), uv_S, texture, 'area_light_scene.xml')

        command = 'mitsuba -o non_parametric.exr area_light_scene.xml'
        print(command)
        os.system(command)

        warped_texture = hdrio.imread('non_parametric.exr')
        # for j in range(3):
        #     warped_texture[:,:,j]/=(warped_texture[:,:,j].max()/ambients[j])

        warped_texture[:,:,0:3] = warped_texture[:,:,0:3]**2/2
        # combine them
        final_prediction = warped_texture[:,:,0:3] + iblN
        # final_prediction = iblN
        hdrio.imwrite(final_prediction, "combined_envmap.exr")

        # render
        command = 'mitsuba -o final_render.exr '+' -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' -Denvmap=combined_envmap.exr render_scene.xml'
        print(command)
        os.system(command)
        # os.system("mtsutil tonemap -m 1 -o final_render_"+str(i).zfill(3)+"_"+str(obj_pos[0])+"_"+str(obj_pos[1])+"_"+str(obj_pos[2])+".png final_render.exr")
        
        compose(imInput, 'final_render.exr', inputImName+"_final_render_"+str(i).zfill(3)+"_"+str(obj_pos[0])+"_"+str(obj_pos[1])+"_"+str(obj_pos[2])+".png", global_modifier_factor=.1)
os.system('rm montage* && montage -tile 5x2 -geometry +2+2 *.png montage.png')