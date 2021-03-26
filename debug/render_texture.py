import os
import cv2
import numpy as np
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout, xyz_from_depth, normalize, gen_colored_ply
from envmap.projections import world2latlong
from gen_mitsuba_xml import gen_mitsuba_xml
from warp_non_parametric import warp_non_parametric

filename = '/Users/henriqueweber/datasets/LavalIndoor/ldrOutputs/test/AG8A1520-others-200-1.37354-1.02082.png'
layout = '/Users/henriqueweber/datasets/LavalIndoor/gt_layout/test/AG8A1520-others-200-1.37354-1.02082.txt'
obj_pos = np.array([0,-.3,2])
obj_ply = 'armadillo.ply'
obj_scale = '3'
scene = 'scene_pano'

# read pano
texture = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
texture = cv2.resize(texture,(128,64))
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

# save ply file to check result
gen_colored_ply(xyz, uv_S, texture, 'colored_scene.ply')

# create xml with area light sources
gen_mitsuba_xml(xyz, uv_S, texture, 'area_light_scene.xml')

command = 'mitsuba -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' area_light_scene.xml'
print(command)
os.system(command)
os.system("mtsutil tonemap -m 1 area_light_scene.exr")

#now time to warp the panorama and render with a pano instead of area lights
warp_non_parametric(obj_pos, filename, layout)

command = 'mitsuba -Denvmap=non_parametric.png -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' debug/scene_pano.xml'
print(command)
os.system(command)
os.system('mtsutil tonemap -m .1 -o debug/scene_pano_warped.png debug/scene_pano.exr')