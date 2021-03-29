import os
import cv2
import numpy as np
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout, xyz_from_depth, normalize, gen_colored_ply, gen_mitsuba_xml
from envmap.projections import world2latlong


filename = '/Users/henriqueweber/codes/ldrestimator/test_data/crops/latlong.png'
layout = '/Users/henriqueweber/codes/ldrestimator/test_data/latlong_layout.txt'
obj_pos = np.array([0,-.3,2])

scale = 2

# read pano
texture = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
texture = cv2.resize(texture,(512,256))
# read layout
cor_id = read_cor_id(layout, texture.shape)

# get 3d points from layout
rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], np.array([[1,0,0],[0,1,0],[0,0,1]]), np.array([0,0,0]), scale)
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
gen_mitsuba_xml(xyz, obj_pos, uv_S, texture, 'area_light_scene.xml')

# command = 'mitsuba -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' area_light_scene.xml'
command = 'mitsuba area_light_scene.xml'
print(command)
os.system(command)
os.system("mtsutil tonemap -m 1 -o non_parametric.png area_light_scene.exr")

#now time to warp the panorama and render with a pano instead of area lights
# warp_non_parametric(obj_pos, filename, layout)

command = 'mitsuba -Denvmap=non_parametric.png -Dscale='+str(scale*.1)+' -Dobjx='+str(obj_pos[0]*scale)+' -Dobjy='+str(obj_pos[1]*scale)+' -Dobjz='+str(obj_pos[2]*scale)+' scene_pano.xml'
print(command)
os.system(command)
os.system("mtsutil tonemap -m 1 -o render_texture_warped_pano"+str(obj_pos[0])+"_"+str(obj_pos[1])+"_"+str(obj_pos[2])+".png scene_pano.exr")