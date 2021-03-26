import os
import cv2
import numpy as np
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout, xyz_from_depth, normalize, gen_colored_ply
from envmap.projections import world2latlong
from gen_mitsuba_xml import gen_mitsuba_xml

root_folder = '/Users/henriqueweber/datasets/LavalIndoor/'
filename = 'ldrOutputs/test/AG8A1520-others-200-1.37354-1.02082.png'
layout = 'gt_layout/test/AG8A1520-others-200-1.37354-1.02082.txt'
obj_pos = [.5,-5.15,-5]
obj_ply = 'sphere.ply'
obj_scale = '3'
scene = 'scene_ply'

# read pano
texture = cv2.cvtColor(cv2.imread(os.path.join(root_folder, filename)), cv2.COLOR_BGR2RGB)
texture = cv2.resize(texture, (1024,512))
# read layout
cor_id = read_cor_id(os.path.join(root_folder,layout), texture.shape)

# get 3d points from layout
rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], 4)
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

os.system('mitsuba -Dobjscale='+str(obj_scale)+' -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' area_light_scene.xml')
os.system("mtsutil tonemap -m 1 -o final_renders/background.png area_light_scene.exr")