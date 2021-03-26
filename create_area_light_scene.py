import os
import cv2
import numpy as np
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout, xyz_from_depth, normalize, gen_colored_ply
from envmap.projections import world2latlong
from gen_mitsuba_xml import gen_mitsuba_xml

root_folder = 'test_data'
filename = 'latlong.png'
layout = 'latlong_layout.txt'

# read pano
texture = cv2.cvtColor(cv2.imread(os.path.join(root_folder, filename)), cv2.COLOR_BGR2RGB)
                    
# read layout
cor_id = read_cor_id(os.path.join(root_folder,layout), texture.shape)

# get 3d points from layout
rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], 100)
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

os.system('mitsuba -Dobjx=0 -Dobjy=-.23 -Dobjz=4 area_light_scene.xml')
os.system("mtsutil tonemap -m 1 area_light_scene.exr")