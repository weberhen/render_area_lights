import os
import cv2
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout
from read_3light_prediction import read_3light_prediction
from create_mitsuba_command import create_mitsuba_command

root_folder = 'test_data'
filename = 'latlong.png'
layout = 'latlong_layout.txt'

pickle_filename = 'class.pkl'

obj_pos = [.5,-1.55,-1]
obj_ply = 'armadillo.ply'
obj_scale = '3'
scene = 'scene_ply'

posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
# posCenters*=100
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)

# read pano
texture = cv2.cvtColor(cv2.imread(os.path.join(root_folder, filename)), cv2.COLOR_BGR2RGB)
                    
# read layout
cor_id = read_cor_id(os.path.join(root_folder,layout), texture.shape)

# get 3d points from layout
rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], 1.65)
gen_polygon_from_layout(rp, posCenters*depths)

os.system(command)
os.system('mtsutil tonemap -m 100 scene_ply.exr')