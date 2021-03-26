import os
import cv2
import hdrio
from utils import read_cor_id, corners_to_xyz, gen_polygon_from_layout
from read_3light_prediction import read_3light_prediction
from create_mitsuba_command import create_mitsuba_command

root_folder = '/Users/henriqueweber/datasets/LavalIndoor/'
filename = 'ldrOutputs/test/AG8A1520-others-200-1.37354-1.02082.png'
layout = 'gt_layout/test/AG8A1520-others-200-1.37354-1.02082.txt'
pickle_filename = '/Users/henriqueweber/datasets/LavalIndoor/output_deepparametric/test/predictedParams/AG8A1520-others-200-1.37354-1.02082.pkl'

obj_pos = [.5,-5.15,-8]
obj_ply = 'armadillo.ply'
obj_scale = '3'
scene = 'scene_ply'

posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)
# posCenters*=100
command = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
scene = 'scene_ply_empty'
command_scene_empty = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)
scene = 'scene_ply_object_only'
command_scene_object_only = create_mitsuba_command(obj_pos, obj_ply, obj_scale, posCenters, radius, intensities, ambients, depths, scene)

# read pano
texture = cv2.cvtColor(cv2.imread(os.path.join(root_folder, filename)), cv2.COLOR_BGR2RGB)
texture = cv2.resize(texture, (128,64))
# read layout
cor_id = read_cor_id(os.path.join(root_folder,layout), texture.shape)

# get 3d points from layout
rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], 4)
gen_polygon_from_layout(rp, posCenters*depths)

os.system(command)
os.system('mtsutil tonemap -m 10 -o final_renders/render_with_obj.png scene_ply.exr')
os.system(command_scene_empty)
os.system('mtsutil tonemap -m 10 -o final_renders/render_without_obj.png scene_ply_empty.exr')
os.system(command_scene_object_only)
imRender = hdrio.imread('scene_ply_object_only.exr').astype('float32')
alphaRender = imRender[..., 3:4]
cv2.imwrite('final_renders/mask.png', alphaRender)