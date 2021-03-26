import numpy as np
import cv2
from utils import rotation_matrix, corners_to_xyz, ensure_camera_inside_layout, gen_polygon_from_layout, xyz_from_depth, normalize, uvmap, read_cor_id
from rotlib import rotate
from envmap.projections import world2latlong


def warp_non_parametric(obj_pos, envmap_filename, layout):
    
    direction_corrector = [1,-1,1] 
        
    T=np.array(obj_pos)*direction_corrector
        
    texture = cv2.imread(envmap_filename)
    texture = cv2.resize(texture, (128,64))
    #######
    #
    ####### 
    yaw = np.deg2rad(0)
    pitch = np.deg2rad(0)
    roll = np.deg2rad(0)
    
    rot_mat = rotation_matrix(yaw, pitch, roll)
    
    cor_id = read_cor_id(layout, texture.shape)
          
    #######
    # Creating the new layout based on the translation and rotation we 
    # want. The last parameter define a scale we want to apply to the 
    # points, so we can more easily place the camera inside the scene.
    #######
    scale = 1
    # T*=scale/2
    
    # TODO remove debug
    # rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], rot_mat, np.array([0,0,0]), 1)
    # gen_polygon_from_layout(rp, 'scene_debug.ply')
    # end debug
    
    rp = corners_to_xyz(cor_id, texture.shape[0], texture.shape[1], rot_mat, T, scale)
    
    gen_polygon_from_layout(rp)
    
    ensure_camera_inside_layout(np.multiply(obj_pos,scale), rp)
    
    xyzp = xyz_from_depth(texture.shape[1], texture.shape[0], 'depth.png')
    
    ########
    ######## XYZ' -> XYZ ########
    ########
    xyzp = np.array(xyzp)
    xyz = rotate(xyzp - T, 'DCM', np.linalg.inv(rot_mat))
    
    ########
    ######## XYZ -> S ########
    ######## 
    xyz = normalize(xyz)
    
    uv_S = np.array(world2latlong(xyz[:,0], xyz[:,1], xyz[:,2]))
    
    uv_S = uv_S.reshape(2,texture.shape[0],texture.shape[1])
    assert not np.isnan(np.sum(uv_S))
    
    image = uvmap(uv_S, texture)
    
    cv2.imwrite('non_parametric.png', image)


if __name__ == "__main__":
    obj_pos = np.array([0,0,.5]).astype(np.float64)
    envmap_filename = '/Users/henriqueweber/codes/render/data/latlong_test.png'
    layout = '/Users/henriqueweber/codes/render/data/latlong_test.txt'
    warp_non_parametric(obj_pos, envmap_filename, layout)