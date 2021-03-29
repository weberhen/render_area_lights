import json
import os
import torch
import numpy as np
import cv2
from scipy.spatial import Delaunay
from envmap import EnvironmentMap
import Imath
import OpenEXR
from rotlib import rotx, roty, rotz, rotate


def read_cor_id(layout, shape):
    with open(layout) as f:
        cor_id = np.array([line.split() for line in f], np.float32)
    # last line of the text file is the size of the pano, here we make sure we 
    # are using the right dimensions to create the 3d 
    assert int(cor_id[-1,0]) == shape[1]
    assert int(cor_id[-1,1]) == shape[0]
    
    return cor_id[:-1,:]
    
    
def montage(config, output_filename):
    # move texture prediction
    os.system('cp '+config['non_param_png'][:-13]+'.png'+' .')
    os.system('mtsutil tonemap *exr')
    command = 'montage '+output_filename+' combined.png parametric.png depth.png montage.png'
    os.system(command)
    print('montage done')

def args_to_string(args):
    '''
    Convert many arguments into a single string. 
    For example, if kwargs == dict_items([('o', 'output.exr'), ('q', '')]), 
    the output will be "-o output.exr -q".
    A variable without argument must be passed with '' as value, like 'q' 
    above.
    '''
    args_as_string = []
    for key,value in args.items():
        if key[:2] == '-D':
            args_as_string.append(key+'='+value)
        else:
            args_as_string.append(" ".join([key,value]))
    return " ".join(args_as_string)


def read_json_file(filename):
    with open(filename, 'r') as j:
        json_data = json.load(j)
    return json_data


def tonemap(filename):
    command = 'mtsutil tonemap -a -o '+filename[:-4]+'.png '+filename
    os.system(command)
    

def np_coorx2theta(coorx, coorW=1024):
    '''
    azimuth
    '''
    
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi

def np_coory2phi(coory, coorH=512):
    '''
    elevation
    '''
    
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xz(coor, y=50, coorW=1024, coorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    
    coor = np.array(coor)
    theta = np_coorx2theta(coor[:, 0], coorW)
    phi = np_coory2phi(coor[:, 1], coorH)
    
    c = y / np.tan(phi)
    x = - c * np.sin(theta)
    z = c * np.cos(theta)
    
    return np.hstack([x[:, None], z[:, None]])


def corners_to_xyz(cor_id, H, W, rot_mat, T, scale=1.0):
    '''
    Convert cor_id to 3d xyz
    '''
    camera_height = 1.6
    
    floor_y = -camera_height
    floor_xz = np_coor2xz(cor_id[1::2], floor_y, W, H)
    c = np.sqrt((floor_xz**2).sum(1))
    phi = np_coory2phi(cor_id[0::2, 1], H)
    ceil_y = (c * np.tan(phi)).mean()
    
    floor_xz*=scale
    floor_y*=scale
    ceil_y*=scale
    
    # You apply T^-1 to the original layout from S
    rp = [] # rotated points
    for xz in floor_xz:
        point = np.array([xz[0], floor_y, xz[1]])
        rp.append(rotate(point, 'DCM', rot_mat)+T)
        point = np.array([xz[0], ceil_y, xz[1]])
        rp.append(rotate(point, 'DCM', rot_mat)+T)
        
    return np.array(rp)

    
# def corners_to_xyz(cor_id, H, W, scale):
#     '''
#     Convert cor_id to 3d xyz
#     '''
#     camera_height = 1.6
    
#     floor_y = -camera_height
#     floor_xz = np_coor2xz(cor_id[1::2], floor_y, W, H)
#     c = np.sqrt((floor_xz**2).sum(1))
#     phi = np_coory2phi(cor_id[0::2, 1], H)
#     ceil_y = (c * np.tan(phi)).mean()
    
#     floor_xz*=scale
#     floor_y*=scale
#     ceil_y*=scale
    
#     # You apply T^-1 to the original layout from S
#     rp = [] # rotated points
#     for xz in floor_xz:
#         point = np.array([xz[0], floor_y, xz[1]])
#         rp.append(point)
#         point = np.array([xz[0], ceil_y, xz[1]])
#         rp.append(point)
        
#     return np.array(rp)


def uvmap(uv_s, texture):
    img = np.zeros(texture.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = texture[int(uv_s[1][i][j]*img.shape[0])][img.shape[1]-1-int(uv_s[0][i][j]*img.shape[1])]
    return img


def generateTriplePoints(n_points): 
    """
    generate groups of 3 points so we can have one plane equation per wall, one for ceiling and one for floor
    """
    group_points = []
    for i in range(0,int(n_points/2),1):
        if i%2==0:
            group_points.append([4, i*2+0, i*2+2, i*2+3, i*2+1])
        else:
            group_points.append([4, i+4, i+3, i+1, i+2])
    if n_points%4==0:
        group_points[-1][3] = 0
        group_points[-1][4] = 1
    else:
        group_points[-1][2] = 0
        group_points[-1][3] = 1
        
        
    return np.array(group_points)


def load_hdr_multichannel(path, color=True, normal=False, depth=False, blender_format=False):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_img_openexr = OpenEXR.InputFile(path)
    rgb_img = rgb_img_openexr.header()['dataWindow']
    size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

    if  blender_format:
        colors = ['R', 'G', 'B']
    else:
        colors = ['color.R', 'color.G', 'color.B']
    if color:
        # color
        redstr = rgb_img_openexr.channel(colors[0], pt)
        red = np.fromstring(redstr, dtype=np.float32)
        red.shape = (size_img[1], size_img[0])

        greenstr = rgb_img_openexr.channel(colors[1], pt)
        green = np.fromstring(greenstr, dtype=np.float32)
        green.shape = (size_img[1], size_img[0])

        bluestr = rgb_img_openexr.channel(colors[2], pt)
        blue = np.fromstring(bluestr, dtype=np.float32)
        blue.shape = (size_img[1], size_img[0])

        color = np.dstack((red, green, blue))
    else:
        color = None

    if normal:
        # normal
        normal_x_str = rgb_img_openexr.channel('normal.R', pt)
        normal_x = np.fromstring(normal_x_str, dtype=np.float32)
        normal_x.shape = (size_img[1], size_img[0])

        normal_y_str = rgb_img_openexr.channel('normal.G', pt)
        normal_y = np.fromstring(normal_y_str, dtype=np.float32)
        normal_y.shape = (size_img[1], size_img[0])

        normal_z_str = rgb_img_openexr.channel('normal.B', pt)
        normal_z = np.fromstring(normal_z_str, dtype=np.float32)
        normal_z.shape = (size_img[1], size_img[0])

        normal = np.dstack((normal_x, normal_y, normal_z))
    else:
        normal = None

    if depth:
        # depth
        depth_str = rgb_img_openexr.channel('distance.Y', pt)
        depth = np.fromstring(depth_str, dtype=np.float32)
        depth.shape = (size_img[1], size_img[0])
    else:
        depth = None

    return color, normal, depth


def normalize(v):
    if np.ndim(v)!=1:
        norm = np.linalg.norm(v,axis=1)
    else:
        norm = np.linalg.norm(v)
    return v / norm[..., np.newaxis]


def xyz_from_depth(width, height, filename):
    os.system('mitsuba scene.xml')
    _, _, depth = load_hdr_multichannel('scene.exr', color=True, depth=True)
    # depth = np.flip(depth,1) # flip horizontically
    # depth = shift_image(depth, int(depth.shape[0]/2),0)
    # depth = np.roll(depth, depth.shape[0], axis=1)
    depth = cv2.resize(depth,(width, height))
    cv2.imwrite(filename, depth/depth.max()*255)
    
    xyz_surface_S = np.array(EnvironmentMap(depth.shape[0],'latlong').worldCoordinates())
    x = xyz_surface_S[0]*depth
    y = xyz_surface_S[1]*depth
    z = xyz_surface_S[2]*depth
    for i in np.argwhere(np.isnan(x)):
        x[i[0]][i[1]] = x[i[0]][i[1]+1]
    for i in np.argwhere(np.isnan(y)):
        y[i[0]][i[1]] = y[i[0]][i[1]+1]
    for i in np.argwhere(np.isnan(z)):
        z[i[0]][i[1]] = z[i[0]][i[1]+1]
    assert not np.isnan(np.sum(x))
    assert not np.isnan(np.sum(y))
    assert not np.isnan(np.sum(z))
    x=x.reshape(-1)
    y=y.reshape(-1)
    z=z.reshape(-1)
    
    return np.array([x,y,z]).transpose() 
    

def gen_polygon_from_layout(rp, extra_points=[], filename='scene.ply'):
    """
    rp is an array with the edges of the layout in 3d space
    """
    ceil = np.delete(rp, range(1, rp.shape[0],2),axis=0)
    ceil = np.delete(ceil,1,axis=1)
    tri = Delaunay(ceil)
    points = tri.simplices
    points_ceil = points*2+1
    new_points_ceil = np.array([points_ceil[:,1], points_ceil[:,2], points_ceil[:,0]]).transpose()
    points_floor = points*2
    new_points_floor = np.array([points_floor[:,2], points_floor[:,1], points_floor[:,0]]).transpose()
    size_pol = np.ones((new_points_ceil.shape[0],1))+2
    new_points_ceil = np.append(size_pol, new_points_ceil, axis=1)
    new_points_floor = np.append(size_pol, new_points_floor, axis=1)
    points_wall = generateTriplePoints(len(rp))
    n_faces = len(new_points_ceil) + len(new_points_floor) + len(points_wall)
    with open(filename, "w") as a_file:
        a_file.write("ply\nformat ascii 1.0\nelement vertex "+str(len(rp)+len(extra_points))+"\nproperty float x\nproperty float y\nproperty float z\nelement face "+str(n_faces)+"\nproperty list uchar int vertex_index\nend_header\n")
    with open(filename, "ab") as a_file:
        np.savetxt(a_file, rp, fmt="%.3f")
        if extra_points != []:
            np.savetxt(a_file, extra_points, fmt="%d")
        np.savetxt(a_file, new_points_ceil, fmt="%d")
        np.savetxt(a_file, new_points_floor, fmt="%d")
        np.savetxt(a_file, points_wall, fmt="%d")
        
        
        
def gen_colored_ply(xyz, uv_S, texture, filename):
    colors = np.zeros(xyz.shape)
    for i in range(uv_S.shape[1]):
        colors[i] = texture[int(uv_S[1][i]*texture.shape[0])][int(uv_S[0][i]*texture.shape[1])]
    xyz_colors = np.concatenate((xyz, colors), axis=1)
    with open(filename, "w") as a_file:
        a_file.write("ply\nformat ascii 1.0\nelement vertex "+str(xyz_colors.shape[0])+"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(filename, "ab") as a_file:
        np.savetxt(a_file, xyz_colors, fmt="%.3f %.3f %.3f %d %d %d")


def ensure_camera_inside_layout(obj_pos, rp):
    #######
    # Making sure we do not place the camera outside of the valid perimeter of
    # the scene.
    #######
    
    print("Limits of the room are (x): ", rp[:,0].min(), rp[:,0].max())
    print("Limits of the room are (y): ", rp[:,1].min(), rp[:,1].max())
    print("Limits of the room are (z): ", rp[:,2].min(), rp[:,2].max())
    
    assert (obj_pos[0] < (rp[:,0].max())).all()
    assert (obj_pos[0] > (rp[:,0].min())).all()
    assert (obj_pos[1] < (rp[:,1].max())).all()
    assert (obj_pos[1] > (rp[:,1].min())).all()
    assert (obj_pos[2] < (rp[:,2].max())).all()
    assert (obj_pos[2] > (rp[:,2].min())).all()
    

def rotation_matrix(azimuth, elevation, roll=0):
    """Returns a camera rotation matrix.
    :azimuth: left (negative) to right (positive) [rad]
    :elevation: upward (negative) to downward (positive) [rad]
    :roll: counter-clockwise (negative) to clockwise (positive) [rad]"""
    return rotz(roll).dot(rotx(elevation)).dot(roty(-azimuth))
    
    
def xyz2sphericalTorch(xyz):
    # Each row is a point
    # x, y, and z coordinates are respectively indices 0, 1, and 2
    # on axis 1
    norm = torch.norm(xyz, 2, dim=1)
    xyz = xyz / norm.unsqueeze(0).t()
    elev = torch.asin(xyz[:, 1] * 0.9999)     # epsilon to ensure that we do not compute asin(±1.00001) because of rounding errors
    azim = torch.atan2(xyz[:, 0], xyz[:, 2])

    return torch.stack((elev, azim), dim=1), norm
    
    
def translateParamsTorch(posCenters, depths, radius, translation):
    # We assume posCenters to be XYZ
    # Translation should be a (x, y, z) vector
    # The up-vector is y, positive z goes behind the camera, positive x to the left

    # Apply the translation
    # Careful : we translate the _lights_ so we should go in the opposite
    # direction than the direction to the crop
    newPosCenters = posCenters * depths.view((posCenters.shape[0], 1))
    newPosCenters = newPosCenters + translation

    # We convert the coordinates back into spherical
    posCentersSpherical, newDepths = xyz2sphericalTorch(newPosCenters)
    newPosCenters = newPosCenters / torch.norm(newPosCenters, dim=1, keepdim=True)

    # We adjust the angular size of each light
    # We assume that the area lights normal are always towards the virtual
    # center of projection, even if we moved in the scene
    # (as if the lights were actually spheres with the same angular area
    # from any reference point in the panorama)
    
    # To do that, we first convert the angular area from the original point
    # to an actual area using the distance
    # IMPORTANT : this assumes `radius` to be in radians, which IS NOT HOW
    # THEY ARE SAVED IN THE PARAMETERS FILE
    radiusMeters = 2 * depths * torch.tan(radius/2)
    
    # From there, we do the inverse computation, but now with the depth
    # at the new virtual viewpoint
    # TODO : there is one *2 that could be removed from both formulaes
    # currently keeping it just to be more clear
    newRadius = 2 * torch.atan(radiusMeters / 2 / newDepths)

    #newPosCenters[:, 1].add_(2*np.pi)
    #newPosCenters[:, 1] = torch.fmod(newPosCenters[:, 1], 2*np.pi)

    if torch.any(torch.isnan(newPosCenters)):
        breakpoint()

    return newPosCenters, newDepths, newRadius


def spherical2xyzNumpy(elev, azim, norm):
    # Each row in elev, azim, and norm is a point
    cosElev = np.cos(elev)
    xn = cosElev * np.sin(azim)              # Positive to the left
    yn = np.sin(elev)                             # up-vector
    zn = cosElev * np.cos(azim)              # Camera direction : -z
    xyzn = np.stack((xn, yn, zn), axis=-1) * np.atleast_2d(norm).T
    return xyzn


g_keepXYZCoordsTorch = {}
def getXYZlatlongTorch(size, dev):
    global g_keepXYZCoordsTorch
    if not (size, dev) in g_keepXYZCoordsTorch:
        ycoords, xcoords = np.mgrid[0:size[0], 0:size[1]]
        ycoords = np.pi/2 - (ycoords * (np.pi / size[0]))
        xcoords = xcoords * (2 * np.pi / size[1])
        xyzn = spherical2xyzNumpy(ycoords, xcoords, norm=1.0)

        g_keepXYZCoordsTorch[(size, dev)] = torch.from_numpy(xyzn).to(dev, dtype=torch.float32, non_blocking=True)

    return g_keepXYZCoordsTorch[(size, dev)]


def spherical2xyzTorch(elev, azim, norm=None):
    # Each row in elev, azim, and norm is a point
    cosElev = elev.cos()
    xn = cosElev * azim.sin()              # Positive to the left
    yn = elev.sin()                             # up-vector
    zn = cosElev * azim.cos()              # Camera direction : -z

    if norm is None:
        # Assume unit norm
        xyzn = torch.stack((xn, yn, zn), dim=1)
    else:
        xyzn = torch.stack((xn, yn, zn), dim=1) * norm.unsqueeze(0).t()
    return xyzn


