
import numpy as np
import torch

from matplotlib import pyplot as plt

import configInfos

#################################################
# Numpy functions
# These do not allow gradient to be propagated
# They are used in the loader, since this can be
# processed on the CPU independently
#################################################

def pixelToElevAndAzim(pixpos, size):
    elevs = np.pi/2 - (pixpos[:, 0] * (np.pi / size[0]))
    azims = pixpos[:, 1] * (2 * np.pi / size[1])
    return elevs, azims

def spherical2xyzNumpy(elev, azim, norm):
    # Each row in elev, azim, and norm is a point
    cosElev = np.cos(elev)
    xn = cosElev * np.sin(azim)              # Positive to the left
    yn = np.sin(elev)                             # up-vector
    zn = cosElev * np.cos(azim)              # Camera direction : -z
    xyzn = np.stack((xn, yn, zn), axis=-1) * np.atleast_2d(norm).T
    return xyzn

def xyz2sphericalNumpy(xyz):
    # Each row is a point
    # x, y, and z coordinates are respectively indices 0, 1, and 2
    # on axis 1
    norm = np.linalg.norm(xyz, axis=1)
    xyz = xyz / np.atleast_2d(norm).T

    elev = np.arcsin(xyz[:, 1] * 0.9999)     # epsilon to ensure that we do not compute asin(±1.00001) because of rounding errors
    azim = np.arctan2(xyz[:, 0], xyz[:, 2])

    return np.stack((elev, azim), axis=-1), norm


def moveIBLwithParams(posCenters, depths, radius, translation, rotationAzimuth):
    # Translation should be specified as a (x, y, z) translation vector
    # Rotation should be specified as an angle in radians around the origin

    # First, we apply the rotation (with spherical coordinates)
    posCenters = posCenters.copy()      # Ensure we do not modify something important
    posCenters[:, 1] -= rotationAzimuth
    posCenters[:, 1] %= (2*np.pi)

    # Then, we convert to xyz (easier to translate than with spherical coordinates)
    xyz = spherical2xyzNumpy(posCenters[:,0], posCenters[:,1], norm=depths.copy())

    # Apply the translation
    # Careful : we translate the _lights_ so we should go in the opposite
    # direction than the direction to the crop
    xyz[:, 2] += translation

    # We convert the coordinates back into spherical
    newPosCenters, newDepths = xyz2sphericalNumpy(xyz)

    # We adjust the angular size of each light
    # We assume that the area lights normal are always towards the virtual
    # center of projection, even if we moved in the scene
    # (as if the lights were actually spheres with the same angular area
    # from any reference point in the panorama)
    
    # To do that, we first convert the angular area from the original point
    # to an actual area using the distance
    # IMPORTANT : this assumes `radius` to be in radians, which IS NOT HOW
    # THEY ARE SAVED IN THE PARAMETERS FILE
    radiusMeters = 2 * depths * np.tan(radius/2)
    
    # From there, we do the inverse computation, but now with the depth
    # at the new virtual viewpoint
    # TODO : there is one *2 that could be removed from both formulaes
    # currently keeping it just to be more clear
    newRadius = 2 * np.arctan(radiusMeters / 2 / newDepths)

    return newPosCenters, newDepths, newRadius


g_keepCoords = {}
g_keepDists = {}
def getDistances(size):
    global g_keepCoords, g_keepDists
    if not size in g_keepDists:
        ycoords, xcoords = np.mgrid[0:size[0], 0:size[1]]
        ycoords = np.pi/2 - (ycoords * (np.pi / size[0]))
        xcoords = xcoords * (2 * np.pi / size[1])
        xyzn = spherical2xyzNumpy(ycoords, xcoords, norm=1.0)
        dist = np.zeros(size[:2] + size[:2], dtype='float32')
        g_keepCoords[size] = xyzn

        for i in range(size[0]):
            for j in range(size[1]):
                dist[i, j] = np.matmul(xyzn, xyzn[i, j])

        g_keepDists[size] = dist

    return g_keepDists[size]

def convertToIBLNumpy(centers, depths, radius, colors, ambient, size=(128,256,3)):
    # Convert a parametrization to an IBL of the requested size
    # Numpy version (non differentiable)
    distMat = getDistances(size)
    ibl = np.ones(size, dtype='float32') * ambient
    zeros = np.zeros_like(ibl, dtype='float32')
    
    # TODO check !!!
    centerAnglePixElev = ((0.5 - centers[:, 0] / np.pi) * size[0]).astype('int32')
    centerAnglePixAzim = (centers[:, 1] / (2*np.pi) * size[1]).astype('int32')
    
    for lightIndex in range(centers.shape[0]):
        lightPixels = colors[lightIndex]

        lightPixels = lightPixels[np.newaxis, np.newaxis, :]
        
        # Distance measurement is just a lookup
        dist = distMat[centerAnglePixElev[lightIndex], centerAnglePixAzim[lightIndex]]
        # Caution the distance here is not exactly a cosine distance
        # The actual cosine distance is (1 - dist)
        # However, it just change the way we threshold it, and
        # avoiding these operations save time

        # When we extract the size of the light, we use a projection with a 1:1 ratio and a 60 degrees FoV
        # We ensure that the threshold always include at least one pixel
        thresholdAngle = min(np.cos(radius[lightIndex]), np.cos(np.pi/size[0]))

        cond = (dist > thresholdAngle)[..., np.newaxis]
        ibl += np.where(cond, lightPixels, zeros)
        
    return ibl

def renderNumpy(transportMatrix, ibl, output):
    for c in range(3):
        output[..., c] = np.dot(transportMatrix, ibl[..., c].reshape(-1)).reshape(output[..., c].shape)
    return output


#################################################
# Torch functions
# These are used for the network prediction, so
# to be able to give feedback to the network
#################################################

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

def xyz2sphericalTorch(xyz):
    # Each row is a point
    # x, y, and z coordinates are respectively indices 0, 1, and 2
    # on axis 1
    norm = torch.norm(xyz, 2, dim=1)
    xyz = xyz / norm.unsqueeze(0).t()
    elev = torch.asin(xyz[:, 1] * 0.9999)     # epsilon to ensure that we do not compute asin(±1.00001) because of rounding errors
    azim = torch.atan2(xyz[:, 0], xyz[:, 2])

    return torch.stack((elev, azim), dim=1), norm


def moveIBLwithParamsTorch(posCenters, depths, radius, translation, rotationAzimuth):
    # Translation should be specified as a (x, y, z) translation vector
    # Rotation should be specified as an angle in radians around the origin

    if configInfos.coordinateSystem == "xyz" and posCenters.shape[-1] == 3:
        # We have to convert everything in spherical coordinates first
        posCenters, _ = xyz2sphericalTorch(posCenters)
    else:
        posCenters = posCenters.clone()   # Ensure we do not modify something important

    # First, we apply the rotation (with spherical coordinates)
    posCenters = posCenters.clone()      # Ensure we do not modify something important
    posCenters[:, 1].sub_(rotationAzimuth)

    # Then, we convert to xyz (easier to translate than with spherical coordinates)
    xyz = spherical2xyzTorch(posCenters[:, 0], posCenters[:, 1], norm=depths.clone())

    # Apply the translation
    # Careful : we translate the _lights_ so we should go in the opposite
    # direction than the direction to the crop
    xyz[:, 2].add_(translation)

    # We convert the coordinates back into spherical
    newPosCenters, newDepths = xyz2sphericalTorch(xyz)

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

    newPosCenters[:, 1].add_(2*np.pi)
    newPosCenters[:, 1] = torch.fmod(newPosCenters[:, 1], 2*np.pi)

    if torch.any(torch.isnan(newPosCenters)):
        breakpoint()

    return newPosCenters, newDepths, newRadius




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


g_keepDistsTorch = {}
def getDistancesTorch(size, dev):
    global g_keepDistsTorch
    if not size in g_keepDists:
        numpyDist = getDistances(size)
        dist = torch.from_numpy(numpyDist)
        dist = dist.to(dev, dtype=torch.float32, non_blocking=True)
        g_keepDistsTorch[size] = dist

    return g_keepDistsTorch[size]


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


def convertToIBLTorch(centers, depths, radius, colors, ambient, size=(128,256,3), p=None, device=None, convertFromXYZ=False):

    if device is None:
        device = configInfos.device

    #distMat = getDistancesTorch(size, device)
    clampedRadius = torch.clamp(radius, np.pi/(size[0]+1), np.pi)

    if isinstance(ambient, int) and ambient == 0:
        ibl = torch.zeros(size).to(device, dtype=torch.float32)
    else:
        ibl = ambient.unsqueeze(0).unsqueeze(0).expand(size[0], size[1], 3)
    xyzmat = getXYZlatlongTorch(size, device)
    oneT = torch.ones((1,)).to(device, dtype=torch.float32)

    for lightIndex in range(centers.shape[0]):
        # Compute the distance matrix for this light
        if convertFromXYZ:
            xyzCenter = centers[lightIndex:lightIndex+1]
        else:
            xyzCenter = spherical2xyzTorch(centers[lightIndex:lightIndex+1, 0],
                                        centers[lightIndex:lightIndex+1, 1],
                                        norm=oneT)
        dist = torch.matmul(xyzmat, xyzCenter.t())

        # Adjust the radius
        # and binarize it (it is kept as float, but the values are either 0 or 1)
        dist.div_(torch.cos(clampedRadius[lightIndex]))
        dist[dist < 1] = 0
        distNormalized = torch.div(dist, dist.max())

        # Expand it with 3 channels
        distNormalized = distNormalized.view(size[:2]).unsqueeze(2).expand(size[0], size[1], 3)

        # Apply the color to create the light
        light = torch.mul(distNormalized, colors[lightIndex])

        # Add it to the IBL
        ibl = torch.add(ibl, light)

    return ibl

g_keepTransportMatrixTorch = None
def renderTorch(transportMatrix, ibls, renderSize):
    global g_keepTransportMatrixTorch
    # NOTE : contrary to renderNumpy, this takes a _batch_ of IBLs as input
    if g_keepTransportMatrixTorch is None:
        g_keepTransportMatrixTorch = torch.from_numpy(transportMatrix)
        g_keepTransportMatrixTorch = g_keepTransportMatrixTorch.to(dtype=torch.float32, device=configInfos.device, non_blocking=True)
    
    tm = g_keepTransportMatrixTorch

    output = torch.empty((ibls.shape[0], ibls.shape[1], 3, renderSize[0], renderSize[1]), dtype=torch.float32, device=configInfos.device)

    # We put the channels dimension before height and width to avoid
    # suboptimal memory access in the upcoming matmul
    ibls = ibls.transpose(2, 3).transpose(2, 4).contiguous()
    
    # Note that technically, we could do all these three nested loops in a single matmul
    # But this uses way too much memory
    for b in range(ibls.shape[0]):      # Batch index
        for p in range(ibls.shape[1]):  # Position index
            for c in range(3):          # Color channel
                output[b, p, c] = torch.matmul(tm, ibls[b, p, c].view(-1)).view(renderSize[0], renderSize[1])

    return output



def convertToIBLTorchSUN360(centers, radius, size=(128, 256), p=None):
    distMat = getDistancesTorch(size)
    clampedRadius = torch.clamp(radius, np.pi/(size[0]+1), np.pi/2)

    ibl = torch.zeros(size[:2]).to(configInfos.device, dtype=torch.float32)
    xyzmat = getXYZlatlongTorch(size)
    oneT = torch.ones((1,)).to(configInfos.device, dtype=torch.float32)

    for lightIndex in range(centers.shape[0]):
        # Compute the distance matrix for this light
        xyzCenter = spherical2xyzTorch(centers[lightIndex:lightIndex+1, 0],
                                        centers[lightIndex:lightIndex+1, 1],
                                        norm=oneT)
        dist = torch.matmul(xyzmat, xyzCenter.t())

        # Adjust the radius
        # and binarize it (it is kept as float, but the values are either 0 or 1)
        # TODO radius fixed
        dist.div_(torch.cos(radius)) #clampedRadius[lightIndex]))
        dist[dist < 1] = 0
        distNormalized = torch.div(dist, dist.max())

        # Expand it with 3 channels
        distNormalized = distNormalized.view(size[:2])

        # Apply the color to create the light
        light = torch.mul(distNormalized, oneT)

        # Add it to the IBL
        ibl = torch.add(ibl, light)

    # Clamp to 1 (in case there are multiple lights one over another)
    ibl = torch.clamp(ibl, max=1.0)

    return ibl


def convertSGToIBLTorchSUN360(centers, radius, size=(128, 256), p=None, intensities=None):
    ibl = torch.zeros(size[:2]).to(configInfos.device, dtype=torch.float32)
    xyzmat = getXYZlatlongTorch(size, configInfos.device)
    oneT = torch.ones((1,)).to(configInfos.device, dtype=torch.float32)

    for lightIndex in range(centers.shape[0]):
        # Compute the distance matrix for this light
        if configInfos.coordinateSystem == 'spherical':
            xyzCenter = spherical2xyzTorch(centers[lightIndex:lightIndex+1, 0],
                                            centers[lightIndex:lightIndex+1, 1],
                                            norm=oneT)
        elif configInfos.coordinateSystem == 'xyz':
            xyzCenter = centers[lightIndex:lightIndex+1]
        else:
            raise ValueError("Invalid coordinate system '{}'".format(configInfos.coordinateSystem))

        dist = torch.matmul(xyzmat, xyzCenter.t())      # The order changes nothing here, right?

        light = torch.exp(torch.div(dist-1, radius[lightIndex].div(3)))

        light = light.squeeze().mul(intensities[lightIndex])

        # Add it to the IBL
        ibl = torch.add(ibl, light.squeeze())
    
    #ibl = ibl.div(ibl.max())
    # Clamp to 1 (in case there are multiple lights one over another)
    #ibl = torch.clamp(ibl, max=1.0)

    return ibl

def convertSGToIBLTorchUlavalHDR(centers, radius, intensities, ambients, size=(128, 256, 3), p=None):
    ibl = ambients.unsqueeze(0).unsqueeze(0).expand(*size)
    
    xyzmat = getXYZlatlongTorch(size, configInfos.device)
    oneT = torch.ones((1,)).to(configInfos.device, dtype=torch.float32)

    for lightIndex in range(centers.shape[0]):
        # Compute the distance matrix for this light
        if configInfos.coordinateSystem == 'spherical':
            xyzCenter = spherical2xyzTorch(centers[lightIndex:lightIndex+1, 0],
                                            centers[lightIndex:lightIndex+1, 1],
                                            norm=oneT)
        elif configInfos.coordinateSystem == 'xyz':
            xyzCenter = centers[lightIndex:lightIndex+1]
        else:
            raise ValueError("Invalid coordinate system '{}'".format(configInfos.coordinateSystem))

        dist = torch.matmul(xyzmat, xyzCenter.t())      # The order changes nothing here, right?

        light = torch.exp(torch.div(dist-1, radius[lightIndex].div(4*np.pi)))

        # Expand it to 3 channels
        light = light.expand(size[0], size[1], 3)
        light = light.squeeze().mul(intensities[lightIndex])

        # Add it to the IBL
        ibl = torch.add(ibl, light.squeeze())

    return ibl


def convertSGToIBLTorch(centers, depths, radius, colors, ambients, size=(128,256,3), p=None, device=None, convertFromXYZ=False):
    
    if device is None:
        device = configInfos.device

    ibl = ambients.unsqueeze(0).unsqueeze(0).expand(size[0], size[1], 3)
    xyzmat = getXYZlatlongTorch(size[:2], device)
    oneT = torch.ones((1,)).to(device, dtype=torch.float32)
    
    for lightIndex in range(centers.shape[0]):
        # Compute the distance matrix for this light
        if convertFromXYZ:
            assert centers.shape[1] == 3
            xyzCenter = centers[lightIndex:lightIndex+1]
        else:
            xyzCenter = spherical2xyzTorch(centers[lightIndex:lightIndex+1, 0],
                                            centers[lightIndex:lightIndex+1, 1],
                                            norm=oneT)
        dist = torch.matmul(xyzmat, xyzCenter.t())      # The order changes nothing here, right?
        # Adjust the radius
        light = torch.exp(torch.div(dist-1, radius[lightIndex].div(4*np.pi)))

        # Expand it to 3 channels
        light = light.view(size[:2]).unsqueeze(2).expand(size[0], size[1], 3)

        light = light.mul(colors[lightIndex])

        # Add it to the IBL
        ibl = torch.add(ibl, light)

    return ibl
