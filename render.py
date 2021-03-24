import os
import hdrio
from scipy.ndimage.interpolation import zoom
import imageio
import numpy as np

# obj_positions = [
#                  [.25,-.21,4],
#                  [.25,-.18,5],
#                  [.25,.05,10] ]
steps = 80
obj_positions = [[.25, i/100, j] for i,j in zip(np.linspace(-27,0,steps), np.linspace(1,10,steps))]
imInput = imageio.imread(os.path.join('class.jpg')).astype('float32') / 255.
imInputResized = zoom(imInput, (400/imInput.shape[0], 600/imInput.shape[1], 1.0), order=1)

command = 'mitsuba render_pano.xml'
os.system(command)

for i, obj_pos in enumerate(obj_positions):
    command = 'mitsuba -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' render_sphere.xml'
    print(command)
    os.system(command)
    
    os.system('mtsutil tonemap -m 100 -o render_'+str(i).zfill(3)+'.png render_sphere.exr')
    
    command = 'mitsuba -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' render_sphere_envmap.xml'
    print(command)
    os.system(command)
    
    os.system('mtsutil tonemap -m 100 -o render_envmap_'+str(i).zfill(3)+'.png render_sphere_envmap.exr')
    
    imRender = hdrio.imread('render_sphere.exr').astype('float32')
    
    alphaRender = imRender[..., 3:4]
    imRender = (imRender[..., :3]*100)**(1/2.2)
    imCompose = (1-alphaRender)*imInputResized + alphaRender*imRender

    imComposeUint8 = (np.clip(imCompose, 0.0, 1.0)*255).astype('uint8')

    imageio.imsave('composite_'+str(i).zfill(3)+'.png', imComposeUint8)
    
    imRender = hdrio.imread('render_sphere_envmap.exr').astype('float32')
    
    alphaRender = imRender[..., 3:4]
    imRender = (imRender[..., :3]*100)**(1/2.2)
    imCompose = (1-alphaRender)*imInputResized + alphaRender*imRender

    imComposeUint8 = (np.clip(imCompose, 0.0, 1.0)*255).astype('uint8')

    imageio.imsave('envmap_composite_'+str(i).zfill(3)+'.png', imComposeUint8)