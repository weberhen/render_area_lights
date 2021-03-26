import os
import hdrio
from scipy.ndimage.interpolation import zoom
import imageio
import numpy as np
from read_3light_prediction import read_3light_prediction
from create_mitsuba_command import create_mitsuba_command

steps = 8
obj_positions = [[.25, i/100, j] for i,j in zip(np.linspace(-27,0,steps), np.linspace(1,10,steps))]
imInput = imageio.imread(os.path.join('class.jpg')).astype('float32') / 255.
imInputResized = zoom(imInput, (400/imInput.shape[0], 600/imInput.shape[1], 1.0), order=1)

pickle_filename = 'class.pkl'

command = 'mitsuba render_pano.xml'
os.system(command)

# scenes = ['render_sphere_envmap', 'render_sphere']
# scenes = ['render_armadillo_envmap', 'render_armadillo']
scenes = ['render_armadillo']

posCenters, radius, intensities, ambients, depths = read_3light_prediction(pickle_filename)

for scene in scenes:
    for i, obj_pos in enumerate(obj_positions):
        command = create_mitsuba_command(obj_pos, posCenters, radius, intensities, ambients, depths, scene)
        # command = 'mitsuba -Dobjx='+str(obj_pos[0])+' -Dobjy='+str(obj_pos[1])+' -Dobjz='+str(obj_pos[2])+' '+scene+'.xml'
        print(command)
        os.system(command)
        
        os.system('mtsutil tonemap -m 100 -o render_'+str(i).zfill(3)+'.png '+scene+'.exr')
        
        imRender = hdrio.imread(scene+'.exr').astype('float32')
        
        alphaRender = imRender[..., 3:4]
        imRender = (imRender[..., :3]*100)**(1/2.2)
        imCompose = (1-alphaRender)*imInputResized + alphaRender*imRender

        imComposeUint8 = (np.clip(imCompose, 0.0, 1.0)*255).astype('uint8')

        imageio.imsave(scene+'_composite_'+str(i).zfill(3)+'.png', imComposeUint8)
        
    os.system('convert -delay 10 '+scene+'_composite_*.png -loop 0 '+scene+'_render.gif')
    os.system('rm *.png && rm *.exr && rm *.log')
    