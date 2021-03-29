from os.path import join
import numpy as np


def gen_mitsuba_xml(xyz, cam_pos, uv_S, texture, filename):
    colors = np.zeros(xyz.shape)
    for i in range(uv_S.shape[1]):
        colors[i] = texture[int(uv_S[1][i]*texture.shape[0])][int(uv_S[0][i]*texture.shape[1])]
        
    xml_string = '\n'.join(["<scene version='0.5.0'>",
                                "   <integrator type='path'>",
                                "       <boolean name='hideEmitters' value='false' />",
                                "       <integer name='maxDepth' value='8'/>",
                                "   </integrator>",
                                "   <sensor type='spherical'>",
                                "       <transform name='toWorld'>",
                                "           <rotate y='1' angle='180'/>",
                                "           <translate x='"+str(cam_pos[0])+"' y='"+str(cam_pos[1])+"' z='"+str(cam_pos[2])+"' />",
                                "       </transform>",
                                # "       <float name='fov' value='50' />",
                                "       <sampler type='independent'>",
                                "           <integer name='sampleCount' value='8' />",
                                "       </sampler>",
                                "       <film type='hdrfilm'>",
                                "           <integer name='width' value='600' />",
                                "           <integer name='height' value='300' />",
                                "           <boolean name='banner' value='false' />",
                                "           <string name='pixelFormat' value='rgba' />",
                                "           <boolean name='attachLog' value='false' />",
                                "       </film>",
                                "   </sensor>",
                                # "    <shape type='sphere'>",
                                # "        <transform name='toWorld'>",
                                # "            <scale value='.1' />",
                                # "            <translate x='$objx' y='$objy' z='$objz' />",
                                # "        </transform>",
                                # "        <bsdf type='conductor'>",
                                # "            <string name='material' value='none' />",
                                # "        </bsdf>",
                                # "    </shape>",
                                # "   <shape type='ply'>",
                                # "       <transform name='toWorld'>",
                                # "           <scale value='100'/>",
                                # "           <translate x='$objx' y='$objy' z='$objz' />",
                                # "           <translate x='0' y='-120' z='-80' />",
                                # "       </transform>",
                                # "       <string name='filename' value='armadillo.ply'/>",
                                # "   </shape>",
                                # "   <shape type='rectangle'>",
                                # "       <bsdf type='diffuse'>",
                                # "           <spectrum name='reflectance' value='1, 1, 1'/>",
                                # "       </bsdf>",
                                # "       <transform name='toWorld'>",
                                # "           <rotate x='1' angle='-120'/>",
                                # "           <scale x='.4' y='.5' z='10'/>",
                                # "           <translate x='.05' y='-.18' z='10'/>",
                                # "       </transform>",
                                # "   </shape>\n",
                                # "    <shape type='sphere'>",
                                # "        <transform name='toWorld'>",
                                # "            <scale value='1.0885' />",
                                # "            <translate x='-20.1356406' y='20.23461943' z='-0.3468532' />",
                                # "        </transform>",
                                # "        <emitter type='area'>",
                                # "            <spectrum name='radiance' value='400.5285, 400.6863, 600.2572' />",
                                # "        </emitter>",
                                # "    </shape>\n",
                                # "    <shape type='sphere'>",
                                # "        <transform name='toWorld'>",
                                # "            <scale value='1.0665' />",
                                # "            <translate x='10.3037382' y='20.29353069' z='8.9429226' />",
                                # "        </transform>",
                                # "        <emitter type='area'>",
                                # "            <spectrum name='radiance' value='600.5889, 80.1095, 60.2580' />",
                                # "        </emitter>",
                                # "    </shape>",
                                # "    <shape type='sphere'>",
                                # "        <transform name='toWorld'>",
                                # "            <scale value='1.0596' />",
                                # "            <translate x='14.76108' y='14.3538714' z='19.780449' />",
                                # "        </transform>",
                                # "        <emitter type='area'>",
                                # "            <spectrum name='radiance' value='600.5108, 70.5243, 80.8027' />",
                                # "        </emitter>",
                                # "    </shape>",
                                ])
    
    for xyz_p, color_p in zip(xyz, colors):
        # if xyz_p[1] >= -150.:
        xml_string+='\n'.join([
            "   <shape type='sphere'>",
            "        <transform name='toWorld'>",
            "            <scale value='.1' />",
            "            <translate x='"+str(xyz_p[0])+"' y='"+str(xyz_p[1])+"' z='"+str(xyz_p[2])+"' />",
            "        </transform>",
            "       <emitter type='area'>",
            "            <spectrum name='radiance' value='"+", ".join([str(color_p[0]/255.),str(color_p[1]/255.),str(color_p[2]/255.)])+"' />",
            "        </emitter>",
            "    </shape>\n"
            ])
        # else:
        # xml_string+='\n'.join([
        #     "   <shape type='sphere'>",
        #     "        <transform name='toWorld'>",
        #     "            <scale value='.1' />",
        #     "            <translate x='"+str(xyz_p[0])+"' y='"+str(xyz_p[1])+"' z='"+str(xyz_p[2])+"' />",
        #     "        </transform>",
        #     "       <bsdf type='diffuse'>",
        #     "            <spectrum name='reflectance' value='1, 1, 1' />",
        #     "       </bsdf>",
        #     "    </shape>\n"
        #     ])

    xml_string+="</scene>"
    
    
    with open(filename, "w") as a_file:
        a_file.write(xml_string)
    s=1        
	
    
# gen_mitsuba_xml(None, None, None, 'test.xml')