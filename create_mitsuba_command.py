

def create_mitsuba_command(obj_pos, posCenters, radius, intensities, ambients, depths, scene_filename):
    posLights = posCenters*depths
    command = ' '.join(['mitsuba',
                       '-Dobjx='+str(obj_pos[0]),
                       '-Dobjy='+str(obj_pos[1]),
                       '-Dobjz='+str(obj_pos[2]),
                       '-DposL1x='+str(posLights[0,0]),
                       '-DposL1y='+str(posLights[0,1]),
                       '-DposL1z='+str(posLights[0,1]),
                       '-DposL2x='+str(posLights[1,0]),
                       '-DposL2y='+str(posLights[1,1]),
                       '-DposL2z='+str(posLights[1,1]),
                       '-DposL3x='+str(posLights[2,0]),
                       '-DposL3y='+str(posLights[2,1]),
                       '-DposL3z='+str(posLights[2,1]),
                       '-DscaleL1='+str(radius[0]),
                       '-DscaleL2='+str(radius[1]),
                       '-DscaleL3='+str(radius[2]),
                       '-DintL1="'+str(', '.join([str(i) for i in intensities[0]]))+'"',
                       '-DintL2="'+str(', '.join([str(i) for i in intensities[1]]))+'"',
                       '-DintL3="'+str(', '.join([str(i) for i in intensities[2]]))+'"',
                       scene_filename+'.xml'])
    
    return command