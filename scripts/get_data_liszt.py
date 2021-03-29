import os


inputImNames = [
        'AG8A1520-others-200-1.37354-1.02082.png',
        # 'AG8A4786-others-160-1.63987-1.10975.png',
        # '9C4A7811-others-120-1.44844-1.01319.png',
        # 'AG8A8710-others-00-1.73133-1.00330.png',
        # '9C4A8295-others-00-1.73210-1.06243.png',
        # '9C4A0632-others-00-1.84997-0.94982.png',
        # 'AG8A9772-others-280-1.66504-1.06817.png',
        # 'AG8A9746-others-160-1.61411-1.05738.png',
        # 'AG8A9666-others-280-1.61196-1.10387.png',
        # 'AG8A9171-others-200-1.72190-0.87783.png',
        # 'AG8A9100-others-280-2.07701-1.08082.png',
        # 'AG8A8687-others-00-2.16029-1.13881.png',
    ]

input_folder = '/Users/henriqueweber/liszt/LavalIndoor/'
output_folder = '/Users/henriqueweber/datasets/LavalIndoor/'
os.system('mkdir -p '+output_folder+'ldrInputs/test/')
os.system('mkdir -p '+output_folder+'hdrInputs/test/')
os.system('mkdir -p '+output_folder+'ldrOutputs/test/')
os.system('mkdir -p '+output_folder+'output_deepparametric/test/predictedParams/')
os.system('mkdir -p '+output_folder+'output_ldrestimator/test/')
os.system('mkdir -p '+output_folder+'gt_layout/test/')

for filename in inputImNames:
    os.system('cp '+input_folder+'ldrInputs/test/'+filename+' '+output_folder+'/ldrInputs/test/')
    os.system('cp '+input_folder+'ldrOutputs/test/'+filename+' '+output_folder+'/ldrOutputs/test/')
    os.system('cp '+input_folder+'hdrInputs/test/'+filename[:-3]+'exr'+' '+output_folder+'/hdrInputs/test/')
    os.system('cp '+input_folder+'output_deepparametric/test/predictedParams/'+filename[:-3]+'pkl'+' '+output_folder+'/output_deepparametric/test/predictedParams')
    os.system('cp '+input_folder+'output_ldrestimator/test/'+filename[:-3]+'_texture.png'+' '+output_folder+'/output_ldrestimator/test/')
    os.system('cp '+input_folder+'gt_layout/test/'+filename[:-3]+'txt'+' '+output_folder+'/gt_layout/test/')
    