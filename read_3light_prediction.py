import pickle

def read_3light_prediction(paramsPath):
    params = pickle.load(open(paramsPath, 'rb'))
    
    posCenters = params['posCenters']
    radius = params['radius']
    intensities = params['intensities']
    ambients = params['ambients']
    depths = params['depths']
    
    return posCenters, radius, intensities, ambients, depths