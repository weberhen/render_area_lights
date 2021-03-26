"""
This module acts as a proxy for the configuration.
It should be initialized using the initConfig function,
and could then be used by calling
thisModule.keywanted

This requires Python >= 3.7, since this is the first
version implementing PEP 562 (module __getattr__)

This module also contains a logging helper function
"""
import json
import time
import pickle
import os
from copy import deepcopy

import numpy as np

__config = None
__configDefault = None

def initConfig(paramsFile, defaultValuesFile):
    global __config, __configDefault

    with open(paramsFile, 'r') as f:
        __config = json.load(f)
    
    if os.path.exists(defaultValuesFile):
        with open(defaultValuesFile, 'r') as f:
            __configDefault = json.load(f)

    assert isinstance(__config, dict)

def addParameter(key, val):
    global __config
    if __config is None:
        raise AttributeError("Configuration not initialized yet!")
    __config[key] = val

def saveConfig(path, provideMissingValuesFromDefault=True):
    global __config, __configDefault

    if provideMissingValuesFromDefault and __configDefault is not None:
        d = deepcopy(__configDefault)
    else:
        d = {}
    d.update(deepcopy(__config))

    with open(path, 'w') as f:
        json.dump(d, f)

def __getattr__(attr):
    global __config, __configDefault
    if __config is None:
        raise AttributeError("Configuration not initialized yet!")
    
    if attr in __config:
        return __config[attr]
    elif __configDefault is not None and attr in __configDefault:
        return __configDefault[attr]
    else:
        raise AttributeError(f"Attribute {attr} does not exist, nor has a default value provided")

__lastTime = 0
def log(message, inprocess=True):
    global __lastTime
    # message can either be a string or None
    # if inprocess is True, then the message is printed without
    # a carriage return at the end, and the next call with message == None
    # will print the ellapsed time
    if inprocess and message is not None:
        __lastTime = time.time()
        print(message, end=' ')
    elif inprocess and message is None:
        print(" Done in {:.4f} seconds".format(time.time() - __lastTime))
    else:
        print(message)
    