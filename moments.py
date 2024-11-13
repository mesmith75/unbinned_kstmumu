import numpy as np


def calculateAllMoments(inframe):
    """
    This is a function that calculates the moments for the input data frame.
    Does as a DataFrame for maximum speed
    """
    
    inframe['moment_FL'] = 2. - 2.5*(1. - inframe['ctk']**2)
    inframe['moment_S3'] = 3.125*(1. - inframe['ctk']**2)*(1. - inframe['ctl']**2)*np.cos(2.*inframe['phi'])
    inframe['moment_S5'] = 2.5 * np.sqrt(1. - inframe['ctl']*inframe['ctl']) * 2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk']) * np.cos(inframe['phi'])
    inframe['moment_S4'] = 3.125*2*inframe['ctl'] * np.sqrt(1. - inframe['ctl']*inframe['ctl']) * 2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk']) * np.cos(inframe['phi'])
    inframe['moment_AFB'] = 0.75*2.5*(1. - inframe['ctk']**2)*inframe['ctl']
    inframe['moment_S7'] = 2.5*2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk'])*np.sqrt(1. - inframe['ctl']*inframe['ctl'])*np.sin(inframe['phi'])
    inframe['moment_S8'] = 3.125*2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk'])*2.*inframe['ctl']*np.sqrt(1. - inframe['ctl']*inframe['ctl'])*np.sin(inframe['phi'])
    inframe['moment_S9'] = 3.125*(1. - inframe['ctk']**2)*(1. - inframe['ctl']**2)*np.sin(2.*inframe['phi'])
    return inframe

def calculateOptimisedObservables(inframe):
    """
    This is a function that calculates the optimised observables from the regular ones.
    Does this as a DataFrame for maximum speed
    """
    inframe["S2s"] = 0.25*(1. - inframe["moment_FL"])
    inframe["S2c"] = -1. * inframe["moment_FL"]
    inframe["denom"] = np.sqrt(-1.*inframe["S2s"]*inframe["S2c"])
    
    inframe["P1"] = 0.5*(inframe["moment_S3"]/inframe["S2s"])
    inframe["P2"] = (1./6.)*(inframe['moment_AFB']/inframe['S2s'])
    inframe["P3"] = -0.25*inframe["moment_S9"]/inframe["S2s"]
    inframe["P4p"] = 0.5*inframe["moment_S4"]/inframe["denom"]
    inframe["P5p"] = 0.5*inframe["moment_S5"]/inframe["denom"]
    inframe["P6p"] = 0.5*inframe["moment_S7"]/inframe["denom"]
    inframe["P8p"] = 0.5*inframe["moment_S8"]/inframe["denom"]
    
    return inframe

def calculateOptimisedObservable(siarr, flarr, obs):
    
    optimisedObsTranslator = {'FL' : 'FL', 'S3' : 'P1', 'S4' : 'P4p', 'S5' : 'P5p', 'AFB' : 'P2', 'S7' : 'P6p', 'S8' : 'P8p', 'S9' : 'P3'}
    
    tobs = obs
    if obs in optimisedObsTranslator.keys():
        tobs = optimisedObsTranslator[obs]
    flarr = np.array(flarr)
    siarr = np.array(siarr)
    s2sarr = 0.25*(1. - flarr)
    s2carr = -1.*flarr
    denomarr = np.sqrt(flarr*s2sarr)
    if tobs == "FL":
        return flarr
    elif tobs == 'P1':
        retarr = 0.5*(siarr/s2sarr)
    elif tobs == 'P2':
        retarr = (1./6.)*(siarr/s2sarr)
    elif tobs == 'P3':
        retarr = -0.25*(siarr/s2sarr)
    else:
        retarr = 0.5*siarr/denomarr
    return retarr
    