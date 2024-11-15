import numpy as np


def calculateAllMoments(inframe):
    """
    This is a function that calculates the moments for the input data frame.
    Does as a DataFrame for maximum speed
    """
    
    inframe['moment_FL'] = 2. - 2.5*(1. - inframe['ctk']**2)
    inframe['moment_M1s'] = (1. - inframe['ctk']**2)
    inframe['moment_M1c'] = inframe['ctk']**2
    inframe['moment_M2s'] = (1. - inframe['ctk']**2) * (2.*inframe['ctl']**2 - 1.)
    inframe['moment_M2c'] = inframe['ctk']**2 * (2.*inframe['ctl']**2 - 1.)
    inframe['moment_S1s'] = 0.0625*(21.*inframe['moment_M1s'] - 14.*inframe['moment_M1c'] + 15.*inframe['moment_M2s'] - 10.*inframe['moment_M2c'])
    inframe['moment_S1c'] = 0.125*(-7.*inframe['moment_M1s'] + 28.*inframe['moment_M1c'] - 5.*inframe['moment_M2s'] + 20.*inframe['moment_M2c'])
    inframe['moment_S2s'] = 0.0625*(15.*inframe['moment_M1s'] - 10.*inframe['moment_M1c'] + 45.*inframe['moment_M2s'] - 30.*inframe['moment_M2c'])
    inframe['moment_S2c'] = 0.125*(-5.*inframe['moment_M1s'] + 20.*inframe['moment_M1c'] - 15.*inframe['moment_M2s'] + 60.*inframe['moment_M2c'])
    inframe['moment_S3'] = 3.125*(1. - inframe['ctk']**2)*(1. - inframe['ctl']**2)*np.cos(2.*inframe['phi'])
    inframe['moment_S5'] = 2.5 * np.sqrt(1. - inframe['ctl']*inframe['ctl']) * 2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk']) * np.cos(inframe['phi'])
    inframe['moment_S4'] = 3.125*2*inframe['ctl'] * np.sqrt(1. - inframe['ctl']*inframe['ctl']) * 2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk']) * np.cos(inframe['phi'])
    inframe['moment_AFB'] = 0.75*2.5*(1. - inframe['ctk']**2)*inframe['ctl']
    inframe['moment_S7'] = 2.5*2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk'])*np.sqrt(1. - inframe['ctl']*inframe['ctl'])*np.sin(inframe['phi'])
    inframe['moment_S8'] = 3.125*2 * inframe['ctk'] * np.sqrt(1. - inframe['ctk']*inframe['ctk'])*2.*inframe['ctl']*np.sqrt(1. - inframe['ctl']*inframe['ctl'])*np.sin(inframe['phi'])
    inframe['moment_S9'] = 3.125*(1. - inframe['ctk']**2)*(1. - inframe['ctl']**2)*np.sin(2.*inframe['phi'])
    return inframe

def calculateAllVariances(inframe, indices, vals, n_neighbours):
    _i = 0
    for _ind in indices:
        rs = inframe.iloc[_ind]
        
        angles_m1s = 1./(n_neighbours*(n_neighbours-1)) * np.sum((vals['M1s'] - (1. - rs['ctk']**2))**2)
        angles_m1c = 1./(n_neighbours*(n_neighbours-1)) * np.sum((vals['M1c'] - (rs['ctk']**2))**2)
        angles_m2s = 1./(n_neighbours*(n_neighbours-1)) * np.sum((vals['M2s'] - (1. - rs['ctk']**2) * (2*inframe['ctl']**2 - 1.))**2)
        angles_m2c = 1./(n_neighbours*(n_neighbours-1)) * np.sum((vals['M2c'] - (rs['ctk']**2) * (2*inframe['ctl']**2 - 1.))**2)

        vals.append["err_M1s"].append(np.sqrt(angles_m1s))
        vals.append["err_M1c"].append(np.sqrt(angles_m1c))
        vals.append["err_M2s"].append(np.sqrt(angles_m2s))
        vals.append["err_M2c"].append(np.sqrt(angles_m2c))
        
        vals["err_S1s"] = np.sqrt(0.0625**2 * (441. * angles_m1s + 196. * angles_m1c + 225.*angles_m2s + 100.*angles_m2c))*vals["S1s"][_i]
        vals["err_S1c"] = np.sqrt(0.125**2 * (49. * angles_m1s + 784. * angles_m1c + 25.*angles_m2s + 400.*angles_m2c))*vals["S1c"][_i]
        vals["err_S2s"] = np.sqrt(0.0625**2 * (225. * angles_m1s + 100. * angles_m1c + 2025.*angles_m2s + 900.*angles_m2c))*vals["S2s"][_i]
        vals["err_S2c"] = np.sqrt(0.125**2 * (25. * angles_m1s + 400. * angles_m1c + 225.*angles_m2s + 3600.*angles_m2c))*vals["S2c"][_i]
    
        angles2 = 0.8 - 0.4*vals["FL"][_i] - (1. - rs['ctk']**2)
        vals["err_FL"].append(2.5*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles2**2)))
    
        angles3 = 0.32*vals["S3"][_i] - (1. - rs['ctk']**2)*(1. - rs['ctl']**2)*np.cos(2.*rs['phi'])
        vals["err_S3"].append(3.125*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles3**2)))
    
        angles4 = 0.32*vals["S4"][_i] - 2*rs['ctl'] * np.sqrt(1. - rs['ctl']*rs['ctl']) * 2 * rs['ctk'] * np.sqrt(1. -rs['ctk']*rs['ctk']) * np.cos(rs['phi'])
        vals["err_S4"].append(3.125*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles4**2)))
    
        angles5 = 0.4*vals["S5"][_i] - np.sqrt(1. - rs['ctl']*rs['ctl']) * 2 * rs['ctk'] * np.sqrt(1. - rs['ctk']*rs['ctk']) * np.cos(rs['phi'])
        vals["err_S5"].append(2.5*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles5**2)))
    
        angles6 = 0.4*(4./3.)*vals["AFB"][_i] - (1. - rs['ctk']**2)*rs['ctl']
        vals["err_AFB"].append(0.75*2.5*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles6**2)))
    
        angles7 = 0.4*vals["S7"][_i] - 2 * rs['ctk'] * np.sqrt(1. - rs['ctk']*rs['ctk'])*np.sqrt(1. - rs['ctl']*rs['ctl'])*np.sin(rs['phi'])
        vals["err_S7"].append(2.5*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles7**2)))
    
        angles8 = 0.32*vals["S8"][_i] - 2 * rs['ctk'] * np.sqrt(1. - rs['ctk']*rs['ctk'])*2.*rs['ctl']*np.sqrt(1. - rs['ctl']*rs['ctl'])*np.sin(rs['phi'])
        vals["err_S8"].append(3.125*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles8**2)))
    
        angles9 = 0.32*vals["S9"][_i] - (1. - rs['ctk']**2)*(1. - rs['ctl']**2)*np.sin(2.*rs['phi'])
        vals["err_S9"].append(3.125*np.sqrt( 1./(n_neighbours*(n_neighbours-1)) * np.sum(angles9**2)))
        
        _i += 1

    return vals

def calculateOptimisedObservables(inframe):
    """
    This is a function that calculates the optimised observables from the regular ones.
    Does this as a DataFrame for maximum speed
    """
    #inframe["S2s"] = 0.25*(1. - inframe["moment_FL"])
    #inframe["S2c"] = -1. * inframe["moment_FL"]
    inframe["denom"] = np.sqrt(-1.*inframe["S2s"]*inframe["S2c"])
    
    inframe["P1"] = 0.5*(inframe["moment_S3"]/inframe["S2s"])
    inframe["P2"] = (1./6.)*(inframe['moment_AFB']/inframe['S2s'])
    inframe["P3"] = -0.25*inframe["moment_S9"]/inframe["S2s"]
    inframe["P4p"] = 0.5*inframe["moment_S4"]/inframe["denom"]
    inframe["P5p"] = 0.5*inframe["moment_S5"]/inframe["denom"]
    inframe["P6p"] = 0.5*inframe["moment_S7"]/inframe["denom"]
    inframe["P8p"] = 0.5*inframe["moment_S8"]/inframe["denom"]
    
    return inframe

def calculateOptimisedObservable(siarr, flarr, s2sarr, s2carr, obs):
    
    optimisedObsTranslator = {'FL' : 'FL', 'S3' : 'P1', 'S4' : 'P4p', 'S5' : 'P5p', 'AFB' : 'P2', 'S7' : 'P6p', 'S8' : 'P8p', 'S9' : 'P3'}
    
    tobs = obs
    if obs in optimisedObsTranslator.keys():
        tobs = optimisedObsTranslator[obs]
    flarr = np.array(flarr)
    siarr = np.array(siarr)
    s2sarr = np.array(s2sarr)#0.25*(1. - flarr)
    s2carr = np.array(s2carr)#-1.*flarr
    denomarr = np.sqrt(-s2carr*s2sarr)
    if tobs == "FL":
        return flarr
    elif tobs == "S2c":
        return s2carr
    elif tobs == "S2s":
        return s2sarr
    elif tobs == 'P1':
        retarr = 0.5*(siarr/s2sarr)
    elif tobs == 'P2':
        retarr = (1./6.)*(siarr/s2sarr)
    elif tobs == 'P3':
        retarr = -0.25*(siarr/s2sarr)
    else:
        retarr = 0.5*siarr/denomarr
    return retarr
    