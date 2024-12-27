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
    
    inframe['moment_M6s'] = (1. - inframe['ctk']**2)*inframe['ctl']
    inframe['moment_M6c'] = (inframe['ctk']**2) * inframe['ctl']
    
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

        vals["err_M1s"].append(np.sqrt(angles_m1s))
        vals["err_M1c"].append(np.sqrt(angles_m1c))
        vals["err_M2s"].append(np.sqrt(angles_m2s))
        vals["err_M2c"].append(np.sqrt(angles_m2c))
        
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

def calculateDerivedObs(inframe, returnFrame, add_variances = True):
   
    returnFrame["FL"] = -inframe["S2c"]
    returnFrame["AFB"] = (3./4.)*inframe["S6s"]

    returnFrame["denom"] = np.sqrt(-1.*inframe["S2s"]*inframe["S2c"])
   
    returnFrame["P1"] = 0.5*(inframe["S3"]/inframe["S2s"])
    returnFrame["P2"] = (1./8.)*(inframe['S6s']/inframe['S2s'])
    returnFrame["P3"] = -0.25*inframe["S9"]/inframe["S2s"]
    returnFrame["P4p"] = 2.*0.5*inframe["S4"]/returnFrame["denom"]
    returnFrame["P5p"] = 0.5*inframe["S5"]/returnFrame["denom"]
    returnFrame["P6p"] = -0.5*inframe["S7"]/returnFrame["denom"]
    returnFrame["P8p"] = 0.5*inframe["S8"]/returnFrame["denom"]

    if add_variances:

        var_to_index = {0:'S2s', 1:'S2c', 2:'S3', 3:'S4', 4:'S5', 5:'S6s', 6:'S7', 7:'S8', 8:'S9',
                        9:'P1', 10:'P2', 11:'P3', 12:'P4p', 13:'P5p', 14:'P6p', 15:'P8p'}
        J = {}
        cov = {}
        for _i in range(9):
            for _j in range(16):
                if _i == _j:
                    J[f'{_i}_{_j}'] = 1.
                else:
                    J[f'{_i}_{_j}'] = 0.
        for _i in range(9):
            for _j in range(_i,9):
                cov[f'{_i}_{_j}'] = inframe['var_{_a}_{_b}'.format(_a = var_to_index[_i], _b = var_to_index[_j])]
                cov[f'{_j}_{_i}'] = inframe['var_{_a}_{_b}'.format(_a = var_to_index[_i], _b = var_to_index[_j])]
                
        J['0_9'] = -returnFrame["P1"]/inframe["S2s"]
        J['2_9'] = returnFrame["P1"]/inframe["S3"]
        J['0_10'] = -returnFrame["P2"]/inframe["S2s"]
        J['5_10'] = returnFrame["P2"]/inframe["S6s"]
        J['0_11'] = -returnFrame["P3"]/inframe["S2s"]
        J['8_11'] = returnFrame["P3"]/inframe["S9"]
        
        J['0_12'] = -0.5*returnFrame["P4p"]/inframe["S2s"]
        J['1_12'] = -0.5*returnFrame["P4p"]/inframe["S2c"]
        J['3_12'] = returnFrame["P4p"]/inframe["S4"]
        
        J['0_13'] = -0.5*returnFrame["P5p"]/inframe["S2s"]
        J['1_13'] = -0.5*returnFrame["P5p"]/inframe["S2c"]
        J['4_13'] = returnFrame["P5p"]/inframe["S5"]
        
        J['0_14'] = -0.5*returnFrame["P6p"]/inframe["S2s"]
        J['1_14'] = -0.5*returnFrame["P6p"]/inframe["S2c"]
        J['6_14'] = returnFrame["P6p"]/inframe["S7"]
        
        J['0_15'] = -0.5*returnFrame["P8p"]/inframe["S2s"]
        J['1_15'] = -0.5*returnFrame["P8p"]/inframe["S2c"]
        J['7_15'] = returnFrame["P8p"]/inframe["S8"]
        
        outcov = {}
        for _i in range(16):
            for _j in range(16):
                outcov[f'{_i}_{_j}'] = 0.
                for _l in range(9):
                    for _k in range(9):
                        outcov[f'{_i}_{_j}'] += J[f'{_l}_{_i}'] * cov[f'{_l}_{_k}'] * J[f'{_k}_{_j}']
        for _v in range(0,16):
            returnFrame["var_{_a}".format(_a = var_to_index[_v])] = outcov[f'{_v}_{_v}']
            for _v2 in range(_v,16):
                returnFrame["var_{_a}_{_b}".format(_a = var_to_index[_v], _b = var_to_index[_v2])] = outcov[f'{_v}_{_v2}']
        
        #inframe["var_P1"] = inframe["var_S3"]*((inframe["P1"]/inframe["S3"]))**2 + inframe["var_S2s"]*((inframe["P1"]/inframe["S2s"]))**2 - 2.*inframe["var_S2s_S3"]*(inframe["P1"]/inframe["S3"])*(inframe["P1"]/inframe["S2s"])
    
        #inframe["var_P2"] = inframe["var_S6s"]*((inframe["P2"]/inframe["S6s"]))**2 + inframe["var_S2s"]*((inframe["P2"]/inframe["S2s"]))**2 - 2.*inframe["var_S2s_S6s"]*(inframe["P2"]/inframe["S6s"])*(inframe["P2"]/inframe["S2s"])

        #inframe["var_P3"] = inframe["var_S9"]*((inframe["P3"]/inframe["S9"]))**2 + inframe["var_S2s"]*((inframe["P3"]/inframe["S2s"]))**2 - 2.*inframe["var_S2s_S9"]*(inframe["P3"]/inframe["S9"])*(inframe["P3"]/inframe["S2s"])
    
        #inframe["var_P4p"] = inframe["var_S4"]*((inframe["P4p"]/inframe["S4"]))**2 + 0.25*inframe["var_S2s"]*((inframe["P4p"]/inframe["S2s"]))**2 + 0.25*inframe["var_S2c"]*((inframe["P4p"]/inframe["S2c"]))**2 - 0.5*2.*inframe["var_S2s_S4"]*(inframe["P4p"]/inframe["S4"])*(inframe["P4p"]/inframe["S2s"]) - 0.5*2.*inframe["var_S2c_S4"]*(inframe["P4p"]/inframe["S4"])*(inframe["P4p"]/inframe["S2c"]) + 0.5*0.5*2.*inframe["var_S2s_S2c"]*(inframe["P4p"]/inframe["S2s"])*(inframe["P4p"]/inframe["S2c"])
    
        #inframe["var_P5p"] = inframe["var_S5"]*((inframe["P5p"]/inframe["S5"]))**2 + 0.25*inframe["var_S2s"]*((inframe["P5p"]/inframe["S2s"]))**2 + 0.25*inframe["var_S2c"]*((inframe["P5p"]/inframe["S2c"]))**2 - 0.5*2.*inframe["var_S2s_S5"]*(inframe["P5p"]/inframe["S5"])*(inframe["P5p"]/inframe["S2s"]) - 0.5*2.*inframe["var_S2c_S5"]*(inframe["P5p"]/inframe["S5"])*(inframe["P5p"]/inframe["S2c"]) + 0.5*0.5*2.*inframe["var_S2s_S2c"]*(inframe["P5p"]/inframe["S2s"])*(inframe["P5p"]/inframe["S2c"])
    
        #inframe["var_P6p"] = inframe["var_S7"]*((inframe["P6p"]/inframe["S7"]))**2 + 0.25*inframe["var_S2s"]*((inframe["P6p"]/inframe["S2s"]))**2 + 0.25*inframe["var_S2c"]*((inframe["P6p"]/inframe["S2c"]))**2 - 0.5*2.*inframe["var_S2s_S7"]*(inframe["P6p"]/inframe["S7"])*(inframe["P6p"]/inframe["S2s"]) - 0.5*2.*inframe["var_S2c_S7"]*(inframe["P6p"]/inframe["S7"])*(inframe["P6p"]/inframe["S2c"]) + 0.5*0.5*2.*inframe["var_S2s_S2c"]*(inframe["P6p"]/inframe["S2s"])*(inframe["P6p"]/inframe["S2c"])
    
        #inframe["var_P8p"] = inframe["var_S8"]*((inframe["P8p"]/inframe["S8"]))**2 + 0.25*inframe["var_S2s"]*((inframe["P8p"]/inframe["S2s"]))**2 + 0.25*inframe["var_S2c"]*((inframe["P8p"]/inframe["S2c"]))**2 - 0.5*2.*inframe["var_S2s_S8"]*(inframe["P8p"]/inframe["S8"])*(inframe["P8p"]/inframe["S2s"]) - 0.5*2.*inframe["var_S2c_S8"]*(inframe["P8p"]/inframe["S8"])*(inframe["P8p"]/inframe["S2c"]) + 0.5*0.5*2.*inframe["var_S2s_S2c"]*(inframe["P8p"]/inframe["S2s"])*(inframe["P8p"]/inframe["S2c"])    
        
    else:
        for _p in ['P1','P2','P3','P4p','P5p','P6p','P8p']:
            returnFrame[f"var_{_p}"] = 1.
    return returnFrame


def calculateQuimObs(inframe, T, add_variances = True):
    mb = 4.8
    mB = 5.279
    mbh = 4.8/5.279
    mB2 = mB**2
    q2arr = T #.reshape(nq2)
    sh = q2arr/mB2
    A_B = 8.*(0.892**2 / mB2)*sh/((1. - sh)**2)

    z = np.sqrt(4.*inframe['P2']**2 - 8.*inframe['P2']*inframe['P4p']*inframe['P5p'] + inframe['P5p']**2)
    c9_c7_ratio = -(mbh)*(1 + (inframe['P5p']*(sh - 1.))/(z) + sh )
    c9_c7_ratio /= sh
    
    ff_ratio = np.sqrt((-inframe["S2c"]/(1.+inframe["S2c"])) * A_B * ((inframe["P5p"] + z)**2/(inframe["P2"]**2)))
    ff_ratio *= (1./np.sqrt(2))
    
    A = inframe["P2"]/z
    B = np.sqrt(2*inframe["P2"]*inframe["P4p"] + inframe["P5p"] - 2.*inframe["P4p"]**2 * inframe["P5p"] + z)
    D = 1./np.sqrt(-2.*inframe["P2"]*inframe["P4p"] + inframe["P5p"] + z)
    C = 2.*(sh - 1) * mbh/sh
    
    
    inframe["ff"] = ff_ratio
    inframe["C9_C7"] = c9_c7_ratio
    inframe["C10_C7"] = C*A*B*D
    
    if add_variances:
        var_to_index = {0:'S2c', 1:'P2', 2:'P4p', 3:'P5p', 4:'C9_C7', 5:'ff', 6:'C10_C7'}
        J = {}
        cov = {}
        for _i in range(4):
            for _j in range(7):
                if _i == _j:
                    J[f'{_i}_{_j}'] = 1.
                else:
                    J[f'{_i}_{_j}'] = 0.
        for _i in range(4):
            for _j in range(_i,4):
                cov[f'{_i}_{_j}'] = inframe['var_{_a}_{_b}'.format(_a = var_to_index[_i], _b = var_to_index[_j])]
                cov[f'{_j}_{_i}'] = inframe['var_{_a}_{_b}'.format(_a = var_to_index[_i], _b = var_to_index[_j])]
                
        J['1_4'] = 0.5*(mbh/sh)*(8.*inframe["P2"] - 8.*inframe["P4p"]*inframe["P5p"])*(inframe['P5p']*(sh - 1.))/(z**3)
        J['2_4'] = -0.5*(mbh/sh)*4.*inframe["P2"]*inframe["P5p"] * (inframe['P5p']*(sh - 1.))/(z**3)
        J['3_4'] = (sh - 1.)*(1./z**2) * (z - (0.5*inframe["P5p"]*(-4.*inframe["P2"]*inframe["P4p"] + 2.*inframe["P5p"])*(1./z)))
        
        J['0_5'] = 0.5*inframe["ff"]/(inframe["S2c"]*(1+inframe["S2c"]))
        
        dz_dp5p = (1./z**2) * (z - (0.5*inframe["P5p"]*(-4.*inframe["P2"]*inframe["P4p"] + 2.*inframe["P5p"])*(1./z)))
        K = ((inframe["P5p"] + z)**2/(inframe["P2"]**2))
        L = (inframe["S2c"]/(1.+inframe["S2c"]))
        
        dzdp2 = (4.*inframe["P2"] - 4.*inframe["P4p"]*inframe["P5p"])/z
        dzdp4p = -2.*inframe["P4p"]*inframe["P5p"]/z
        dzdp5p = 0.5*(-4.*inframe["P2"]*inframe["P4p"] + 2.*inframe["P5p"])/z
        
        dAdP2 = (z - inframe["P2"]*(4.*inframe["P2"] - 4.*inframe["P4p"]*inframe["P5p"]))/z**2
        dAdP4p = 2.*inframe["P2"]*inframe["P4p"]*inframe["P5p"]/z**3
        dAdP5p = (-inframe["P2"]/z**2) * 0.5 * (-4.*inframe["P2"]*inframe["P4p"] + 2.*inframe["P5p"])/z
        
        dBdP2 = 0.5*(2.*inframe["P4p"] + dzdp2)/B
        dBdP4p = 0.5*(2.*inframe["P2"] - 4.*inframe["P4p"]*inframe["P5p"] + dzdp4p)/B
        dBdP5p = 0.5*(1. - 2.*inframe["P4p"]**2 + dzdp5p)/B
        
        dDdP2 = -0.5*(-2.*inframe["P4p"] + dzdp2)*D**3
        dDdP4p = -0.5*(-2.*inframe["P2"] + dzdp4p)*D**3
        dDdP5p = -0.5*(1. + dzdp5p)*D**3
        
        
        J['1_5'] = (1./(2.*K)) * inframe["ff"] * (-2.*K/inframe["P2"] + ((2*z + 2*inframe["P5p"])*(8.*inframe["P2"] - 4.*inframe["P4p"]*inframe["P5p"]))/(2.*inframe["P2"]**2 * z))
        J['2_5'] = (1./np.sqrt(2))* A_B * (-2.*inframe["P2"]*inframe["P5p"]/z)*((-inframe["S2c"]/(1.+inframe["S2c"]))*((2*z + 2*inframe["P5p"])/(inframe["P2"]**2)))
        J['3_5'] = (1./(2.*K)) * inframe["ff"] * (1./(inframe["P2"]**2))*(2.*inframe["P5p"] + 2.*z*dz_dp5p + 2.*z + 2.*inframe["P5p"]*dz_dp5p)

        J['1_6'] = C*(dAdP2*B*D + A*dBdP2*D + A*B*dDdP2)
        J['2_6'] = C*(dAdP4p*B*D + A*dBdP4p*D + A*B*dDdP4p)
        J['3_6'] = C*(dAdP5p*B*D + A*dBdP5p*D + A*B*dDdP5p)
        
        outcov = {}
        for _i in range(7):
            for _j in range(7):
                outcov[f'{_i}_{_j}'] = 0.
                for _l in range(4):
                    for _k in range(4):
                        outcov[f'{_i}_{_j}'] += J[f'{_l}_{_i}'] * cov[f'{_l}_{_k}'] * J[f'{_k}_{_j}']
        for _v in range(4,7):
            inframe["var_{_a}".format(_a = var_to_index[_v])] = outcov[f'{_v}_{_v}']
            for _v2 in range(_v,7):
                inframe["var_{_a}_{_b}".format(_a = var_to_index[_v], _b = var_to_index[_v2])] = outcov[f'{_v}_{_v2}']
    else:
        inframe["var_ff"] = 1.
        inframe["var_C9_C7"] = 1.
        inframe["var_C10_C7"] = 1.
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
    