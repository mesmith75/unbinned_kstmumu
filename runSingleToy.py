import pandas as pd
import numpy as np
import cmath
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
obsList = [ 'S1s', 'S1c', 'S2s', 'S2c','S3', 'S4', 'S5', 'S6s','S6c', 'S7', 'S8', 'S9','FL', 'AFB']
momentsList = ['S1s', 'S2s', 'S1c', 'S2c', 'S3', 'S4', 'S5', 'S6s', 'S6c', 'S7', 'S8', 'S9', 'S6s', 'S6c', 'M6s', 'M6c', 'M1s', 'M1c', 'M2s', 'M2c']
optObsList = ['P1', 'P2', 'P3', 'P4p', 'P5p', 'P6p', 'P8p']
totalObsList = ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9', 'S1s', 'S2s', 'S1c', 'S2c', 'P1', 'P2', 'P3', 'P4p', 'P5p', 'P6p', 'P8p','S6s','S6c']
optimisedObsTranslator = {'FL' : 'FL', 'S3' : 'P1', 'S4' : 'P4p', 'S5' : 'P5p', 'AFB' : 'P2', 'S7' : 'P6p', 'S8' : 'P8p', 'S9' : 'P3', 'S1s' : 'S1s', 'S2s' : 'S2s', 'S1c' : 'S1c', 'S2c' : 'S2c', 'M1s' : 'M1s', 'M2s' : 'M2s', 'M1c' : 'M1c', 'M2c' : 'M2c'}
obs_to_label = {'FL' : '$F_{L}$', 'S3' : '$S_{3}$', 'S4' : '$S_{4}$', 'S5' : '$S_{5}$', 'AFB' : '$A_{FB}$', 'S7' : '$S_{7}$', 'S8' : '$S_{8}$', 'S9' : '$S_{9}$',
                'M1s' : '$M_{1}^{s}$', 'M2s' : '$M_{2}^{s}$', 'M1c' : '$M_{1}^{c}$', 'M2c' : '$M_{2}^{c}$', 'S1s' : '$S_{1}^{s}$', 'S2s' : '$S_{2}^{s}$', 'S1c' : '$S_{1}^{c}$', 'S2c' : '$S_{2}^{c}$',
                'P1' : '$P_{1}$', 'P2' : '$P_{2}$', 'P3' : '$P_{3}$', 'P4p' : '$P_{4}^{\prime}$', 'P5p' : '$P_{5}^{\prime}$', 'P6p' : '$P_{6}^{\prime}$', 'P8p' : '$P_{8}^{\prime}$','M6s':'$M_{6}^{s}$',
                'M6c':'$M_{6}^{s}$', 'S6s' : '$S_{6}^{s}$', 'S6c' : '$S_{6}^{c}$'}

import data_classes
import angular_func

print("INFO: Setting up the PDFs")
# Set up the PDFs and make some data sets
mypdf_SM = angular_func.angular_func()
mypdf_NP1 = angular_func.angular_func(model = 'NP_1')
mypdf_NP2 = angular_func.angular_func(model = 'NP_2')
datasize = 300000
q2range = mypdf_SM.getQ2Range()
mkpirange = mypdf_SM.getMkpiRange()
evts_SM = mypdf_SM.genEvt(datasize)
evts_NP1 = mypdf_NP1.genEvt(datasize)
evts_NP2 = mypdf_NP2.genEvt(datasize)

pp_diag = PdfPages("diagnostics.pdf")

plt.figure()
r_bins = plt.hist(evts_SM['q2'], bins = 50)
plt.xlabel("$q^{2}$ [GeV$^{2}$]")
pp_diag.savefig()

plt.figure()
r_bins_mkpi = plt.hist(evts_SM['mkpi'], bins = 50)
plt.xlabel("$m(K\pi)$ [GeV]")
pp_diag.savefig()


mom_bins = {}
fig, ax = plt.subplots(nrows=3, ncols=5, figsize = (28,7))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            break
        if obsList[_i] in evts_SM.columns:
            mom_bins[obsList[_i]] = _col.hist(evts_SM['q2'], weights = evts_SM['%s' % obsList[_i]], bins = 50)
        else:
            mom_bins[obsList[_i]] = _col.hist(evts_SM['q2'], weights = evts_SM['%s' % obsList[_i].replace('S','m')], bins = 50)
        _col.set_xlabel("$q^{2}$ [GeV$^{2}$]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _i += 1
        
mom_bins_mkpi = {}
fig, ax = plt.subplots(nrows=3, ncols=5, figsize = (28,7))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            break
        if obsList[_i] in evts_SM.columns:
            mom_bins_mkpi[obsList[_i]] = _col.hist(evts_SM['mkpi'], weights = evts_SM['%s' % obsList[_i]], bins = 50)
        else:
            mom_bins_mkpi[obsList[_i]] = _col.hist(evts_SM['mkpi'], weights = evts_SM['%s' % obsList[_i].replace('S','m')], bins = 50)
        _col.set_xlabel("$m(K\pi)$ [GeV]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _i += 1
        

fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (25,17))
q2range = mypdf_SM.getQ2Range()
nq2 = 10000
q2vals = np.linspace(q2range[0], q2range[1], nq2)
predVals_SM = {}
predVals_NP1 = {}
predVals_NP2 = {}
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            break
        pred_SM, conversion_SM = mypdf_SM.getPredictions(obsList[_i])
        predVals_SM[obsList[_i]] = conversion_SM*pred_SM(q2vals)
        pred_NP1, conversion_NP1 = mypdf_NP1.getPredictions(obsList[_i])
        predVals_NP1[obsList[_i]] = conversion_NP1*pred_NP1(q2vals)        
        pred_NP2, conversion_NP2 = mypdf_NP2.getPredictions(obsList[_i])
        predVals_NP2[obsList[_i]] = conversion_NP2*pred_NP2(q2vals)             
        _col.plot(r_bins[1][0:50]+0.2, mom_bins[obsList[_i]][0]/r_bins[0], '.', label = 'SM pseudo-data')
        _col.plot(q2vals, predVals_SM[obsList[_i]], label = 'SM prediction')
        _col.plot(q2vals, predVals_NP1[obsList[_i]], label = 'NP1 prediction')
        _col.plot(q2vals, predVals_NP2[obsList[_i]], label = 'NP2 prediction')
        _col.set_xlabel("$q^{2}$ [GeV$^{2}$]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _col.legend(frameon = False)
        _i += 1
pp_diag.savefig()        

fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (25,7))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            break
        _col.plot(r_bins_mkpi[1][0:50]+0.2, mom_bins_mkpi[obsList[_i]][0]/r_bins_mkpi[0], '.')
        _col.set_xlabel("$m(K\pi)$ [GeV]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _i += 1

pp_diag.savefig()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn import neighbors
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d 

# Now set up the neighbours
n_neighbours = 4000
knn = neighbors.KNeighborsRegressor(n_neighbours, weights="uniform", algorithm="kd_tree", n_jobs=8, leaf_size = 1000, p = 2)
T = np.linspace(q2range[0],q2range[1],nq2)[:, np.newaxis]
unbins_SM = {}
unbins_NP1 = {}
unbins_NP2 = {}
print("INFO: Calculating the moments with the nearest neighbours")
# Make a note of the nominal
smear_power = 300
import itertools
unbins_SM[-1] = {}
unbins_NP1[-1] = {}
unbins_NP2[-1] = {}
for _obs in obsList:
    tempFit = knn.fit(np.array(evts_SM["q2"]).reshape(datasize,1), evts_SM[_obs])
    unbins_SM[-1][_obs] = tempFit.predict(T)
    tempFit = knn.fit(np.array(evts_SM["q2"]).reshape(datasize,1), evts_SM[f'{_obs}_{_obs}'])
    unbins_SM[-1][f'var_{_obs}'] = tempFit.predict(T)
    
    tempFit = knn.fit(np.array(evts_NP1["q2"]).reshape(datasize,1), evts_NP1[_obs])
    unbins_NP1[-1][_obs] = tempFit.predict(T)
    tempFit = knn.fit(np.array(evts_NP1["q2"]).reshape(datasize,1), evts_NP1[f'{_obs}_{_obs}'])
    unbins_NP1[-1][f'var_{_obs}'] = tempFit.predict(T)
    
    tempFit = knn.fit(np.array(evts_NP2["q2"]).reshape(datasize,1), evts_NP2[_obs])
    unbins_NP2[-1][_obs] = tempFit.predict(T)
    tempFit = knn.fit(np.array(evts_NP2["q2"]).reshape(datasize,1), evts_NP2[f'{_obs}_{_obs}'])
    unbins_NP2[-1][f'var_{_obs}'] = tempFit.predict(T)

for _p in itertools.combinations_with_replacement(obsList,2):
    tempFit = knn.fit(np.array(evts_SM["q2"]).reshape(datasize,1), evts_SM['{_a}_{_b}'.format(_a = _p[0], _b = _p[1])])
    unbins_SM[-1]['var_{_a}_{_b}'.format(_a = _p[0], _b = _p[1])] = tempFit.predict(T)
    
    tempFit = knn.fit(np.array(evts_NP1["q2"]).reshape(datasize,1), evts_NP1['{_a}_{_b}'.format(_a = _p[0], _b = _p[1])])
    unbins_NP1[-1]['var_{_a}_{_b}'.format(_a = _p[0], _b = _p[1])] = tempFit.predict(T)

    tempFit = knn.fit(np.array(evts_NP2["q2"]).reshape(datasize,1), evts_NP2['{_a}_{_b}'.format(_a = _p[0], _b = _p[1])])
    unbins_NP2[-1]['var_{_a}_{_b}'.format(_a = _p[0], _b = _p[1])] = tempFit.predict(T)

import moments

smoothed_SM = unbins_SM[-1]
smoothed_NP1 = unbins_NP1[-1]
smoothed_NP2 = unbins_NP2[-1]
for _o in obsList:
    smoothed_SM[_o] = gaussian_filter1d(unbins_SM[-1][_o], sigma = smear_power)
    smoothed_NP1[_o] = gaussian_filter1d(unbins_NP1[-1][_o], sigma = smear_power)
    smoothed_NP2[_o] = gaussian_filter1d(unbins_NP2[-1][_o], sigma = smear_power)
    
unbins_SM[-1] = moments.calculateDerivedObs(smoothed_SM)
unbins_SM[-1] = pd.DataFrame(unbins_SM[-1])
unbins_NP1[-1] = moments.calculateDerivedObs(smoothed_NP1)
unbins_NP1[-1] = pd.DataFrame(unbins_NP1[-1])
unbins_NP2[-1] = moments.calculateDerivedObs(smoothed_NP2)
unbins_NP2[-1] = pd.DataFrame(unbins_NP2[-1])

fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (30,7))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            continue
        _col.plot(T, unbins_SM[-1][obsList[_i]])
        _col.set_xlabel("$q^{2}$ [GeV$^{2}$]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _i += 1
pp_diag.savefig()
print("INFO: Making some plots")
# Plot the Si basis observables with uncertainties from bootstrapping
smear_power = 20
fig, ax = plt.subplots(nrows=4, ncols=4, figsize = (30,15))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(obsList):
            break
        _obs = obsList[_i]
        while _obs in ['M1s', 'M1c', 'M2s', 'M2c','M6s', 'M6c', 'FL']:
            _i += 1
            if _i == len(obsList):
                break
            _obs = obsList[_i]
        _col.plot(T, gaussian_filter1d(unbins_SM[-1][_obs], sigma = smear_power))
        _col.fill_between(T.reshape(nq2), gaussian_filter1d(unbins_SM[-1][_obs] - np.sqrt(unbins_SM[-1][f'var_{_obs}']/(n_neighbours-1)), sigma = smear_power),
                                          gaussian_filter1d(unbins_SM[-1][_obs] + np.sqrt(unbins_SM[-1][f'var_{_obs}']/(n_neighbours - 1)), sigma = smear_power), alpha = 0.5)
        if _obs == 'AFB':
            _col.plot(q2vals, 0.75*predVals_SM["S6s"])
        elif _obs not in ["S6c"]:
            _col.plot(q2vals, predVals_SM[_obs])
        elif _obs == "S6c":
            _col.axhline(0, color='orange')
        _col.set_xlabel("$q^{2}$ [GeV$^{2}$]")
        _col.set_ylabel(obs_to_label[obsList[_i]], rotation=0)
        _i += 1
pp = PdfPages('si_observables.pdf')
for _o in obsList:
    plt.figure()
    plt.plot(T, gaussian_filter1d(unbins_SM[-1][_o], sigma = smear_power))
    plt.fill_between(T.reshape(nq2), gaussian_filter1d(unbins_SM[-1][_o] - np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours-1)), sigma = smear_power),
                                      gaussian_filter1d(unbins_SM[-1][_o] + np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours - 1)), sigma = smear_power), alpha = 0.5)
    plt.plot(T, gaussian_filter1d(unbins_NP1[-1][_o], sigma = smear_power))
    plt.fill_between(T.reshape(nq2), gaussian_filter1d(unbins_NP1[-1][_o] - np.sqrt(unbins_NP1[-1][f'var_{_o}']/(n_neighbours-1)), sigma = smear_power),
                                      gaussian_filter1d(unbins_NP1[-1][_o] + np.sqrt(unbins_NP1[-1][f'var_{_o}']/(n_neighbours - 1)), sigma = smear_power), alpha = 0.5)    
    if _o == 'AFB':
        plt.plot(q2vals, 0.75*predVals_SM["S6s"])
        plt.plot(q2vals, 0.75*predVals_NP1["S6s"])
    elif _o not in ["S6c"]:
        plt.plot(q2vals, predVals_SM[_o])
        plt.plot(q2vals, predVals_NP1[_o])
    elif _o == "S6c":
        plt.axhline(0, color='orange')
    plt.xlabel("$q^{2}$ [GeV$^{2}$]", rotation = 0, fontsize = 20)
    plt.ylabel(obs_to_label[_o], rotation = 0, fontsize = 20, labelpad = 12)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    pp.savefig()
    
pp.close()


predValsOpt = {}
_i = 0
for _o in optObsList:
        pred, conversion = mypdf_SM.getPredictions(_o)
        predValsOpt[_o] = conversion*pred(q2vals)

# Plot the Pi basis with uncertainties from bootstrapping
fig, ax = plt.subplots(nrows=2, ncols=4, figsize = (25,7))
_i = 0
for _row in ax:
    for _col in _row:
        if _i == len(optObsList):
            break
        _obs = optObsList[_i]
        while _obs in ['M1s', 'M1c', 'M2s', 'M2c', 'FL']:
            _i += 1
            if _i == len(optObsList):
                break
            _obs = optObsList[_i]
        _col.plot(T, gaussian_filter1d(unbins_SM[-1][_obs], sigma=smear_power))
        _col.fill_between(T.reshape(nq2), gaussian_filter1d(unbins_SM[-1][_obs] - np.sqrt(unbins_SM[-1][f'var_{_obs}']/(n_neighbours-1)), sigma=smear_power),
                                           gaussian_filter1d(unbins_SM[-1][_obs] + np.sqrt(unbins_SM[-1][f'var_{_obs}']/(n_neighbours - 1)), sigma=smear_power), alpha = 0.5)
        if _obs == 'AFB':
            _col.plot(q2vals, 0.75*predValsOpt["S6s"])
        elif _obs not in ["S6c"]:
            _col.plot(q2vals, predValsOpt[_obs])
        _col.set_xlabel("$q^{2}$ [GeV$^{2}$]")
        _col.set_ylabel(obs_to_label[_obs], rotation=0)
        pred_SM, conversion_SM = mypdf_SM.getPredictions(optObsList[_i])
        predVals_SM[optObsList[_i]] = conversion_SM*pred_SM(q2vals)
        
        _i += 1
pp = PdfPages('pi_observables.pdf')
for _o in optObsList:
    plt.figure()
    plt.plot(T, unbins_SM[-1][_o])
    plt.fill_between(T.reshape(nq2), unbins_SM[-1][_o] - np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours-1)),
                                     unbins_SM[-1][_o] + np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours - 1)), alpha = 0.5)
    plt.plot(T, unbins_NP1[-1][_o])
    plt.fill_between(T.reshape(nq2), unbins_NP1[-1][_o] - np.sqrt(unbins_NP1[-1][f'var_{_o}']/(n_neighbours-1)),
                                     unbins_NP1[-1][_o] + np.sqrt(unbins_NP1[-1][f'var_{_o}']/(n_neighbours - 1)), alpha = 0.5)
    if _o == 'AFB':
        plt.plot(q2vals, 0.75*predValsOpt["S6s"])
    elif _o not in ["S6c"]:
        plt.plot(q2vals, predValsOpt[_o])
    elif _o == "S6c":
        plt.axhline(0, color='orange')
    plt.xlabel("$q^{2}$ [GeV$^{2}$]", rotation = 0, fontsize = 20)
    plt.ylabel(obs_to_label[_o], rotation = 0, fontsize = 20, labelpad = 12)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.tight_layout()
    pp.savefig()
pp.close()

print("INFO: Sorting the crossing points")
# 0 crossing :
zero_crossing_dict = {}
pp_zero = PdfPages("zero_crossing.pdf")
q2_cut_range = '(q2vals<8.) & (q2vals>1.1)'
for _o in ["P2", "P5p"]:
    p2_central = interp1d(unbins_SM[-1][_o], T.reshape(nq2))
    p2_lower = interp1d(unbins_SM[-1][_o] - np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours-1)), T.reshape(nq2))
    p2_upper = interp1d(unbins_SM[-1][_o] + np.sqrt(unbins_SM[-1][f'var_{_o}']/(n_neighbours-1)), T.reshape(nq2))
    p2_central_NP1 = interp1d(unbins_NP1[-1][_o], T.reshape(nq2))
    p2_lower_NP1 = interp1d(unbins_NP1 [-1][_o] - np.sqrt(unbins_NP1 [-1][f'var_{_o}']/(n_neighbours-1)), T.reshape(nq2))
    p2_upper_NP1 = interp1d(unbins_NP1 [-1][_o] + np.sqrt(unbins_NP1 [-1][f'var_{_o}']/(n_neighbours-1)), T.reshape(nq2))

    print(p2_lower(0.))
    print(r"%s zero crossing point = $%.3f ^{+%.3f}_{-%.3f}$" % (_o, p2_central(0.), p2_upper(0.), p2_lower(0.)))
    plt.figure()
    plt.plot(q2vals[eval(q2_cut_range)], unbins_SM[-1][_o][eval(q2_cut_range)], label = "SM")
    plt.fill_between(q2vals[eval(q2_cut_range)], unbins_SM[-1][_o][eval(q2_cut_range)] - np.sqrt(unbins_SM[-1][f'var_{_o}'][eval(q2_cut_range)]/(n_neighbours-1)),
                                 unbins_SM[-1][_o][eval(q2_cut_range)] + np.sqrt(unbins_SM[-1][f'var_{_o}'][eval(q2_cut_range)]/(n_neighbours - 1)), alpha = 0.5)

    plt.plot(q2vals[eval(q2_cut_range)], unbins_NP1[-1][_o][eval(q2_cut_range)], label = "NP1")
    plt.fill_between(q2vals[eval(q2_cut_range)], unbins_NP1[-1][_o][eval(q2_cut_range)] - np.sqrt(unbins_NP1[-1][f'var_{_o}'][eval(q2_cut_range)]/(n_neighbours-1)),
                                 unbins_NP1[-1][_o][eval(q2_cut_range)] + np.sqrt(unbins_NP1[-1][f'var_{_o}'][eval(q2_cut_range)]/(n_neighbours - 1)), alpha = 0.5)

    #plt.axvline(p2_central(0), color = "r")
    plt.axvline(p2_lower(0), color = "b", linestyle="dotted", linewidth = 2)
    plt.axvline(p2_upper(0), color = "b", linestyle="dotted", linewidth = 2)
    plt.axvline(p2_lower_NP1(0), color = "orange", linestyle="dotted", linewidth = 2)
    plt.axvline(p2_upper_NP1(0), color = "orange", linestyle="dotted", linewidth = 2)
    plt.axhline(0, color = "black", linewidth = 0.5)
    plt.xlabel("$q^{2}$ [GeV$^{2}$]", rotation = 0, fontsize = 20)
    plt.ylabel(obs_to_label[_o], rotation = 0, fontsize = 20, labelpad = 12)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(frameon = False, fontsize = 15)
    plt.tight_layout()
    pp_zero.savefig()

    zero_crossing_dict[_o+"_SM"] = (p2_central(0), p2_lower(0), p2_upper(0))
    zero_crossing_dict[_o+"_NP1"] = (p2_central_NP1(0), p2_lower_NP1(0), p2_upper_NP1(0))
pp_zero.close()
zero_df = pd.DataFrame(zero_crossing_dict)
zero_df.to_json("zeros.json")

# Some ratios for Quim
pp = PdfPages("quim_ratios.pdf")
print("INFO: Calculating Quim's fancy ratios")
predVals_SM = moments.calculateQuimObs(predVals_SM, T.reshape(nq2), False)
unbins_SM[-1] = moments.calculateQuimObs(unbins_SM[-1], T.reshape(nq2))
unbins_NP1[-1] = moments.calculateQuimObs(unbins_NP1[-1], T.reshape(nq2))
unbins_NP2[-1] = moments.calculateQuimObs(unbins_NP2[-1], T.reshape(nq2))
        
q2_cut_range = '(q2vals<2.) & (q2vals>1.1)'
plt.figure()
plt.plot(q2vals[eval(q2_cut_range)], unbins_SM[-1]["C9_C7"][eval(q2_cut_range)], label = 'SM pseudo-data')
plt.fill_between(q2vals[eval(q2_cut_range)], unbins_SM[-1]["C9_C7"][eval(q2_cut_range)] - np.sqrt(unbins_SM[-1]["var_C9_C7_C9_C7"][eval(q2_cut_range)]/(n_neighbours-1)),
                                             unbins_SM[-1]["C9_C7"][eval(q2_cut_range)] + np.sqrt(unbins_SM[-1]["var_C9_C7_C9_C7"][eval(q2_cut_range)]/(n_neighbours-1)),
                alpha = 0.5)

plt.plot(q2vals[eval(q2_cut_range)], unbins_NP1[-1]["C9_C7"][eval(q2_cut_range)], label = 'NP1 pseudo-data')
plt.fill_between(q2vals[eval(q2_cut_range)], unbins_NP1[-1]["C9_C7"][eval(q2_cut_range)] - np.sqrt(unbins_NP1[-1]["var_C9_C7_C9_C7"][eval(q2_cut_range)]/(n_neighbours-1)),
                                             unbins_NP1[-1]["C9_C7"][eval(q2_cut_range)] + np.sqrt(unbins_NP1[-1]["var_C9_C7_C9_C7"][eval(q2_cut_range)]/(n_neighbours-1)),
                alpha = 0.5)

plt.plot(q2vals[eval(q2_cut_range)], predVals_SM["C9_C7"][eval(q2_cut_range)], label = 'SM prediction')
plt.xlabel("$q^{2}$ [GeV$^{2}$]", fontsize = 15)
plt.ylabel(r"$\frac{C_{9}}{C_{7}}$", rotation = 0, fontsize = 20, labelpad = 12)
plt.legend(fontsize = 15, frameon = False)
plt.tight_layout()
pp.savefig()

plt.figure()
plt.plot(q2vals[eval(q2_cut_range)], unbins_SM[-1]["ff"][eval(q2_cut_range)], label = 'SM pseudo-data')
plt.fill_between(q2vals[eval(q2_cut_range)], unbins_SM[-1]["ff"][eval(q2_cut_range)] - np.sqrt(unbins_SM[-1]["var_ff_ff"][eval(q2_cut_range)]/(n_neighbours-1)),
                                             unbins_SM[-1]["ff"][eval(q2_cut_range)] + np.sqrt(unbins_SM[-1]["var_ff_ff"][eval(q2_cut_range)]/(n_neighbours-1)),
                 alpha = 0.5)

plt.plot(q2vals[eval(q2_cut_range)], unbins_NP1[-1]["ff"][eval(q2_cut_range)], label = 'NP1 pseudo-data')
plt.fill_between(q2vals[eval(q2_cut_range)], unbins_NP1[-1]["ff"][eval(q2_cut_range)] - np.sqrt(unbins_NP1[-1]["var_ff_ff"][eval(q2_cut_range)]/(n_neighbours-1)),
                                             unbins_NP1[-1]["ff"][eval(q2_cut_range)] + np.sqrt(unbins_NP1[-1]["var_ff_ff"][eval(q2_cut_range)]/(n_neighbours-1)),
                 alpha = 0.5)

plt.plot(q2vals[eval(q2_cut_range)], predVals_SM["ff"][eval(q2_cut_range)], label = 'SM prediction')
plt.xlabel("$q^{2}$ [GeV$^{2}$]", fontsize = 15)
plt.ylabel(r"$\frac{\xi_\|}{\xi_\perp}$", rotation = 0, fontsize = 20, labelpad = 10)
plt.legend(loc = 'upper left', fontsize = 15, frameon = False)
plt.tight_layout()
pp.savefig()
pp.close()

print("INFO: All done")
