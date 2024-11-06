import pandas as pd
import uproot
import numpy as np
import cmath
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from collections import OrderedDict
import flavio
import flavio.plots
from flavio.statistics import probability
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
#Supress the warnings about QCDF corrections above 6 being unreliable
import warnings
warnings.filterwarnings("ignore")
import json
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pp = PdfPages("final_fit.pdf")

import random
def createMeasurement(name, observables, covariance, q2min, q2max, lumi_scale = 1.):
  m = flavio.classes.Measurement(name)
  observablesspec = []
  central_values  = []
  for obs in observables.values():
    observablesspec.append(('<{}>(B0->K*mumu)'.format(obs['name']), q2min, q2max))
    central_values.append(obs['value'])
  myerrs = np.sqrt(np.diagonal(covariance))
  if lumi_scale > 1.:
    mycorr = pd.DataFrame(covariance).divide(myerrs, axis=0)
    mycorr = mycorr.divide(myerrs, axis=1)
    myerrs = myerrs/np.sqrt(lumi_scale)
    newcov = mycorr.multiply(myerrs, axis=0)
    newcov = newcov.multiply(myerrs, axis=1)
    covariance = newcov.to_numpy()
    myerrs = np.sqrt(np.diagonal(covariance))
  covariance = np.zeros(shape=(8,8))
  new_centrals = [0. for _i in range(8)]
  for _i in range(8):
        covariance[_i][_i] = myerrs[_i]**2
        new_centrals[_i] = random.gauss(central_values[_i], myerrs[_i])
  
  m.add_constraint(observablesspec, probability.MultivariateNormalDistribution(new_centrals, covariance))

  return (m, new_centrals, myerrs)

wc_np = flavio.WilsonCoefficients()
wc_np.set_initial({'C9_bsmumu': -0.5, 'C10_bsmumu': 0.}, scale=4.8, eft='WET', basis='flavio')
stdbinList = [('0.1', '0.98'), ('1.1', '2.5'), ('2.5', '4'), ('4', '6'), ('6', '8'), ('15', '17'), ('17', '19'), ('11', '12.5')]
q2_vals = [0.54, 1.8, 3.25, 5., 7., 16., 18., 11.75]

obs_si = ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']
stdPreds_si = {}
for _obs in obs_si:
    stdPreds_si[_obs] = {'val' : [], 'err' : []}
    for _bin in stdbinList:
        tbin = (float(_bin[0]), float(_bin[1]))
        stdPreds_si[_obs]['val'].append(flavio.np_prediction('<%s>(B0->K*mumu)' % _obs, wc_np, *tbin))
        stdPreds_si[_obs]['err'].append(flavio.np_uncertainty('<%s>(B0->K*mumu)' % _obs, wc_np, *tbin))

stdMeas_si = {}
lhcb_meas = []
lhcb_meas_obj = []
for mname, mobj in flavio.Measurement.instances.items():
    if 'B->K*mumu' in mobj.name:
            lhcb_meas.append(mname)
            lhcb_meas_obj.append(mobj)

my_covs = {}
my_vals = {}
_i = 0
for _b in stdbinList:
    for _m in lhcb_meas_obj:
        if _m.name == 'LHCb B->K*mumu 2020 S %s-%s' % (_b[0], _b[1]):
            my_covs[_i] = _m.get_yaml_dict()['constraints'][0]['values']['covariance']
        my_vals[_i] = {}
        for _obs in obs_si:
            my_vals[_i][_obs] = {'name' : _obs, 'value' : stdPreds_si[_obs]['val'][_i]}
            
    _i +=1

published_meas = {}
current_meas = {}
future_meas = {}
new_centrals = {}
new_centrals_current = {}
new_centrals_future = {}
published_errs = {}
current_errs = {}
future_errs = {}
_i = 1
for _b in stdbinList[1:4]:
    published_meas[_i], new_centrals[_i], published_errs[_i] = createMeasurement('published {} {}'.format(_b[0], _b[1]), my_vals[_i], my_covs[_i], float(_b[0]), float(_b[1]), lumi_scale = 1.)
    current_meas[_i], new_centrals_current[_i], current_errs[_i] = createMeasurement('current {} {}'.format(_b[0], _b[1]), my_vals[_i], my_covs[_i], float(_b[0]), float(_b[1]), lumi_scale = 2.)
    future_meas[_i], new_centrals_future[_i], future_errs[_i] = createMeasurement('future {} {}'.format(_b[0], _b[1]), my_vals[_i], my_covs[_i], float(_b[0]), float(_b[1]), lumi_scale = 15.)
    _i += 1

plt.figure()
plt.errorbar([_q2 - 0.2 for _q2 in q2_vals[1:4]], [new_centrals[_i][3] for _i in range(1,4)], yerr = [published_errs[_i][3] for _i in range(1,4)], fmt = '.')
plt.errorbar(q2_vals[1:4], [new_centrals_current[_i][3] for _i in range(1,4)], yerr = [current_errs[_i][3] for _i in range(1,4)], fmt = '.')
plt.errorbar([_q2 + 0.2 for _q2 in q2_vals[1:4]], [new_centrals_future[_i][3] for _i in range(1,4)], yerr = [future_errs[_i][3] for _i in range(1,4)], fmt = '.')

import flavio.statistics.likelihood
from wilson import Wilson
all_binned_obs = []
for _m in published_meas.values():
    all_binned_obs.extend(_m.all_parameters)
print(all_binned_obs)
FL = flavio.statistics.likelihood.FastLikelihood(name='published',
                       par_obj = flavio.default_parameters,
                       fit_parameters = [],
                       nuisance_parameters = [],# flavio.default_parameters.all_parameters,
                       observables=all_binned_obs,
                       include_measurements=[m.name for m in published_meas.values()])
par = flavio.parameters.default_parameters.get_central_all()
def FLL(x):
    Re_C9, Re_C10 = x
    w = Wilson({'C10_bsmumu': Re_C10, 'C9_bsmumu': Re_C9},
                scale=4.8,
                eft='WET', basis='flavio')
    return FL.log_likelihood(par, w)


def chi2(x):
    return -2*FLL(x)
FL.make_measurement(300)

FL_current = flavio.statistics.likelihood.FastLikelihood(name='current',
                       par_obj = flavio.default_parameters,
                       fit_parameters = [],
                       nuisance_parameters = [],#flavio.default_parameters.all_parameters,
                       observables=all_binned_obs,
                       include_measurements=[m.name for m in current_meas.values()])
def FLL_current(x):
    Re_C9, Re_C10 = x
    w = Wilson({'C10_bsmumu': Re_C10, 'C9_bsmumu': Re_C9},
                scale=4.8,
                eft='WET', basis='flavio')
    return FL_current.log_likelihood(par, w)

FL_current.make_measurement(300)

FL_future = flavio.statistics.likelihood.FastLikelihood(name='future',
                       par_obj = flavio.default_parameters,
                       fit_parameters = [],
                       nuisance_parameters = [],#flavio.default_parameters.all_parameters,
                       observables=all_binned_obs,
                       include_measurements=[m.name for m in future_meas.values()])
def FLL_future(x):
    Re_C9, Re_C10 = x
    w = Wilson({'C10_bsmumu': Re_C10, 'C9_bsmumu': Re_C9},
                scale=4.8,
                eft='WET', basis='flavio')
    return FL_future.log_likelihood(par, w)

FL_future.make_measurement(300)
"""
steps=20
plt.figure(figsize=(4.2,4.2))
print('Profile binned 1 sigma')
#flavio.plots.likelihood_contour(FLL, -1.0, 0.1, -0.8, 0.5, n_sigma=1, steps=steps, col=1,
#                      interpolation_factor=3, label=r'published', color='green')
flavio.plots.likelihood_contour(FLL_current, -1.0, 0.1, -0.8, 0.5, n_sigma=1, steps=steps, col=2,
                      interpolation_factor=3, label=r'current', color='red')
flavio.plots.likelihood_contour(FLL_future, -1.0, 0.1, -0.8, 0.5, n_sigma=1, steps=steps, col=3,
                      interpolation_factor=3, label=r'future')


plt.axhline(0, c='k', lw=0.2)
plt.axvline(0, c='k', lw=0.2)
plt.legend()
plt.xlabel(r'$\text{Re}\, C_9^\text{NP}$')
plt.ylabel(r'$\text{Re}\, C_{10}^\text{NP}$')
plt.tight_layout()
plt.savefig("nonsens.pdf")
"""
import json
with open('means.json') as fp:
    mean_vals = json.load(fp)
with open('sigmas.json') as fp:
    sigma_vals = json.load(fp)
with open('covs.json') as fp:
    points = json.load(fp)
unbinned_q2 = np.linspace(1.1,6.0,10000)

def is_pos_def(x):
     return np.all(np.linalg.eigvals(x) > 0)

def createUnbinnedMeasurement(name, means, errs, covs, q2, sep):
  m = flavio.classes.Measurement(name)
  observablesspec = []
  central_values  = []
  covariance_maker = []
  these_means = []
  nos = [_i for _i in range(10000)]
  labels = []
  for obs in ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']:
      for q2 in unbinned_q2[::sep]:
        observablesspec.append(('{}(B0->K*mumu)'.format(obs), q2))
#        observablesspec.append(('<{}>(B0->K*mumu)'.format(obs), q2-0.000245, q2+0.000245))
      these_means.extend(means[obs][::sep])
      for _n in nos[::sep]:
          covariance_maker.append(covs[obs][str(_n)])

  covariance = np.cov(covariance_maker)

  if not is_pos_def(covariance):
    #Fix machine non zeros
    covariance += 1.e-11 * np.eye(*covariance.shape)

  errs = np.sqrt(np.diag(covariance))
  corr_matrix = pd.DataFrame(covariance)
  corr_matrix = corr_matrix.divide(errs, axis = 0)
  corr_matrix = corr_matrix.divide(errs, axis = 1)

  err_scale = errs*2.

#  new_corr = np.ones(corr_matrix.shape)
#  corr_matrix = pd.DataFrame(new_corr)

  new_cov = corr_matrix.multiply(err_scale, axis=0)
  new_cov = new_cov.multiply(err_scale, axis=1)

  if not is_pos_def(new_cov):
    #Fix machine non zeros
    new_cov += 1.e-11 * np.eye(*new_cov.shape)

    #observablesspec.append(('<{}>(B0->K*mumu)'.format(obs), q2-0.000245, q2+0.000245))
#  covariance = np.zeros(shape=(8,8))
#  for _i in range(8):
#        covariance[_i][_i] = errs[_i]**2
  m.add_constraint(observablesspec, probability.MultivariateNormalDistribution(these_means, new_cov))

  return (m, these_means, new_cov, unbinned_q2[::sep])

unbinned_meas  = {}
unbinned_meas[0], central_vals, cov_matrix, unbinned_q2_vals = createUnbinnedMeasurement('unbinned', mean_vals, sigma_vals, points, 0.01, 400)
_i = 0
#for _q2 in unbinned_q2:
#    tmeans = [mean_vals[_obs][_i] for _obs in ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']]
#    terrs = [sigma_vals[_obs][_i] for _obs in ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']]
#    unbinned_meas[_i], _, _ = createUnbinnedMeasurement('unbinned {}'.format(_q2), tmeans, terrs, _q2)
#    _i += 1



things = []
for _m in list(unbinned_meas.values()):
    things.extend(_m.all_parameters)

print("INFO: things: ")
print(things)

FL_unbinned = flavio.statistics.likelihood.FastLikelihood(name='unbinned',
                       par_obj = flavio.default_parameters,
                       fit_parameters = [],
                       nuisance_parameters = [],#flavio.default_parameters.all_parameters,
                       observables=things,
                       include_measurements=[m.name for m in list(unbinned_meas.values())])
def FLL_unbinned(x):
    Re_C9, Re_C10 = x
    w = Wilson({'C10_bsmumu': Re_C10, 'C9_bsmumu': Re_C9},
                scale=4.8,
                eft='WET', basis='flavio')
    return FL_unbinned.log_likelihood(par, w)

FL_unbinned.make_measurement(300)

steps=40
plt.figure(figsize=(4.2,4.2))
print('Profile binned 1 sigma')
#flavio.plots.likelihood_contour(FLL, -2.0, 2.0, -1.5, 1.5, n_sigma=1, steps=steps, col=1,
#                      interpolation_factor=3, label=r'published', color='green')
#flavio.plots.likelihood_contour(FLL_current, -2.0, 2.0, -1.5, 1.5, n_sigma=1, steps=steps, col=1,
#                      interpolation_factor=3, label=r'current', color='red')
flavio.plots.likelihood_contour(FLL_future, -1.0, 0., -1.0, 1.0, n_sigma=2, steps=steps, col=2,
                      interpolation_factor=3, label=r'Projected 30fb$^{-1}$, $q^{2}$ binned')
flavio.plots.likelihood_contour(FLL_unbinned, -1.0, 0., -1.0, 1.0, n_sigma=2, steps=steps, col=1,
                      interpolation_factor=3, label=r'Projected 30fb$^{-1}$, $q^{2}$ unbinned')
plt.axhline(0, c='k', lw=0.2)
plt.axvline(0, c='k', lw=0.2)
plt.legend()
plt.xlabel(r'$\text{Re}\, C_9^\text{NP}$')
plt.ylabel(r'$\text{Re}\, C_{10}^\text{NP}$')
plt.tight_layout()
pp.savefig()

_o = 0
_q = 0
all_errs = np.sqrt(np.diagonal(cov_matrix))
pandas_cov = pd.DataFrame(cov_matrix)
cor_matrix = pandas_cov.divide(all_errs, axis=1)
cor_matrix = cor_matrix.divide(all_errs, axis=0)

for obs in ['FL', 'S3', 'S4', 'S5', 'AFB', 'S7', 'S8', 'S9']:
    plt.figure()
    plt.errorbar([_q2 + 0.2 for _q2 in q2_vals[1:4]], [new_centrals_future[_i][_q] for _i in range(1,4)], yerr = [future_errs[_i][_q] for _i in range(1,4)], fmt = '.', label = 'future binned')
    plt.errorbar(unbinned_q2_vals, central_vals[_o : _o + len(unbinned_q2_vals)], yerr = all_errs[_o : _o + len(unbinned_q2_vals)], fmt = '.', label = 'unbinned points')
    plt.ylabel(obs)
    plt.xlabel('$q^{2}$ [GeV$^{2}$]')
    _q += 1
    plt.legend()
    plt.tight_layout()
    pp.savefig()

    sub_cor = cor_matrix.iloc[_o:_o+len(unbinned_q2_vals),_o:_o+len(unbinned_q2_vals)]
    plt.figure()
    mask = np.tril(np.ones_like(sub_cor, dtype=bool))
    sub_cor = sub_cor.where(mask)
    sns.heatmap(sub_cor, vmax=.9, vmin=-.9, center=0, cmap="RdBu",
        square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False, fmt='.2f')
    plt.tight_layout()
    pp.savefig()
    _o += len(unbinned_q2_vals)


pp.close()
