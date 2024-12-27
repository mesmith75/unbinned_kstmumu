import numpy as np
import pandas as pd
import scipy.interpolate as spi
import json
from math import pi
import cmath
from data_classes import *
from functools import reduce

prefactor = 9./(32.*pi)
nprect = np.vectorize(cmath.rect)

class angular_func:
    def __init__(self, model = 'SM'):
        self.maxProb = 0.
        
        if model not in ['SM','NP_1', 'NP_2']:
            raise Exception

        predictions_json = None
        with open(f'predictions_toy_mc_{model}.json', "r") as _f:
            predictions_json = json.load(_f)

        self.preds = {}
        for _k, _v in predictions_json.items():
            if _k in ['q2', 'WC_NP']:
                continue
            self.preds[_k] = spi.interp1d(predictions_json['q2'], _v)
        self.q2_min = min(predictions_json['q2'])
        self.q2_max = max(predictions_json['q2'])
        self.mkpi_min = 0.75
        self.mkpi_max = 1.1
        self.q0 = self.CalculateMomentum(0.895, 0.0473, 0.139570)        
        self.maxProb = self.genEvt(10000, True)
    
    def evalPDF(self, _evt):
        pwave = 0
        for _v in ['1s', '1c', '2s', '2c', '3', '4', '5', '6s', '6c', '7', '8', '9']:
            pwave += self.preds[f'S{_v}'](_evt.q2) * eval(f'_evt.M{_v}')
        pwave *= prefactor
        pwave *= self.preds['BR'](_evt.q2)
        pwave *= self.evalMkpi(_evt.mkpi)
        return pwave
   
    def genEvt(self, nToGen = 10, getMaxProb = False):
        
        evtsToReturn = pd.DataFrame()
        while len(evtsToReturn) < nToGen:
            ctlGen = np.random.uniform(-1., 1., nToGen)
            ctkGen = np.random.uniform(-1., 1., nToGen)
            phiGen = np.random.uniform(-pi, pi, nToGen)
            q2Gen = np.random.uniform(self.q2_min, self.q2_max, nToGen)
            mkpiGen = np.random.uniform(self.mkpi_min, self.mkpi_max, nToGen)
        
            _evt = evt(q2Gen, ctlGen, ctkGen, phiGen, mkpiGen)
            pdfEval = self.evalPDF(_evt)
        
            if getMaxProb:
                return max(pdfEval)
            else:
                randProb = np.random.uniform(0., self.maxProb, nToGen)
                evtsToReturn = pd.concat([evtsToReturn, _evt.getFrame()[pdfEval>randProb]], ignore_index=True)
        evtsToReturn = evtsToReturn.sample(nToGen)
            
        return evtsToReturn

    def getPredictions(self, var):
        prefactor = 1.
        if var in ['AFB']:
            prefactor = -1.
        return (self.preds[var], prefactor)
    
    def getQ2Range(self):
        return (self.q2_min, self.q2_max)
    
    def getMkpiRange(self):
        return(self.mkpi_min, self.mkpi_max)

    def CalculateMomentum(self, m, m1, m2):
        add_12 = m1 + m2
        sub_12 = m1 - m2
        return np.sqrt((m*m - add_12*add_12)*(m*m - sub_12*sub_12))/(2.0*m)

    def BlattWeisskopfFormFactor(self, q, q_J, r, J):
        r2 = r*r;
        q2 = q*q;
        q_J2 = q_J*q_J;
        if J == 0:
            return 1.0
        elif J == 1:
            return np.sqrt((1 + r2*q_J2) / (1 + r2*q2))
        elif J == 2:
            return np.sqrt((9 + 3*r2*q_J2 + r2*r2*q_J2*q_J2) / (9 + 3*r2*q2 + r2*r2*q2*q2))

    def RelativisticSpinBreitWigner(self, m, mass_J, width_J, r, J, q, q_J):
        # calculate some helpers
        q_ratio = q / q_J
        m_ratio = mass_J / m
        width = width_J*pow(q_ratio, 2*J + 1)*m_ratio*self.BlattWeisskopfFormFactor(q,q_J,r,J)*self.BlattWeisskopfFormFactor(q,q_J,r,J)
        tanPhi = (mass_J * width)/(mass_J*mass_J - m*m)
        phi = np.arctan(tanPhi)
        myAmp = nprect(np.sin(phi), phi)
        return myAmp

    
    def evalMkpi(self, mkpi):
        q = self.CalculateMomentum(mkpi, 0.0473, 0.139570)
        amp = self.RelativisticSpinBreitWigner(mkpi, 0.892, 0.0473, 1.6, 1, q, self.q0) * self.BlattWeisskopfFormFactor(q, self.q0, 1.6,1)
        return np.abs(amp)**2
        