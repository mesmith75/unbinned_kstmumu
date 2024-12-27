import numpy as np
import pandas as pd
import itertools

class evt:
    
    def __init__(self):
        self.q2 =  None
        self.ctl = None
        self.ctk = None
        self.phi = None
        self.mkpi = None
        
        self.M1s = None #1. - _ctk**2
        self.M1c = None #_ctl**2
        self.M2s = None #(1. - _ctk**2) * (2.*_ctl**2 - 1.)
        self.M2c = None #_ctk**2 * (2.*_ctl**2 - 1.)
        
        self.M3 = None #(1. - _ctk**2)*(1. - _ctl**2)*np.cos(2.*_phi)
        self.M5 = None #np.sqrt(1. - _ctl*_ctl) * 2. * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.M4 = None #2.*_ctl * np.sqrt(1. - _ctl*_ctl) * 2 * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.M7 = None #2. * _ctk * np.sqrt(1. - _ctk*_ctk)*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.M8 = None #2. * _ctk * np.sqrt(1. - _ctk*_ctk)*2.*_ctl*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.M9 = None #(1. - _ctk**2)*(1. - _ctl**2)*np.sin(2.*_phi)
        
        self.S3 = None #3.125*(1. - _ctk**2)*(1. - _ctl**2)*np.cos(2.*_phi)
        self.S5 = None #2.5 * np.sqrt(1. - _ctl*_ctl) * 2. * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.S4 = None #3.125*2*_ctl * np.sqrt(1. - _ctl*_ctl) * 2 * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.AFB = None #0.75*2.5*(1. - _ctk**2)*_ctl
        self.S7 = None #2.5*2 * _ctk * np.sqrt(1. - _ctk*_ctk)*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.S8 = None #3.125*2 * _ctk * np.sqrt(1. - _ctk*_ctk)*2.*_ctl*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.S9 = None #3.125*(1. - _ctk**2)*(1. - _ctl**2)*np.sin(2.*_phi)
        self.M6s = None #(1. = _ctk**2)*_ctl
        self.M6c = None #_ctk**2 * _ctl
        self.FL = None # 2. - 2.5*(1. - _ctl**2)
        self.S1s = None
        self.S1c = None
        self.S2s = None
        self.S2c = None
        self.S6s = None
        self.S6c = None
        
        for _p in itertools.combinations_with_replacement(['1s','1c','2s','2c','3','4','5','6s','6c','7','8','9','FL','AFB'],2):
            if _p[0] not in ['FL', 'AFB'] and _p[1] not in ['FL','AFB']:
                exec('self.M{_a}_M{_b} = None'.format(_a = _p[0], _b = _p[1]))
                exec('self.S{_a}_S{_b} = None'.format(_a = _p[0], _b = _p[1]))
            elif _p[0] in ['FL', 'AFB'] and _p[1] not in ['FL', 'AFB']:
                exec('self.{_a}_S{_b} = None'.format(_a = _p[0], _b = _p[1]))
            elif _p[1] in ['FL', 'AFB'] and _p[0] not in ['FL', 'AFB']:
                exec('self.S{_a}_{_b} = None'.format(_a = _p[0], _b = _p[1]))
            else:
                exec('self.{_a}_{_b} = None'.format(_a = _p[0], _b = _p[1]))
    
    def __init__(self, _q2, _ctl, _ctk, _phi, _mkpi):
        self.q2 = _q2
        self.ctl = _ctl
        self.ctk = _ctk
        self.phi = _phi
        self.mkpi = _mkpi
        
        self.M1s = 1. - _ctk**2
        self.M1c = _ctk**2
        self.M2s = (1. - _ctk**2) * (2.*_ctl**2 - 1.)
        self.M2c = _ctk**2 * (2.*_ctl**2 - 1.)
        
        self.M3 = (1. - _ctk**2)*(1. - _ctl**2)*np.cos(2.*_phi)
        self.M5 =  np.sqrt(1. - _ctl*_ctl) * 2. * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.M4 = 2.*_ctl * np.sqrt(1. - _ctl*_ctl) * 2 * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.M7 = 2. * _ctk * np.sqrt(1. - _ctk*_ctk)*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.M8 = 2. * _ctk * np.sqrt(1. - _ctk*_ctk)*2.*_ctl*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.M9 = (1. - _ctk**2)*(1. - _ctl**2)*np.sin(2.*_phi)
        
        self.S3 = 3.125*(1. - _ctk**2)*(1. - _ctl**2)*np.cos(2.*_phi)
        self.S5 = 2.5 * np.sqrt(1. - _ctl*_ctl) * 2. * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.S4 = 3.125*2*_ctl * np.sqrt(1. - _ctl*_ctl) * 2 * _ctk * np.sqrt(1. - _ctk*_ctk) * np.cos(_phi)
        self.AFB = 0.75*2.5*(1. - _ctk**2)*_ctl
        self.S7 = 2.5*2 * _ctk * np.sqrt(1. - _ctk*_ctk)*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.S8 = 3.125*2 * _ctk * np.sqrt(1. - _ctk*_ctk)*2.*_ctl*np.sqrt(1. - _ctl*_ctl)*np.sin(_phi)
        self.S9 = 3.125*(1. - _ctk**2)*(1. - _ctl**2)*np.sin(2.*_phi)
        self.M6s = (1. - _ctk**2)*_ctl
        self.M6c = _ctk**2 * _ctl
        self.FL = 2. - 2.5*(1. - _ctl**2)
        
        self.S1s = (1./16.) * (21.*self.M1s - 14.*self.M1c + 15.*self.M2s - 10.*self.M2c)
        self.S1c = (1./8.) * (-7.*self.M1s + 28.*self.M1c - 5.*self.M2s + 20.*self.M2c)
        self.S2s = (1./16.) * (15.*self.M1s - 10.*self.M1c + 45.*self.M2s - 30.*self.M2c)
        self.S2c = (1./8.) * (-5.*self.M1s + 20.*self.M1c - 15.*self.M2s + 60.*self.M2c)
        
        self.S6s = 3.*self.M6s - 2.*self.M6c
        self.S6c = 8.*self.M6c - 2.*self.M6s

        for _p in itertools.combinations_with_replacement(['1s','1c','2s','2c','3','4','5','6s','6c','7','8','9','FL','AFB'],2):
            if _p[0] not in ['FL', 'AFB'] and _p[1] not in ['FL','AFB']:
                exec('self.M{_a}_M{_b} = self.M{_a}*self.M{_b}'.format(_a = _p[0], _b = _p[1]))
                exec('self.S{_a}_S{_b} = self.S{_a}*self.S{_b}'.format(_a = _p[0], _b = _p[1]))
            elif _p[0] in ['FL', 'AFB'] and _p[1] not in ['FL', 'AFB']:
                exec('self.{_a}_S{_b} = self.{_a}*self.S{_b}'.format(_a = _p[0], _b = _p[1]))
            elif _p[1] in ['FL', 'AFB'] and _p[0] not in ['FL', 'AFB']:
                exec('self.S{_a}_{_b} = self.S{_a}*self.{_b}'.format(_a = _p[0], _b = _p[1]))
            else:
                exec('self.{_a}_{_b} = self.{_a}*self.{_b}'.format(_a = _p[0], _b = _p[1]))
        
    def _allvars(self):
        return self.__dict__.keys()
        #return ['q2', 'ctl', 'ctk', 'phi', 'm1s', 'm1c', 'm2s', 'm2c', 'm3','m4','m5','m6s','m6c','m7','m8','m9',
        #        's3','s4','s5','AFB','s7','s8','s9', 'FL']
        
    def extend(self, _evt):
        if self.q2:
            for _v in self._allvars():
                self._allvars = np.concatenate([eval(f"self.{_v}"), eval(f"_evt.{_v}")])
        else:
            for _v in self._allvars():
                eval(f"self.{_v} = _evt.getVar({_v})")
    
    def getVar(self, varname):
        if hasattr(self, varname):
            return eval(f"self.{varname}")
        else:
            raise Exception

    def getAllVars(self):
        return self.__dict__.keys()
    
    def getNevents(self):
        if not self.q2:
            return 0
        elif isinstance(self.q2, int) or isinstance(self.q2, float):
            return 1
        else:
            return len(self.q2)
        
    def getFrame(self):
        retFrame = pd.DataFrame(self.__dict__)
        return retFrame