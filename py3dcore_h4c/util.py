# -*- coding: utf-8 -*-

"""util.py

3DCORE utility functions.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from matplotlib.dates import  DateFormatter
import datetime
from datetime import timedelta
import time
import pickle
import cdflib
import seaborn as sns

import astropy
import astropy.constants as const
import astropy.units as u
import sunpy
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.time import parse_time
import astrospice

import heliopy.data.spice as spicedata
import heliopy.spice as spice

import logging

import numba 
import numpy 
import py3dcore_h4c 
import sys 
import os

from numba import guvectorize
from scipy.signal import detrend, welch
from typing import Sequence, Tuple

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from astropy.constants import au
AU=au.value/1e3 #in km

from astropy.constants import R_sun
RS=R_sun.value*1e-3 #in km


@guvectorize([
    "void(float32[:, :], float32[:, :])",
    "void(float64[:, :], float64[:, :])"],
    '(n, n) -> (n, n)')
def ldl_decomp(mat: np.ndarray, res: np.ndarray) -> None:
    """Computes the LDL decomposition, returns L.sqrt(D).
    """
    n = mat.shape[0]

    _lmat = np.identity(n)
    _dmat = np.zeros((n, n))

    for i in range(n):
        _dmat[i, i] = mat[i, i] - np.sum(_lmat[i, :i]**2 * np.diag(_dmat)[:i])

        for j in range(i + 1, n):
            if _dmat[i, i] == 0:
                _lmat[i, i] = 0
            else:
                _lmat[j, i] = mat[j, i] - np.sum(_lmat[j, :i] * _lmat[i, :i] * np.diag(_dmat)[:i])
                _lmat[j, i] /= _dmat[i, i]

    res[:] = np.dot(_lmat, np.sqrt(_dmat))[:]


def mag_fft(dt: Sequence[datetime.datetime], bdt: np.ndarray, sampling_freq: int) -> Tuple[np.ndarray, np.ndarray]:
    
    """Computes the mean power spectrum distribution from a magnetic field measurements over all three vector components.

    Note: Assumes that P(k) is the same for all three vector components.
    
    Arguments:
        dt                datetime objects (time axis)
        bdt               according magnetic field data
        sampling_freq     Sampling frequency of data
        
    Returns:
        fF                array containing the sample frequencies
        fS                mean power spectrum 
    """
    
    n_s = int(((dt[-1] - dt[0]).total_seconds() / 3600) - 1) 
    n_perseg = np.min([len(bdt), 256]) # gives the length of the dataframe if it is larger than 256, otherwise 256

    # the scipy function detrend removes any linear trend along a given axis
    
    p_bX = detrend(bdt[:, 0], type="linear", bp=n_s) 
    p_bY = detrend(bdt[:, 1], type="linear", bp=n_s)
    p_bZ = detrend(bdt[:, 2], type="linear", bp=n_s)

    # the scipy function welch estimates the power spectral density using Welch’s method for a given length of each segment and the sampling frequency. It returns an array of sample frequencies and the power spectral density.
    
    _,  wX = welch(p_bX, fs=1 / sampling_freq, nperseg=n_perseg)
    _,  wY = welch(p_bY, fs=1 / sampling_freq, nperseg=n_perseg)
    wF, wZ = welch(p_bZ, fs=1 / sampling_freq, nperseg=n_perseg)

    wS = (wX + wY + wZ) / 3 # the power spectrum is averages across all three components

    # convert P(k) into suitable form for fft
    fF = np.fft.fftfreq(len(p_bX), d=sampling_freq) # returns array of length len(p_bX) containing the sample frequencies
    fS = np.zeros((len(fF)))

    for i in range(len(fF)):
        k = np.abs(fF[i])
        fS[i] = np.sqrt(wS[np.argmin(np.abs(k - wF))]) 

    return fF, fS


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    _numba_set_random_seed(seed)


@numba.njit
def _numba_set_random_seed(seed: int) -> None:
    np.random.seed(seed)

    
    
def cdftopickle(magpath, swapath, sc):
    
    '''creating a pickle file from cdf'''
    
    if sc == 'solo':
        fullname = 'solar orbiter'
    if sc == 'psp':
        fullname = 'Parker Solar Probe'
    if sc == 'wind':
        fullname = 'Wind'
    
    timep = np.zeros(0,dtype=[('time',object)])
    den = np.zeros(0)
    temp = np.zeros(0)
    vr = np.zeros(0)
    vt = np.zeros(0)
    vn = np.zeros(0)
        
    if os.path.exists(swapath):
        ll_path = swapath
    
        files = os.listdir(ll_path)
        files.sort()
        llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    

        for i in np.arange(0,len(llfiles)):
            p1 = cdflib.CDF(llfiles[i])

            den1 = p1.varget('N')
            speed1 = p1.varget('V_RTN')
            temp1 = p1.varget('T')

            vr1 = speed1[:, 0]
            vt1 = speed1[:, 1]
            vn1 = speed1[:, 2]

            vr = np.append(vr1, vr)
            vt = np.append(vt1, vt)
            vn = np.append(vn1, vn)


            temp = np.append(temp1, temp)
            den = np.append(den1, den)


            time1 = p1.varget('EPOCH')
            t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time1, format=None)).datetime
            timep = np.append(timep, t1)

            temp = temp*(1.602176634*1e-19) / (1.38064852*1e-23) # from ev to K 

    ll_path = magpath

    files = os.listdir(ll_path)
    files.sort()
    llfiles = [os.path.join(ll_path, f) for f in files if f.endswith('.cdf')]
    
    br1 = np.zeros(0)
    bt1 = np.zeros(0)
    bn1 = np.zeros(0)
    time1 = np.zeros(0,dtype=[('time',object)])
    
    for i in np.arange(0,len(llfiles)):
        m1 = cdflib.CDF(llfiles[i])
        
        if sc == 'solo':
            print('solo cdf')
            
            try:
                b = m1.varget('B_RTN')
            except:
                time = m1.varget('EPOCH')

        elif sc == 'psp':
            print('psp cdf')
            b = m1.varget('psp_fld_l2_mag_RTN_1min')
            time = m1.varget('epoch_mag_RTN_1min')
        #except:
        #    b = m1.varget('psp_fld_l2_mag_RTN_1min')

        elif sc =='wind':
            print('wind cdf')
            b = m1.varget('BRTN')
            time = m1.varget('Epoch')

        else:
            print('No data from cdf file')
            
        br = b[:, 0]
        bt = b[:, 1]
        bn = b[:, 2]

        br1 = np.append(br1, br)
        bt1 = np.append(bt1, bt)
        bn1 = np.append(bn1, bn)

        #try:
        #    time = m1.varget('EPOCH')
        #except:
        #    time = m1.varget('epoch_mag_RTN_1min')
            
        t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time, format=None)).datetime
        time1 = np.append(time1,t1)
        
    starttime = time1[0].replace(hour = 0, minute = 0, second=0, microsecond=0)
    endtime = time1[-1].replace(hour = 0, minute = 0, second=0, microsecond=0)
    
    time_int = []
    while starttime < endtime:
        time_int.append(starttime)
        starttime += timedelta(minutes=1)


    time_int_mat = mdates.date2num(time_int)
    time1_mat = mdates.date2num(time1)
    timep_mat = mdates.date2num(timep)  
    
    if os.path.exists(swapath):
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float),('vx', float),('vy', float),('vz', float),('vt', float),('tp', float),('np', float) ] )
    else:
        ll = np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),('r', float),('lat', float),('lon', float),('x', float),('y', float),('z', float)] )
        

    ll = ll.view(np.recarray)  
    
    # replace all unreasonable large and small values with nan
    thresh = 1e6
    br1[br1 > thresh] = np.nan
    br1[br1 < -thresh] = np.nan
    bt1[bt1 > thresh] = np.nan
    bt1[bt1 < -thresh] = np.nan
    bn1[bn1 > thresh] = np.nan
    bn1[bn1 < -thresh] = np.nan

    
    ll.time = time_int
    ll.bx = np.interp(time_int_mat, time1_mat[~np.isnan(br1)], br1[~np.isnan(br1)])
    ll.by = np.interp(time_int_mat, time1_mat[~np.isnan(bt1)], bt1[~np.isnan(bt1)])
    ll.bz = np.interp(time_int_mat, time1_mat[~np.isnan(bn1)], bn1[~np.isnan(bn1)])
    ll.bt = np.sqrt(ll.bx**2 + ll.by**2 + ll.bz**2)
    
    if os.path.exists(swapath):
        ll.np = np.interp(time_int_mat, timep_mat, den)
        ll.tp = np.interp(time_int_mat, timep_mat, temp) 
        ll.vx = np.interp(time_int_mat, timep_mat, vr)
        ll.vy = np.interp(time_int_mat, timep_mat, vt)
        ll.vz = np.interp(time_int_mat, timep_mat, vn)
        ll.vt = np.sqrt(ll.vx**2 + ll.vy**2 + ll.vz**2)

    if sc == 'solo':
        #Solar Orbiter position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        #time=heeq.obstime.to_datetime()
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)
        
        #Solar Orbiter position with astrospice
        #kernels = astrospice.registry.get_kernels(fullname, 'predict')
        #solo_kernel = kernels[0]

        #solo_coords = astrospice.generate_coords(fullname, time_int)
        
        #solo_coords_heeq = solo_coords.transform_to(sunpy.coordinates.HeliographicStonyhurst())

        #ll.lon = solo_coords_heeq.lon.value
        #ll.lat = solo_coords_heeq.lat.value
        #ll.r = solo_coords_heeq.radius.to(u.au).value    

        #ll = sphere2cart(ll)
        #print(ll.x, ll.y, ll.z)
        
    #try:
    #    #spacecraft position with astrospice
    #    kernels = astrospice.registry.get_kernels(fullname, 'predict')
    #    solo_kernel = kernels[0]
    #
    #    solo_coords = astrospice.generate_coords(fullname, time_int)
    #    
    #    solo_coords_heeq = solo_coords.transform_to(sunpy.coordinates.HeliographicStonyhurst())
    #
    #    ll.lon = solo_coords_heeq.lon.value
    #    ll.lat = solo_coords_heeq.lat.value
    #    ll.r = solo_coords_heeq.radius.to(u.au).value    
    #
    #    ll = sphere2cart(ll)
    #    print(ll.x, ll.y, ll.z)
    
    elif sc == 'psp':    
        # PSP position with sunpy
        coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)
            
    #except:  
    #    coord = get_horizons_coord(sc, time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
    #    heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
    #    
    #    ll.r = heeq.radius.value
    #    ll.lon = heeq.lon.value
    #    ll.lat = heeq.lat.value
    #    
    #    ll = sphere2cart(ll)
    #    
    #    print(ll.x, ll.y, ll.z)

    elif sc == 'wind':
        # WIND position with sunpy
        
        # use 
        coord = get_horizons_coord('EM-L1', time={'start': time_int[0], 'stop': time_int[-1], 'step': '1m'})  
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
        ll.r = heeq.radius.value
        ll.lon = heeq.lon.value
        ll.lat = heeq.lat.value
        
        
        ll = sphere2cart(ll)
        
        print(ll.x, ll.y, ll.z)
     
    else:
        print('no spacecraft position')
        
        
    return ll
    
    
def sphere2cart(ll):
        
    ll.x = ll.r * np.cos(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.y = ll.r * np.sin(np.deg2rad(ll.lon)) * np.cos(np.deg2rad(ll.lat))
    ll.z = ll.r * np.sin(np.deg2rad(ll.lat))
    
    return ll