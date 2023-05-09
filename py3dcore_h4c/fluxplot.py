import os

import numpy as np
import pickle as p
import pandas as pds
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')


from matplotlib.widgets import Slider, Button

from matplotlib.colors import LightSource

import datetime as datetime
from datetime import timedelta
import py3dcore_h4c
from py3dcore_h4c.fitter.base import custom_observer, BaseFitter, get_ensemble_mean


from sunpy.coordinates import frames, get_horizons_coord
import heliosat
    
from scipy.optimize import least_squares

from py3dcore_h4c.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

from .rotqs import generate_quaternions

import matplotlib.pyplot as plt

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import matplotlib.dates as mdates

from itertools import product

import logging

logger = logging.getLogger(__name__)

def get_overwrite(out):
    
    """ creates iparams from parameter statistic"""
    
    overwrite = {
        "cme_longitude": {
                "maximum": out['lon'].mean()+out['lon'].std(),
                "minimum": out['lon'].mean()-out['lon'].std()
            },
        "cme_latitude": {
                "maximum": out['lat'].mean()+out['lat'].std(),
                "minimum": out['lat'].mean()-out['lat'].std()
            },
        "cme_inclination" :{
                "maximum": out['inc'].mean()+out['inc'].std(),
                "minimum": out['inc'].mean()-out['inc'].std()
            },
        "cme_diameter_1au" :{
                "maximum": out['D1AU'].mean()+out['D1AU'].std(),
                "minimum": out['D1AU'].mean()-out['D1AU'].std()
            },
        "cme_aspect_ratio": {
                "maximum": out['delta'].mean()+out['delta'].std(),
                "minimum": out['delta'].mean()-out['delta'].std()
            },
        "cme_launch_radius": {
                "maximum": out['launch radius'].mean()+out['launch radius'].std(),
                "minimum": out['launch radius'].mean()-out['launch radius'].std()
            },
        "cme_launch_velocity": {
                "maximum": out['launch speed'].mean()+out['launch speed'].std(),
                "minimum": out['launch speed'].mean()-out['launch speed'].std()
            },
        "t_factor": {
                "maximum": out['t factor'].mean()+out['t factor'].std(),
                "minimum": out['t factor'].mean()-out['t factor'].std()
            },
        "magnetic_field_strength_1au": {
                "maximum": out['B1AU'].mean()+out['B1AU'].std(),
                "minimum": out['B1AU'].mean()-out['B1AU'].std()
            },
        "background_drag": {
                "maximum": out['gamma'].mean()+out['gamma'].std(),
                "minimum": out['gamma'].mean()-out['gamma'].std()
            },
        "background_velocity": {
                "maximum": out['vsw'].mean()+out['vsw'].std(),
                "minimum": out['vsw'].mean()-out['vsw'].std()
            }
    }
    
    return overwrite


def get_params(filepath, give_mineps = False):
    
    """ Gets params from file. """
    
    # read from pickle file
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    model_objt = data["model_obj"]
    maxiter = model_objt.ensemble_size-1

    # get index ip for run with minimum eps    
    epses_t = data["epses"]
    ip = np.argmin(epses_t[0:maxiter])    
    
    # get parameters (stored in iparams_arr) for the run with minimum eps
    
    iparams_arrt = model_objt.iparams_arr
    
    meanparams = np.mean(model_objt.iparams_arr, axis=0)
    
    resparams = iparams_arrt[ip]
    
    names = ['lon: ', 'lat: ', 'inc: ', 'diameter 1 AU: ', 'aspect ratio: ', 'launch radius: ', 'launch speed: ', 't factor: ', 'expansion rate: ', 'magnetic field decay rate: ', 'magnetic field 1 AU: ', 'drag coefficient: ', 'sw background speed: ']
    if give_mineps == True:
        logger.info("Retrieved the following parameters for the run with minimum epsilon:")
    
        for count, name in enumerate(names):
            logger.info(" --{} {:.2f}".format(name, resparams[count+1]))

    return resparams, iparams_arrt, ip, meanparams

def get_ensemble_stats(filepath):
    
    ftobj = BaseFitter(filepath) # load Fitter from path
    model_obj = ftobj.model_obj
    
    df = pds.DataFrame(model_obj.iparams_arr)
    cols = df.columns.values.tolist()

    # drop first column, and others in which you are not interested
    df.drop(df.columns[[0, 9, 10]], axis=1, inplace=True)

    # rename columns
    df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'B1AU', 'gamma', 'vsw']
    
    df.describe()
    
    return df
    

def scatterparams(path):
    
    ''' returns scatterplots from a results file'''
    
    res, iparams_arrt, ind, meanparams = get_params(path)
    
    df = pds.DataFrame(iparams_arrt)
    cols = df.columns.values.tolist()

    # drop first column, and others in which you are not interested
    df.drop(df.columns[[0, 9, 10]], axis=1, inplace=True)

    # rename columns
    df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'B1AU', 'gamma', 'vsw']

    g = sns.pairplot(df, 
                     corner=True,
                     plot_kws=dict(marker="+", linewidth=1)
                    )
    g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2") #  levels are 2-sigma and 1-sigma contours
    g.savefig(path[:-7] + 'scatter_plot_matrix.pdf')
    plt.show()
    

def equal_t_creator(start,n,delta):
    
    """ Creates a list of n datetime entries separated by delta hours starting at start. """
    
    t = [start + i * datetime.timedelta(hours=delta) for i in range(n)]
    
    return t

def loadpickle(path = None, number = -1):

    """ Loads the filepath of a pickle file. """

    # Get the list of all files in path
    dir_list = sorted(os.listdir(path))

    resfile = []
    respath = []
    # we only want the pickle-files
    for file in dir_list:
        if file.endswith(".pickle"):
            resfile.append(file) 
            respath.append(os.path.join(path,file))
            
    filepath = path + resfile[number]

    return filepath

def returnmodel(filepath):
    
    ''' returns a model using the statistics from a previous result'''
    
    t_launch = BaseFitter(filepath).dt_0
    
    out = get_ensemble_stats(filepath)
    overwrite = get_overwrite(out)
    model_obj = py3dcore_h4c.ToroidalModel(t_launch, 1, iparams=overwrite)
    print(model_obj.iparams_arr)
    model_obj.generator()
    
    return model_obj

def plot_traj(ax, sat, t_snap, frame="HEEQ", traj_pos=True, traj_minor=None, custom_data = False, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", 1)
    color = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)
    kwargs["s"] = kwargs.pop("s", 25)
    traj_major = kwargs.pop("traj_major", 80)
    
    if custom_data == False:
        if sat == "Solar Orbiter":
            observer = "SOLO"
        inst = getattr(heliosat, observer)() # get observer obj
        logger.info("Using HelioSat to retrieve observer data")
        
    elif custom_data=='sunpy':
        
        _s = kwargs.pop("s")
        
        if "ls" in kwargs:
            kwargs.pop("ls")
            
        _ls = "--"
        _lw = kwargs.pop("lw") / 2

        if traj_major and traj_major > 0:
            start = t_snap - datetime.timedelta(hours=traj_major)
            end = t_snap + datetime.timedelta(hours=traj_major+15)
            t, pos, traj = getpos(sat, t_snap.strftime('%Y-%m-%d-%H'), start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            ax.plot(traj[0]*np.cos(np.radians(traj[1])),traj[0]*np.sin(np.radians(traj[1])),0, color='k', alpha=0.9, ls=_ls, lw=_lw)

        if traj_minor and traj_minor > 0:
            start = t_snap - datetime.timedelta(hours=traj_minor)
            end = t_snap + datetime.timedelta(hours=traj_minor)
            t, pos, traj = getpos(sat, t_snap.strftime('%Y-%m-%d-%H'), start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            ax.plot(traj[0]*np.cos(np.radians(traj[1])),traj[0]*np.sin(np.radians(traj[1])),0, color='k', alpha=0.9, ls=_ls, lw=_lw)

        
        if traj_pos:
            start = t_snap - datetime.timedelta(hours=30)
            end = t_snap + datetime.timedelta(hours=30)
            _, pos, _ = getpos(sat, t_snap.strftime('%Y-%m-%d-%H'), start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            plot_satellite(ax,pos, label = sat,s=_s, lw=_lw, color = color)
        
        return
        
    else:
        inst = custom_observer(custom_data)
    
    
    _s = kwargs.pop("s")

    if traj_pos:
        pos = inst.trajectory(t_snap, frame)

        ax.scatter(*pos.T, s=_s, **kwargs)
        
    if traj_major and traj_major > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_major, traj_major)], frame)
        #ax.plot(*traj.T, **kwargs)
        
    if traj_minor and traj_minor > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_minor, traj_minor)], frame)
        
    if "ls" in kwargs:
        kwargs.pop("ls")

    _ls = "--"
    _lw = kwargs.pop("lw") / 2

    ax.plot(*traj.T, ls=_ls, lw=_lw, **kwargs)

def getpos(sc, date, start, end):
    
    """returns the positions for a spacecraft using bett""" 
    
    coord = get_horizons_coord(sc, time={'start': start, 'stop': end, 'step': '1m'})  
    heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
    hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE

    time = heeq.obstime.to_datetime()
    r = heeq.radius.value
    lon = np.deg2rad(heeq.lon.value)
    lat = np.deg2rad(heeq.lat.value)
    
    # get position of spacecraft for specific date

    t = []

    for i in range(len(time)):
        tt = time[i].strftime('%Y-%m-%d-%H')
        t.append(tt)

    ind = t.index(date)    
    logger.info("Indices of date: %i", ind)
    
    logger.info("%s - r: %f, lon: %f, lat: %f, ", sc, r[ind], heeq.lon.value[ind], heeq.lat.value[ind])
    
    pos = np.asarray([r[ind], heeq.lon.value[ind], heeq.lat.value[ind]])
    
    traj = np.asarray([r, heeq.lon.value, heeq.lat.value])
    
    return t, pos, traj

def plot_shift(axis,extent,cx,cy,cz):
    #shift center of plot
    axis.set_xbound(cx-extent, cx+extent)
    axis.set_ybound(cy-extent, cy+extent)
    axis.set_zbound(cz-extent * 0.75, cz + extent * 0.75)
    
#define sun here so it does not need to be recalculated every time
scale=695510/149597870.700 #Rs in km, AU in km
# sphere with radius Rs in AU
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
x = np.cos(u)*np.sin(v)*scale
y = np.sin(u)*np.sin(v)*scale
z = np.cos(v)*scale

def full3d_multiview(t_launch, filepath):
    
    """
    Plots 3d from multiple views.
    """
    
    TP_A =  t_launch + datetime.timedelta(hours=2)
    TP_B =  t_launch + datetime.timedelta(hours=46)

    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15, 11),dpi=100)

    #define subplot grid
    ax1 = plt.subplot2grid((2, 3), (0, 0),rowspan=2,colspan=2,projection='3d')  
    ax2 = plt.subplot2grid((2, 3), (0, 2),projection='3d')  
    ax3 = plt.subplot2grid((2, 3), (1, 2),projection='3d')  
    
    model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean') #returnmodel(filepath)
    
    ######### tilted view
    plot_configure(ax1, view_azim=150, view_elev=25, view_radius=.2, light_source=True) #view_radius=.08

    plot_3dcore(ax1, model_obj, TP_A, color=C_A,light_source = True)
    plot_3dcore_field(ax1, model_obj, color=C_A, step_size=0.0005, lw=1.0, ls="-")
    plot_traj(ax1, "Parker Solar Probe", t_snap = TP_A, frame="HEEQ", custom_data = 'sunpy', color=C_A)
    
    plot_3dcore(ax1, model_obj, TP_B, color=C_B, light_source = True)
    plot_3dcore_field(ax1, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    plot_traj(ax1, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)
    
    #dotted trajectory
    #plot_traj(ax1, "PSP", TP_B, frame="ECLIPJ2000", color="k", traj_pos=False, traj_major=None, traj_minor=144,lw=1.5)

    #shift center
    plot_shift(ax1,0.31,-0.25,0.0,-0.2)
    
    
    ########### top view panel
    plot_configure(ax2, view_azim=165-90, view_elev=90, view_radius=.08,light_source=True)
    
    plot_3dcore(ax2, model_obj, TP_A, color=C_A,light_source = True)
    plot_3dcore_field(ax2, model_obj, color=C_A, step_size=0.0005, lw=1.0, ls="-")
    plot_traj(ax2, "Parker Solar Probe", t_snap = TP_A, frame="HEEQ", custom_data = 'sunpy', color=C_A)
    
    plot_3dcore(ax2, model_obj, TP_B, color=C_B, light_source = True)
    plot_3dcore_field(ax2, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    plot_traj(ax2, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)
    plot_shift(ax2,0.26,-0.41,0.08,0.0) 
    
    
    ############## edge on view panel
    plot_configure(ax3, view_azim=65, view_elev=-5, view_radius=.01,light_source=True)
    plot_traj(ax3, "Parker Solar Probe", t_snap=TP_A, frame="HEEQ", custom_data = 'sunpy', color=C_A)

    plot_3dcore(ax3, model_obj, TP_B, color=C_B, light_source = True)
    plot_3dcore_field(ax3, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    plot_traj(ax3, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)

    plot_shift(ax3,0.26,-0.41,0.08,0.0)


    plt.savefig(filepath[:-7] + 'multiview.pdf',bbox_inches='tight')
    
    
def full3d_multiview_movie(t_launch, t, filepath, frametime):
    
    """
    Plots 3d movie from multiple views.
    """
    
    fig = plt.figure(52,figsize=(19.2, 10.8),dpi=100)
    
    #define subplot grid
    ax1 = plt.subplot2grid((3, 3), (0, 0),rowspan=3,colspan=2,projection='3d')  
    ax2 = plt.subplot2grid((3, 3), (0, 2),projection='3d')  
    ax3 = plt.subplot2grid((3, 3), (1, 2),projection='3d')  
    
    #manually set axes positions
        
    ax1.set_position([0,0,0.6,1], which='both')
    ax2.set_position([0.65,0.35,0.35,0.65], which='both')
    ax3.set_position([0.6,0,0.4,0.4], which='both')
    

    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    sns.set_style('whitegrid')
    sns.set_style("ticks",{'grid.linestyle': '--'})
    
    model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean') #returnmodel(filepath)
    
    ######### tilted view
    plot_configure(ax1, view_azim=150, view_elev=25, view_radius=.2,light_source=True) #view_radius=.08

    plot_3dcore(ax1, model_obj, t, color=C_A,light_source = True,lw = 3)
    plot_traj(ax1, "Parker Solar Probe", t_snap = t, frame="HEEQ", custom_data = 'sunpy', color='k')
    plot_traj(ax1, "Solar Orbiter", t_snap = t, frame="HEEQ", custom_data = 'sunpy', color='k')

    #shift center
    plot_shift(ax1,0.31,-0.25,0.0,-0.2)
    
    
    ########### top view panel
    plot_configure(ax2, view_azim=165-90, view_elev=90, view_radius=.08,light_source=True)
    
    plot_3dcore(ax2, model_obj, t, color=C_A,light_source = True,lw = 3)
    plot_traj(ax2, "Parker Solar Probe", t_snap = t, frame="HEEQ", custom_data = 'sunpy', color='k')
    plot_traj(ax2, "Solar Orbiter", t_snap = t, frame="HEEQ", custom_data = 'sunpy', color='k')
    plot_shift(ax2,0.26,-0.41,0.08,0.0) 
    
    
    ############## edge on view panel
    plot_configure(ax3, view_azim=65, view_elev=-5, view_radius=.01,light_source=True,lw = 3)
    plot_traj(ax3, "Parker Solar Probe", t_snap=t, frame="HEEQ", custom_data = 'sunpy', color='k')
    plot_3dcore(ax3, model_obj, t, color=C_A, light_source = True)
    plot_traj(ax3, "Solar Orbiter", t_snap = t, frame="HEEQ", custom_data = 'sunpy', color='k')

    plot_shift(ax3,0.26,-0.41,0.08,0.0)
    
    plt.annotate('$t_{launch}$ +',[0.45,0.15],ha='center',xycoords='figure fraction',fontsize=20)
    plt.annotate(str(frametime),[0.5,0.15],ha='center',xycoords='figure fraction',fontsize=20)
    plt.annotate('hours',[0.54,0.15],ha='center',xycoords='figure fraction',fontsize=20)
    
    return fig
    
    
def full3d(spacecraftlist=['solo', 'psp'], planetlist =['Earth'], t=None, traj = 50, filepath=None, custom_data=False, save_fig = True, legend = True, title=True, view_azim=0, view_elev=45, view_radius=0.2, **kwargs):
    
    """
    Plots 3d.
    """
    
    #colors for 3dplots

    c0 = 'mediumseagreen'
    c1 = "xkcd:red"
    c2 = "xkcd:blue"
    
    #Color settings    
    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    earth_color='blue'
    solo_color='orange'
    venus_color='mediumseagreen'
    mercury_color='grey'
    psp_color='black'
    sta_color='red'
    bepi_color='coral' 
    
    sns.set_context("talk")     

    sns.set_style("ticks",{'grid.linestyle': '--'})
    fsize=15

    fig = plt.figure(figsize=(15,12),dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_configure(ax, view_azim=view_azim, view_elev=view_elev, view_radius=view_radius)
    
    model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean') #returnmodel(filepath)
    
    plot_3dcore(ax, model_obj, t, color=c2)
    plot_3dcore_field(ax, model_obj, color=c2, step_size=0.005, lw=1.1, ls="-")
    

    
    if 'solo' in spacecraftlist:

        plot_traj(ax, sat = 'Solar Orbiter', t_snap = t, frame="HEEQ", traj_pos=True, traj_major = traj, traj_minor=None, custom_data = 'sunpy', color=solo_color,**kwargs)
        
        
    if 'psp' in spacecraftlist:
        plot_traj(ax, sat = 'Parker Solar Probe', t_snap = t, frame="HEEQ", traj_pos=True, traj_major = traj, traj_minor=None, custom_data = 'sunpy', color=psp_color,**kwargs)
        
    
    if 'Earth' in planetlist:
        earthpos = np.asarray([1,0, 0])
        plot_planet(ax,earthpos,color=earth_color,alpha=0.9, label = 'Earth')
        plot_circle(ax,earthpos[0])
        
    if 'Venus' in planetlist:
        t_ven, pos_ven, traj_ven  = getpos('Venus Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_planet(ax,pos_ven,color=venus_color,alpha=0.9, label = 'Venus')
        plot_circle(ax,pos_ven[0])
        
    if 'Mercury' in planetlist:
        t_mer, pos_mer, traj_mer  = getpos('Mercury Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_planet(ax,pos_mer,color=mercury_color,alpha=0.9, label = 'Mercury')
        plot_circle(ax,pos_mer[0])
        
        
    
    if legend == True:
        ax.legend(loc='upper right')
    if title == True:
        plt.title('3DCORE fitting result - ' + t.strftime('%Y-%m-%d-%H'))
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'full3d.pdf', dpi=300)  
    
    return fig

        

def fullinsitu(observer, t_fit=None, start = None, end=None, filepath=None, custom_data=False, save_fig = True, best = True, ensemble = True, mean = False, legend=True, fixed=None, max_index=128, title=True, fit_points=True):
    
    """
    Plots the synthetic insitu data plus the measured insitu data and ensemble fit.

    Arguments:
        observer          name of the observer
        t_fit             datetime points used for fitting
        start             starting point of the plot
        end               ending point of the plot
        path              where to find the fitting results
        number            which result to use
        custom_data       path to custom data, otherwise heliosat is used
        save_fig          whether to save the created figure
        legend            whether to plot legend 
        max_index         how much to keep of the generated ensemble
        title             whether to plot title 
        fit_points        whether to indicate fitting points in plot

    Returns:
        None
    """
    
    if start == None:
        start = t_fit[0]

    if end == None:
        end = t_fit[-1]
    
    
    if custom_data == False:
        observer_obj = getattr(heliosat, observer)() # get observer obj
        logger.info("Using HelioSat to retrieve observer data")
        dt, b = observer_obj.get([start, end], "mag", reference_frame="HEEQ", as_endpoints=True)
        t = []
        t = [datetime.datetime.fromtimestamp(dt[i]) for i in range(len(dt))] # .strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        observer_obj = custom_observer(custom_data)
        t, b = observer_obj.get([start, end], "mag", reference_frame="HEEQ", as_endpoints=True)

    
    pos = observer_obj.trajectory(t, reference_frame="HEEQ")
    
    if best == True:
        model_obj = returnfixedmodel(filepath)
        
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
        print(t, pos, outa)
        
    if fixed is not None:
        model_obj = returnfixedmodel(filepath, fixed)
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
        
    
    if mean == True:
        model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean')
        
        model_obj.generator()
        means = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        means[means==0] = np.nan
    #print(outa)        
    # get ensemble_data
    if ensemble == True:
        ed = py3dcore_h4c.generate_ensemble(filepath, t, reference_frame="HEEQ", reference_frame_to="HEEQ", max_index=128, custom_data=custom_data)
    
    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points
    
    if (observer == 'solo') or (observer == 'SOLO'):
        obs_title = 'Solar Orbiter'

        
    if (observer == 'PSP') or (observer == 'psp'):
        obs_title = 'Parker Solar Probe'

    if (observer == 'Wind') or (observer == 'WIND') or (observer == 'wind'):
        obs_title = 'WIND'  
        
    plt.figure(figsize=(20, 10))
    if title == True:
        plt.title("3DCORE fitting result - "+obs_title)

        
    plt.plot(t, np.sqrt(np.sum(b**2, axis=1)), "k", alpha=0.5, lw=3, label ='Btotal')
    plt.plot(t, b[:, 0], "r", alpha=1, lw=lw_insitu, label ='Br')
    plt.plot(t, b[:, 1], "g", alpha=1, lw=lw_insitu, label ='Bt')
    plt.plot(t, b[:, 2], "b", alpha=1, lw=lw_insitu, label ='Bn')
    if ensemble == True:
        plt.fill_between(t, ed[0][3][0], ed[0][3][1], alpha=0.25, color="k")
        plt.fill_between(t, ed[0][2][0][:, 0], ed[0][2][1][:, 0], alpha=0.25, color="r")
        plt.fill_between(t, ed[0][2][0][:, 1], ed[0][2][1][:, 1], alpha=0.25, color="g")
        plt.fill_between(t, ed[0][2][0][:, 2], ed[0][2][1][:, 2], alpha=0.25, color="b")
        
    if (best == True) or (fixed is not None):
        if best == True:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with min(eps)')
        else:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with fixed iparams')
        plt.plot(t, outa[:, 0], "r", alpha=0.5, linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 1], "g", alpha=0.5, linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 2], "b", alpha=0.5, linestyle='dashed', lw=lw_best)
        
    if mean == True:
        plt.plot(t, np.sqrt(np.sum(means**2, axis=1)), "k", alpha=0.5, linestyle='dashdot', lw=lw_mean)#, label ='run with mean iparams')
        plt.plot(t, means[:, 0], "r", alpha=0.75, linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 1], "g", alpha=0.75, linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 2], "b", alpha=0.75, linestyle='dashdot', lw=lw_mean)
        
        
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    # plt.xlabel("Time")
    plt.xticks(rotation=25, ha='right')
    if legend == True:
        plt.legend(loc='lower right',ncol=2)
    if fit_points == True:    
        for _ in t_fit:
            plt.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'fullinsitu.pdf', dpi=300)    
    plt.show()
    

def insituprofiles(observer, date=None, start=None, end=None, filepath=None, save_fig=True, best=True, mean=False, legend=True, fixed=None):
    
    """
    Plots the synthetic in situ profiles, when there are no spacecraft data.

    Arguments:
        observer          name of the observer
        start             starting point of the plot
        end               ending point of the plot
        filepath          where to find the fitting results
        save_fig          whether to save the created figure
        best              whether to plot run with min(eps)
        mean              whether to plot run with mean parameter values
        legend            whether to plot legend 

    Returns:
        None
    """
    
    if start == None:
        logger.info("Please specify start time of plot")

    if end == None:
        logger.info("Please specify end time of plot")
        
    t = np.asarray([start + datetime.timedelta(hours=i) for i in range(168)])
    end = start + datetime.timedelta(hours=168)
    ttt,pos_temp,traj = getpos(observer, date, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    pos1 = [[traj[0][i],traj[1][i],traj[2][i]] for i in range(len(traj[0]))]
    # print(pos1)
    
    r_temp = [traj[0][i] for i in range(len(traj[0]))]
    lon_temp = [traj[1][i] for i in range(len(traj[0]))]
    lat_temp = [traj[2][i] for i in range(len(traj[0]))]
    
    pos_x = r_temp * np.cos(np.deg2rad(lon_temp)) * np.cos(np.deg2rad(lat_temp))
    pos_y = r_temp * np.sin(np.deg2rad(lon_temp)) * np.cos(np.deg2rad(lat_temp))
    pos_z = r_temp * np.sin(np.deg2rad(lat_temp))
    
    pos = [[pos_x[i],pos_y[i],pos_z[i]] for i in range(len(pos_x))]
    # print(pos)

    if best == True:
        model_obj = returnfixedmodel(filepath)
        
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
        
    if fixed is not None:
        model_obj = returnfixedmodel(filepath, fixed)
        
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
    
    if mean == True:
        model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean')
        means = np.squeeze(np.array(model_obj.simulator(t, pos))[0])     
        means[means==0] = np.nan
        if np.isnan(means).all() == True:
            logger.info("WARNING: Apparently not a hit --- > skipping plot")
            return
    
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    
    if observer == 'solo':
        obs_title = 'Solar Orbiter'
    
    if observer == 'psp':
        obs_title = 'Parker Solar Probe'

    plt.figure(figsize=(20, 10))
    plt.title("3DCORE synthetic profiles - "+obs_title)
        
    if (best == True) or (fixed is not None):
        if best == True:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with min(eps)')
        else:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with fixed iparams')
        plt.plot(t, outa[:, 0], "r", alpha=0.5,linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 1], "g", alpha=0.5,linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 2], "b", alpha=0.5,linestyle='dashed', lw=lw_best)
        
    if mean == True:
        plt.plot(t, np.sqrt(np.sum(means**2, axis=1)), "k", alpha=0.5, linestyle='dashdot', lw=lw_mean)#, label ='run with mean iparams')
        plt.plot(t, means[:, 0], "r", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 1], "g", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 2], "b", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        
        
        
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    # plt.xlabel("Time")
    plt.xticks(rotation=25, ha='right')
    if legend == True:
        plt.legend(loc='lower right')
    if save_fig == True:
        plt.savefig('%s_insituprofiles.pdf' %filepath, dpi=300)    
    plt.show()
    
    
def update_model(model, t_i=None, lon=None, lat=None, inc=None, dia=None, delta=None, r0=None, v0=None, T=None, n_a=None, n_b=None, b=None, bg_d=None, bg_v=None):
    
    """updates a specific parameter in a model"""
    
    if t_i is not None:
        model.iparams_arr[0][0] = t_i
        
    if lon is not None:
        model.iparams_arr[0][1] = lon
    
    if lat is not None:
        model.iparams_arr[0][2] = lat
        
    if inc is not None:
        model.iparams_arr[0][3] = inc
        
    if dia is not None:
        model.iparams_arr[0][4] = dia
        
    if delta is not None:
        model.iparams_arr[0][5] = delta
        
    if r0 is not None:
        model.iparams_arr[0][6] = r0
        
    if v0 is not None:
        model.iparams_arr[0][7] = v0
        
    if T is not None:
        model.iparams_arr[0][8] = T
        
    if n_a is not None:
        model.iparams_arr[0][9] = n_a
        
    if n_b is not None:
        model.iparams_arr[0][10] = n_b
        
    if b is not None:
        model.iparams_arr[0][11] = b
        
    if bg_d is not None:
        model.iparams_arr[0][12] = bg_d
        
    if bg_v is not None:
        model.iparams_arr[0][13] = bg_v
        
    model.sparams_arr = np.empty((model.ensemble_size, model.sparams), dtype=model.dtype)
    model.qs_sx = np.empty((model.ensemble_size, 4), dtype=model.dtype)
    model.qs_xs = np.empty((model.ensemble_size, 4), dtype=model.dtype)
    
    model.iparams_meta = np.empty((len(model.iparams), 7), dtype=model.dtype)
    
    #iparams_meta is updated
    generate_quaternions(model.iparams_arr, model.qs_sx, model.qs_xs)
        
    return model

    
def returnfixedmodel(filepath, fixed_iparams_arr=None):
    
    '''returns a fixed model not generating random iparams'''
    
    ftobj = BaseFitter(filepath) # load Fitter from path
    model_obj = ftobj.model_obj
    
    model_obj.ensemble_size = 1
    
    if fixed_iparams_arr == 'mean':
        logger.info("Plotting run with mean parameters.")
        res, allres, ind, meanparams = get_params(filepath)
        model_obj.iparams_arr = np.expand_dims(meanparams, axis=0)
    
    else:
        try: 
            fixed_iparams_arr_t
            logger.info("Plotting run with fixed parameters.")
            model_obj.iparams_arr = np.expand_dims(fixed_iparams_arr, axis=0)
    
        except:
            logger.info("No iparams_arr given, using parameters for run with minimum eps.")
            res, allres, ind, meanparams = get_params(filepath)
            model_obj.iparams_arr = np.expand_dims(res, axis=0)
            
    
    model_obj.sparams_arr = np.empty((model_obj.ensemble_size, model_obj.sparams), dtype=model_obj.dtype)
    model_obj.qs_sx = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    model_obj.qs_xs = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    
    #model_obj.iparams_kernel = None
    #model_obj.iparams_kernel_decomp = None
    #model_obj.iparams_weight = None
    model_obj.iparams_meta = np.empty((len(model_obj.iparams), 7), dtype=model_obj.dtype)
    
    #iparams_meta is updated
    generate_quaternions(model_obj.iparams_arr, model_obj.qs_sx, model_obj.qs_xs)
    print(model_obj.iparams_arr)
    return model_obj
    
    
    
    
    
    
    
def plot_configure(ax, light_source = False, **kwargs):
    view_azim = kwargs.pop("view_azim", -25)
    view_elev = kwargs.pop("view_elev", 25)
    view_radius = kwargs.pop("view_radius", .5)
    
    ax.view_init(azim=view_azim, elev=view_elev)

    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    ax.set_zlim([-view_radius, view_radius])
    
    if light_source == True:
        #draw sun        
        ls = LightSource(azdeg=320, altdeg=40)  
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow',lightsource=ls, linewidth=0, antialiased=False,zorder=5)
    
    ax.set_axis_off()

def plot_3dcore(ax, obj, t_snap, light_source=False, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", .05)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)

    if light_source == False:
        ax.scatter(0, 0, 0, color="y", s=50) # 5 solar radii
        
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', linewidth=0, antialiased=False)

    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(0,)#visualize_wireframe(index=0)
    ax.plot_wireframe(*wf_model.T, **kwargs)

    
def plot_3dcore_field(ax, obj, step_size=0.005, q0=[0.8, .1, np.pi/2],**kwargs):

    #initial point is q0
    q0i =np.array(q0, dtype=np.float32)
    fl = visualize_fieldline(obj, q0, index=0, steps=8000, step_size=0.005)
    #fl = model_obj.visualize_fieldline_dpsi(q0i, dpsi=2*np.pi-0.01, step_size=step_size)
    ax.plot(*fl.T, **kwargs)
    
def plot_circle(ax,dist,**kwargs):        

    thetac = np.linspace(0, 2 * np.pi, 100)
    xc=dist*np.sin(thetac)
    yc=dist*np.cos(thetac)
    zc=0
    ax.plot(xc,yc,zc,ls='--',color='black',lw=0.3,**kwargs)
      
def plot_satellite(ax,satpos1,**kwargs):

    xc=satpos1[0]*np.cos(np.radians(satpos1[1]))
    yc=satpos1[0]*np.sin(np.radians(satpos1[1]))
    zc=0
    #print(xc,yc,zc)
    ax.scatter3D(xc,yc,zc,marker ='s',**kwargs)
        
def plot_planet(ax,satpos1,**kwargs):

    xc=satpos1[0]*np.cos(np.radians(satpos1[1]))
    yc=satpos1[0]*np.sin(np.radians(satpos1[1]))
    zc=0
    #print(xc,yc,zc)
    ax.scatter3D(xc,yc,zc,s=10,**kwargs)
    
    
def visualize_wireframe(obj, index=0, r=1.0, d=10):
        """Generate model wireframe.

        Parameters
        ----------
        index : int, optional
            Model run index, by default 0.
        r : float, optional
            Surface radius (r=1 equals the boundary of the flux rope), by default 1.0.

        Returns
        -------
        np.ndarray
            Wireframe array (to be used with plot_wireframe).
        """
        r = np.array([np.abs(r)], dtype=obj.dtype)

        c = 360 // d + 1
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=obj.dtype).reshape(c ** 2, 3)

        for i in range(0, len(arr)):
            thin_torus_qs(arr[i], arr[i], obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index])

        return arr.reshape((c, c, 3))
    

def visualize_fieldline(obj, q0, index=0, steps=1000, step_size=0.01):
    
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        index : int, optional
            Model run index, by default 0.
        steps : int, optional
            Number of integration steps, by default 1000.
        step_size : float, optional
            Integration step size, by default 0.01.

        Returns
        -------
        np.ndarray
            Integrated magnetic field lines in (s) coordinates.
        """

        _tva = np.empty((3,), dtype=obj.dtype)
        _tvb = np.empty((3,), dtype=obj.dtype)

        thin_torus_qs(q0, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tva)

        fl = [np.array(_tva, dtype=obj.dtype)]
        def iterate(s):
            thin_torus_sq(s, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_sx[index],_tva)
            thin_torus_gh(_tva, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tvb)
            return _tvb / np.linalg.norm(_tvb)

        while len(fl) < steps:
            # use implicit method and least squares for calculating the next step
            try:
                sol = getattr(least_squares(
                    lambda x: x - fl[-1] - step_size *
                    iterate((x.astype(obj.dtype) + fl[-1]) / 2),
                    fl[-1]), "x")

                fl.append(np.array(sol.astype(obj.dtype)))
            except Exception as e:
                break

        fl = np.array(fl, dtype=obj.dtype)

        return fl