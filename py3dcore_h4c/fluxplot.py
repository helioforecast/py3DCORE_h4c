import os

import numpy as np
import pickle as p
import pandas as pds
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

import datetime as datetime
from datetime import timedelta
import py3dcore_h4c
from py3dcore_h4c.fitter.base import custom_observer, BaseFitter, get_ensemble_mean

from sunpy.coordinates import frames, get_horizons_coord

    
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
    g.savefig(path+'scatter_plot_matrix.png')
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
    
    t_launch = BaseFitter(filepath).dt_0
    
    out = get_ensemble_stats(filepath)
    overwrite = get_overwrite(out)
    
    model_obj = py3dcore_h4c.ToroidalModel(t_launch, 1, iparams=overwrite)
    
    model_obj.generator()
    
    return model_obj

def getpos(sc, date, start, end):
    
    
    coord = get_horizons_coord(sc, time={'start': start, 'stop': end, 'step': '60m'})  
    heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
    hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE

    time=heeq.obstime.to_datetime()
    r=heeq.radius.value
    lon=np.deg2rad(heeq.lon.value)
    lat=np.deg2rad(heeq.lat.value)
    
    # get position of Solar Orbiter for specific date

    t = []

    for i in range(len(time)):
        tt = time[i].strftime('%Y-%m-%d-%H')
        t.append(tt)

    ind = t.index(date)    
    logger.info("Indices of date: %i", ind)
    
    logger.info("%s - r: %f, lon: %f, lat: %f, ", sc, r[ind], np.rad2deg(lon[ind]),np.rad2deg(lat[ind]))
    
    pos= np.asarray([r[ind],np.rad2deg(lon[ind]), np.rad2deg(lat[ind])])
    
    traj = np.asarray([r,np.rad2deg(lon), np.rad2deg(lat)])

    return t, pos, traj

def full3d(spacecraftlist=['solo', 'psp'], planetlist =['Earth'], t=None, start = '2022-09-01', end = '2022-09-15', traj = False, filepath=None, custom_data=False, save_fig = True, legend = True, title = True):
    
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

    fig=plt.figure(1,figsize=(12,9),dpi=70)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_configure(ax, view_azim=0, view_elev=90, view_radius=0.8)
    
    model_obj = returnmodel(filepath)
    
    plot_3dcore(ax, model_obj, t, color=c2)
    plot_3dcore_field(ax, model_obj, color=c2, step_size=0.005, lw=1.1, ls="-")
    

    
    if 'solo' in spacecraftlist:
        t_solo, pos_solo, traj_solo = getpos('Solar Orbiter', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_satellite(ax,pos_solo,color=solo_color,alpha=0.9, label = 'Solar Orbiter')
        plot_circle(ax,pos_solo[0])  
        if traj == True:
             ax.plot(traj_solo[0]*np.sin(traj_solo[1]),traj_solo[0]*np.cos(traj_solo[1]),0, color=solo_color,alpha=0.9)

        
    if 'psp' in spacecraftlist:
        t_psp, pos_psp, traj_psp  = getpos('Parker Solar Probe', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_satellite(ax,pos_psp,color=psp_color,alpha=0.9, label ='Parker Solar Probe')
        if traj == True:
             ax.plot(traj_psp[0]*np.sin(np.radians(traj_psp[1])),traj_psp[0]*np.cos(np.radians(traj_psp[1])),0, color=psp_color,alpha=0.9)
                
                
    if 'STEREO-A' in spacecraftlist:
        t_solo, pos_solo, traj_solo = getpos('STEREO-A', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_satellite(ax,pos_solo,color=sta_color,alpha=0.9, label = 'STEREO-A')
        
        
    
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
        ax.legend()
    if title == True:
        plt.title('3DCORE fitting result - ' + t.strftime('%Y-%m-%d-%H'))

        

def fullinsitu(observer, t_fit=None, start = None, end=None, filepath=None, custom_data=False, save_fig = True, best = True, ensemble = True, mean = False, legend=True, fixed = None):
    
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
    else:
        observer_obj = custom_observer(custom_data)
        
    t, b = observer_obj.get([start, end], "mag", reference_frame="HEEQ", as_endpoints=True)
    
    pos = observer_obj.trajectory(t, reference_frame="HEEQ")
    
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
    
    # get ensemble_data
    if ensemble == True:
        ed = py3dcore_h4c.generate_ensemble(filepath, t, reference_frame="HEEQ",reference_frame_to="HEEQ", max_index=128, custom_data=custom_data)
    
    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points
    
    if observer == 'solo':
        obs_title = 'Solar Orbiter'

        
    if observer == 'psp':
        obs_title = 'Parker Solar Probe'

    plt.figure(figsize=(20, 10))
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
    for _ in t_fit:
        plt.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")
    if save_fig == True:
        plt.savefig('%s.png' %filepath, dpi=300)    
    plt.show()

    
def returnfixedmodel(filepath, fixed_iparams_arr=None):
    
    ftobj = BaseFitter(filepath) # load Fitter from path
    model_obj = ftobj.model_obj
    
    model_obj.ensemble_size = 1
    
    if fixed_iparams_arr == 'mean':
        logger.info("Plotting run with mean parameters.")
        res, allres, ind, meanparams = get_params(filepath)
        model_obj.iparams_arr = np.expand_dims(meanparams, axis=0)
    
    elif (fixed_iparams_arr == None).any():
        logger.info("No iparams_arr given, using parameters for run with minimum eps.")
        res, allres, ind, meanparams = get_params(filepath)
        model_obj.iparams_arr = np.expand_dims(res, axis=0)
    
    else:
        logger.info("Plotting run with fixed parameters.")
        model_obj.iparams_arr = np.expand_dims(fixed_iparams_arr, axis=0)
    
    model_obj.sparams_arr = np.empty((model_obj.ensemble_size, model_obj.sparams), dtype=model_obj.dtype)
    model_obj.qs_sx = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    model_obj.qs_xs = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    
    model_obj.iparams_meta = np.empty((len(model_obj.iparams), 7), dtype=model_obj.dtype)
    
    #iparams_meta is updated
    generate_quaternions(model_obj.iparams_arr, model_obj.qs_sx, model_obj.qs_xs)
    return model_obj
    
    
    
    
    
    
    
def plot_configure(ax, **kwargs):
    view_azim = kwargs.pop("view_azim", -25)
    view_elev = kwargs.pop("view_elev", 25)
    view_radius = kwargs.pop("view_radius", .5)
    
    ax.view_init(azim=view_azim, elev=view_elev)

    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    ax.set_zlim([-view_radius, view_radius])
    
    ax.set_axis_off()

def plot_3dcore(ax, obj, t_snap, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", .05)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)

    ax.scatter(0, 0, 0, color="y", s=250)

    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(0,)#visualize_wireframe(index=0)
    ax.plot_wireframe(*wf_model.T, **kwargs)

    
def plot_3dcore_field(ax, obj, step_size=0.005, q0=[0.8, .1, np.pi/2],**kwargs):

    #initial point is q0
    q0i =np.array(q0, dtype=np.float32)
    fl = visualize_fieldline(obj, q0, index=0, steps=3000, step_size=step_size)
    #fl = model_obj.visualize_fieldline_dpsi(q0i, dpsi=2*np.pi-0.01, step_size=step_size)
    ax.plot(*fl.T, **kwargs)
    
    
def plot_traj(ax, sat, t_snap, frame="HEEQ", traj_pos=True, traj_major=4, traj_minor=None, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", 1)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)
    kwargs["s"] = kwargs.pop("s", 25)
    
    inst = getattr(heliosat, sat)()

    _s = kwargs.pop("s")

    if traj_pos:
        pos = inst.trajectory(t_snap, frame)

        ax.scatter(*pos.T, s=_s, **kwargs)
        
    if traj_major and traj_major > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_major, traj_major)], frame)
        ax.plot(*traj.T, **kwargs)
        
    if traj_minor and traj_minor > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_minor, traj_minor)], frame)
        
        if "ls" in kwargs:
            kwargs.pop("ls")

        _ls = "--"
        _lw = kwargs.pop("lw") / 2
        
        ax.plot(*traj.T, ls=_ls, lw=_lw, **kwargs)

        
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
    ax.scatter3D(xc,yc,zc,s=50,**kwargs)
    
    
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