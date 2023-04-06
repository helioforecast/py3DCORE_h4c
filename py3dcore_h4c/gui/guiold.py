import py3dcore_h4c.fluxplot as fp
import py3dcore_h4c
import datetime as datetime
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from typing import Any, List, Optional, Sequence, Type, Union

def py3dcore_mesh_rotated(dt_0 = 'now', t_snap = None, delta = 10, model_kwargs: dict = {}):
    mesh = py3dcore_mesh(dt_0, t_snap)
    mesh = rotate_mesh(mesh, [lon, lat, tilt])

def rotate_mesh(mesh, neang):
    return Rotation.from_euler('zyx', [neang[2], neang[1], neang[0]]).apply(mesh)
    
def py3dcore_mesh(dt_0 = 'now', t_snap = None, model_kwargs: dict = {}):
    if model_kwargs == {}:
        model_kwargs = get_iparams()
    if dt_0 == 'now':
        dt_0 = datetime.datetime.now().replace(microsecond=0, second=0, minute=0)    
    obj = py3dcore_h4c.ToroidalModel(dt_0, **model_kwargs) # model gets initialized
    obj.generator()
    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(0,)
    return wf_model
    
def py3dcore_mesh_sunpy(dt_0 = 'now', t_snap = None, model_kwargs: dict = {}):
    
    wf_model = py3dcore_mesh(dt_0, t_snap, model_kwargs)

    mc= SkyCoord(wf_model.T[0,:,:]*u.AU, wf_model.T[1,:,:]*u.AU, wf_model.T[2,:,:]*u.AU, frame='heliographic_stonyhurst',representation_type='cartesian')


    #m = mesh.T[[2, 1, 0], :] * sun.constants.radius
    #m[1, :] *= -1
    #mesh_coord = SkyCoord(*(m), frame=frames.HeliographicStonyhurst,
                          #obstime=date, representation_type='cartesian')
    return mc


def guiplot_3d(dt_0 = 'now', delta = 10, model_kwargs: dict = {}):
    if model_kwargs == {}:
        model_kwargs = get_iparams()
    if dt_0 == 'now':
        dt_0 = datetime.datetime.now().replace(microsecond=0, second=0, minute=0)
    model_obj = py3dcore_h4c.ToroidalModel(dt_0, **model_kwargs) # model gets initialized
    
    sns.set_context("talk")     

    sns.set_style("ticks",{'grid.linestyle': '--'})
    fsize=15

    fig = plt.figure(figsize=(15,12),dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    t = dt_0 + datetime.timedelta(hours=delta)
    
    fp.plot_configure(ax, view_azim=0, view_elev=90, view_radius=0.8)
    fp.plot_3dcore(ax, model_obj, t , color="xkcd:blue")

    return fig

def guiplot_insitu(ax, start, end):
    return fig

def get_iparams():
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": 150,
                "maximum": 200,
                "minimum": 100
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": 0,
                "maximum": 10,
                "minimum": -10
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": 0,
                "maximum": 30,
                "minimum": 0
            }, 
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": 0.2,
                "maximum": 0.35,
                "minimum": 0.05
            }, 
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": 3,
                "maximum": 5,
                "minimum": 1
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": 15,
                "maximum": 15,
                "minimum": 14
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": 800,
                "maximum": 1000,
                "minimum": 400
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": 100,
                "maximum": 250,
                "minimum": 50
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": 25,
                "maximum": 50,
                "minimum": 5
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": 1,
                "maximum": 2,
                "minimum": 0.2
            }, 
            "background_velocity": {
                "distribution": "fixed",
                "default_value": 500,
                "maximum": 700,
                "minimum": 400
            } 
        }
    }
    return model_kwargs