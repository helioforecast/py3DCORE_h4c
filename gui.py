import argparse
import datetime as dt
import json
import os
import sys
from typing import List
import py3dcore_h4c
import py3dcore_h4c.gui.guiold as go
from py3dcore_h4c.fitter.base import custom_observer, BaseFitter, get_ensemble_mean
import datetime
import pickle

import py3dcore_h4c.fluxplot as fp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import num2date, date2num, DateFormatter

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QTabWidget, QCheckBox, QFileDialog, QPushButton
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTime, QDateTime, QDate, QSize
from PyQt5.QtGui import QFont

from astropy import units as u
from astropy.coordinates import concatenate
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sunpy import log
from sunpy.coordinates import get_horizons_coord
from sunpy.map import Map
from sunpy.visualization import axis_labels_from_ctype

from py3dcore_h4c.gui.utils.helioviewer import get_helioviewer_client
#from py3dcore_h4c.gui.utils.widgets import SliderAndTextbox

matplotlib.use('Qt5Agg')

hv = get_helioviewer_client()

straight_vertices, front_vertices, circle_vertices = 10, 10, 20
filename = 'gcs_params.json'
draw_modes = ['off', 'point cloud', 'grid']
font_sizes = [10,12,14,16,18]
spacecrafts = ['STA','STB','SOHO']
instruments = ['SECCHI','SECCHI','LASCO']
dets = [['COR2', 'COR1'],['COR2', 'COR1'],['C2', 'C3']]
spatial_units = [None, None]

params = ['CME Launch Time','CME Longitude', 'CME Latitude', 'CME Inclination', 'CME Diameter 1 AU', 'CME Aspect Ratio', 'CME Launch Radius', 'CME Launch Velocity', 'CME Expansion Rate', 'Background Drag', 'Background Velocity']

units = ['h','°', '°', '°', 'AU', '', 'rS','km/s', '', '', 'km/s']

variables = ['\u0394 t','lon', 'lat', 'inc', 'd(1AU)', '\u03B4', 'r\u2080', 'v\u2080', 'n_a', '\u03B3', 'v']

mins = [0, 0, -90, 0, 0.05, 1, 5, 400, 0.3, 0.2, 100]

maxs = [100, 360, 90, 360, 0.35, 6, 100, 1200, 2, 3,700]

inits = [0, 0, 0, 0, 0.2, 3, 20, 800, 1.14, 1, 500]

resolutions = [0.1, 0.1, 0.1, 0.1, 0.01, 1, 1, 1, 0.1, 0.1, 1]


magparams = ['T_factor', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU']
magunits = ['', '', 'nT']
magvariables = ['tau', 'n_b','b(1AU)']
magmins = [-250, 1, 5]
magmaxs = [250, 2, 50]
maginits = [100, 1.64, 25]
magresolutions = [1, 0.01, 1]
# disable sunpy warnings
log.setLevel('ERROR')



#################################################################################

def running_difference(a, b):
    return Map(b.data * 1.0 - a.data * 1.0, b.meta)

def load_image(spacecraft: str, detector: str, date: dt.datetime, runndiff: bool):
    if spacecraft == 'STA':
        observatory = 'STEREO_A'
        instrument = 'SECCHI'
        if detector not in ['COR1', 'COR2']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    elif spacecraft == 'STB':
        observatory = 'STEREO_B'
        instrument = 'SECCHI'
        if detector not in ['COR1', 'COR2']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    elif spacecraft == 'SOHO':
        observatory = 'SOHO'
        instrument = 'LASCO'
        if detector not in ['C2', 'C3']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    else:
        raise ValueError(f'unknown spacecraft: {spacecraft}')

    f = download_helioviewer(date, observatory, instrument, detector)

    if runndiff:
        f2 = download_helioviewer(date - dt.timedelta(hours=1), observatory, instrument, detector)
        return running_difference(f2, f)
    else:
        return f

def download_helioviewer(date, observatory, instrument, detector):
    file = hv.download_jp2(date, observatory=observatory, instrument=instrument, detector=detector)
    f = Map(file)

    if observatory == 'SOHO':
       # add observer location information:
       soho = get_horizons_coord('SOHO', f.date)
       f.meta['HGLN_OBS'] = soho.lon.to('deg').value
       f.meta['HGLT_OBS'] = soho.lat.to('deg').value
       f.meta['DSUN_OBS'] = soho.radius.to('m').value

    return f


###############################################################################################



class py3dcoreGUI(QtWidgets.QWidget): #.QMainWindow
    def __init__(self):
        super().__init__()
        
        # (left, top, width, height)
        self.setGeometry(100,100,2400,1000)
        self.setWindowTitle('3DCORE - GUI')
        
        #set up main window layout
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        ############################### creating sidebar ###############################
        

        sidebar_layout = QVBoxLayout()
        
        # CALENDAR
        self.side_tab_widget = QTabWidget()
        self.side_tab_widget.tabBar().setVisible(False)

        
        sidebar_tab_layout1 = QVBoxLayout()
        
        date_label = QLabel()
        date_label.setText('Select Date:')
        sidebar_tab_layout1.addWidget(date_label)
        calendar = QtWidgets.QCalendarWidget()
        self.calendar = calendar
        self.calendar.setToolTip('Select a date. JHelioviewer will be used to obtain the heliospheric image closest to the date you selected.') 
        sidebar_tab_layout1.addWidget(self.calendar)
        date = QDate(2020, 4, 15)
        self.calendar.setSelectedDate(date)
        self.calendar.clicked.connect(self.update_canvas)
        
        # TIME
        emptylabel = QLabel()
        emptylabel.setText('')
        sidebar_tab_layout1.addWidget(emptylabel)

        time_label = QLabel("Select Time:")
        self.time_combobox = QComboBox()
        self.time_combobox.setToolTip('Select a time. JHelioviewer will be used to obtain the heliospheric image closest to the time you selected.') 
        for hour in range(24):
            for minute in [0, 30]:
                time = QTime(hour, minute)
                self.time_combobox.addItem(time.toString("h:mm AP"))
        currenttime = QTime(6, 0)
        self.time_combobox.setCurrentText(currenttime.toString("h:mm AP"))
        self.time_combobox.currentIndexChanged.connect(self.update_canvas)
        sidebar_tab_layout1.addWidget(time_label)
        sidebar_tab_layout1.addWidget(self.time_combobox)
       
        # SPACECRAFTs
        
        
        sidebar_tab_layout1.addWidget(emptylabel)

        self.checkboxes = []
        self.meshplots = []
        
        spaecrafts_label = QLabel()
        spaecrafts_label.setText('Spacecraft and Instrument:')
        spaecrafts_label.setToolTip('Select the spacecraft you want to load images using JHelioviewer for.') 
        sidebar_tab_layout1.addWidget(spaecrafts_label)
        
        vlayout_sta = QHBoxLayout()
        checkbox_sta = QCheckBox("STEREO-A")
        checkbox_sta.setChecked(True)
        checkbox_sta.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_sta)
        vlayout_sta.addWidget(checkbox_sta)
        self.instr_combobox_sta = QComboBox()
        for i in dets[0]:
            self.instr_combobox_sta.addItem(i)
        self.instr_combobox_sta.currentIndexChanged.connect(self.update_canvas)
        vlayout_sta.addWidget(self.instr_combobox_sta)
        sidebar_tab_layout1.addLayout(vlayout_sta)
        
        vlayout_stb = QHBoxLayout()
        checkbox_stb = QCheckBox("STEREO-B")
        checkbox_stb.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_stb)
        vlayout_stb.addWidget(checkbox_stb)
        self.instr_combobox_stb = QComboBox()
        for i in  dets[1]:
            self.instr_combobox_stb.addItem(i)
        self.instr_combobox_stb.currentIndexChanged.connect(self.update_canvas)
        vlayout_stb.addWidget(self.instr_combobox_stb)
        sidebar_tab_layout1.addLayout(vlayout_stb)
        
        vlayout_soho = QHBoxLayout()
        checkbox_soho = QCheckBox("SOHO")
        checkbox_soho.setChecked(True)
        checkbox_soho.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_soho)
        vlayout_soho.addWidget(checkbox_soho)
        self.instr_combobox_soho = QComboBox()
        for i in  dets[2]:
            self.instr_combobox_soho.addItem(i)
        self.instr_combobox_stb.currentIndexChanged.connect(self.update_canvas)
        vlayout_soho.addWidget(self.instr_combobox_soho)
        sidebar_tab_layout1.addLayout(vlayout_soho)
        
        
        # 3DCORE MODEL
        
        sidebar_tab_layout1.addWidget(emptylabel)

        core_label = QLabel()
        core_label.setText('Draw Grid:')
        sidebar_tab_layout1.addWidget(core_label)
        self.corebox = QCheckBox("3DCORE Flux Rope Model")
        self.corebox.setToolTip('The 3DCORE Flux Rope Model was originally created by Andreas J. Weiss.') 
        self.corebox.stateChanged.connect(self.update_canvas)
        sidebar_tab_layout1.addWidget(self.corebox)
        
        sidebar_tab_widget1 = QWidget()
        sidebar_tab_widget1.setLayout(sidebar_tab_layout1)
        self.side_tab_widget.addTab(sidebar_tab_widget1, "General")

        # create second tab
        
        sidebar_tab_layout2 = QVBoxLayout()
        sidebar_tab_layout2.setAlignment(Qt.AlignTop)
        
        # launchtime
        
        timerangelabel = QLabel("Select Range:")
        timerangelabel.setToolTip('Select start and end time to be shown in the insitu plot.') 
        sidebar_tab_layout2.addWidget(timerangelabel)
                
        startselectlayout = QHBoxLayout()
        startselectlabel = QLabel("Start Time: ")
        startselectlayout.addWidget(startselectlabel)
        startselectlayout.addStretch(1)
        self.startselect = QtWidgets.QDateTimeEdit()
        self.startselect.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.startselect.setDateTime(QDateTime(2020, 4, 15,0,0))
        #self.startselect.setCalendarPopup(True)
        self.startselect.dateTimeChanged.connect(self.plot_mesh)
        startselectlayout.addWidget(self.startselect)
        
        endselectlayout = QHBoxLayout()
        endselectlabel = QLabel("End Time: ")
        endselectlayout.addWidget(endselectlabel)
        endselectlayout.addStretch(1)
        self.endselect = QtWidgets.QDateTimeEdit()
        self.endselect.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.endselect.setDateTime(QDateTime(2020, 4, 15,18,0))
        #self.startselect.setCalendarPopup(True)
        self.endselect.dateTimeChanged.connect(self.plot_mesh)
        endselectlayout.addWidget(self.endselect)
        
        
        # OBSERVER
        observer_label = QLabel("Select Observer:")
        self.observer_combobox = QComboBox()
        self.observer_combobox.addItem('')
        self.observer_combobox.currentIndexChanged.connect(self.observer_changed)
        
        #sidebar_tab_layout2.addWidget(emptylabel)
        #sidebar_tab_layout2.addWidget(emptylabel)
        sidebar_tab_layout2.addLayout(startselectlayout)
        sidebar_tab_layout2.addLayout(endselectlayout)
        #sidebar_tab_layout2.addWidget(emptylabel)

        sidebar_tab_layout2.addWidget(observer_label)
        sidebar_tab_layout2.addWidget(self.observer_combobox)
        
        ### SYNTHETIC BOX
        
        #sidebar_tab_layout2.addWidget(emptylabel)

        synthetic_label = QLabel()
        synthetic_label.setText('Draw in Plot:')
        sidebar_tab_layout2.addWidget(synthetic_label)
        self.syntheticbox = QCheckBox("Synthetic In Situ Data")
        self.syntheticbox.setToolTip('The 3DCORE Flux Rope Model was originally created by Andreas J. Weiss.') 
        self.syntheticbox.stateChanged.connect(self.plot_mesh)
        sidebar_tab_layout2.addWidget(self.syntheticbox)
        
        self.legendbox = QCheckBox("Legend")
        self.legendbox.stateChanged.connect(self.plot_mesh)
        sidebar_tab_layout2.addWidget(self.legendbox)
        
        self.fitlinesbox = QCheckBox("Fit Positions")
        self.fitlinesbox.setToolTip('Show lines to indicate which values were used during fitting.') 
        self.fitlinesbox.stateChanged.connect(self.plot_mesh)
        sidebar_tab_layout2.addWidget(self.fitlinesbox)
        
        
        sidebar_tab_widget2 = QWidget()
        sidebar_tab_widget2.setLayout(sidebar_tab_layout2)
        self.side_tab_widget.addTab(sidebar_tab_widget2, "Magnetic Field")
        
        sidebar_layout.addWidget(emptylabel)
        sidebar_layout.addWidget(self.side_tab_widget) #, 4)
        
        sidebar_layout.addStretch(1)
        
        
        ## create general buttons outside of the tabview
        # FONT SIZE
        font_size_label = QLabel("Font Size:")
        self.font_size_combobox = QComboBox()
        for i in font_sizes:
            self.font_size_combobox.addItem(str(i))
        self.font_size_combobox.setCurrentIndex(1)
        self.font_size_combobox.currentIndexChanged.connect(self.update_canvas)
        sidebar_layout.addWidget(font_size_label)
        sidebar_layout.addWidget(self.font_size_combobox)
        
        sidebar_layout.addWidget(emptylabel)
        vlayout_buttons = QHBoxLayout()
        self.selectbutton = QPushButton('Load Pickle File')
        self.selectbutton.setToolTip('Load a pickle file from a previous fitting run. All parameters will be set according to the file.') 
        self.selectbutton.clicked.connect(self.load)
        vlayout_buttons.addWidget(self.selectbutton)
        self.savebutton = QPushButton('Save Image')
        self.savebutton.setToolTip('Save the currently shown image.') 
        self.savebutton.clicked.connect(self.save)
        vlayout_buttons.addWidget(self.savebutton)
        sidebar_layout.addLayout(vlayout_buttons)
        
        
        # add the sidebar to the main layout
        main_layout.addLayout(sidebar_layout, 0)
        
        ############################### creating main canvas and time slider ---- TAB 1 ###########################
        
        plotdate = dt.datetime(2020,4,15,6)
        runndiff = False
        
        middle_layout = QVBoxLayout()
        
        self.middle_tab_widget = QTabWidget()
        
        self.middle_tab_widget.currentChanged.connect(self.tab_changed)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.middle_tab_widget.addTab(self.canvas, "3D Model") #middle_layout.addWidget(self.canvas, 4)
        self.middle_tab_widget.setTabToolTip(0, 'Shows a 3D Model over Heliospheric Images.') 
        
        self.subplots = []
        self.images = []
        self.checkedsc = []
        self.iparams_list = inits
        self.magiparams_list = maginits
        
        staimage = load_image('STA', 'COR2', plotdate, runndiff)
        self.ax1 = self.fig.add_subplot(121, projection = staimage)
        staimage.plot(axes=self.ax1, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        self.images.append(staimage)
        ctype = self.ax1.wcs.wcs.ctype
        self.ax1.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=12)
        self.ax1.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=12)

        self.ax1.tick_params(axis='x', labelsize=12)
        self.ax1.tick_params(axis='y', labelsize=12)
        self.ax1.title.set_size(12+2)
        self.subplots.append(self.ax1)
        self.checkedsc.append('STA')
        
        sohoimage = load_image('SOHO', 'C2', plotdate, runndiff)
        self.ax2 = self.fig.add_subplot(122, projection = sohoimage)
        sohoimage.plot(axes=self.ax2, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        self.images.append(sohoimage)
        ctype = self.ax2.wcs.wcs.ctype
        self.font_size=12
        self.ax2.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=self.font_size)
        self.ax2.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=self.font_size)
        self.ax2.tick_params(axis='x', labelsize=12)
        self.ax2.tick_params(axis='y', labelsize=12)
        self.ax2.title.set_size(12+2)
        self.subplots.append(self.ax2)
        self.checkedsc.append('SOHO')
        
        # Create the second tab layout
        self.fig2 = Figure(figsize=(10, 5), dpi=100)
        self.canvas2 = FigureCanvas(self.fig2)
        self.middle_tab_widget.addTab(self.canvas2, "Insitu Mag")
        self.middle_tab_widget.setTabToolTip(1, 'Shows synthetic in situ data generated from the model.') 
        
        self.ax1insitu = self.fig2.add_subplot(111)
        infotext = "Please load the pickle file for a specific fitting run. Generating synthetic insitu data does highly depend \non the selected parameters and the whole parameter range cannot be searched by hand."
        self.ax1insitu.text(0.5, 0.5, infotext, ha='center', va='center', fontsize=12)
        self.ax1insitu.set_axis_off()
        self.ax1insitu.set_xticks([])
        self.ax1insitu.set_yticks([])

        # Add the tab widget to the main layout
        middle_layout.addWidget(self.middle_tab_widget, 4)
        
        vlayout_dt = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setToolTip('Here you can propagate a model forward and backward in time. The images in the background will not be modified, but the assumed launch time changes according to how far the cme has propagated.') 
        slider.setRange(0, 1000)
        slider.setValue(0)
        slider.setTickInterval(10)
        vlayout_dt.addWidget(slider)
        self.dt_val = QLabel("\u0394 t: {} h".format(0))
        vlayout_dt.addWidget(self.dt_val)
        
        
        ############################### creating main canvas ---- TAB 2 ###########################
        
        middle_layout.addLayout(vlayout_dt)
        
        main_layout.addLayout(middle_layout,5)
        
        
        ############################### creating right sidebar ###############################
    
        rightbar_layout = QVBoxLayout()
        #rightbar_layout.setAlignment(Qt.AlignTop)
        
        # launchtime
        
        dtlayout = QHBoxLayout()
        
        dtlabel = QLabel("Assumed Launch Time:                     ")
        dtlabel.setToolTip('The assumed Launch Time is calculated using the date shown in the images, as well as the time which has passed according to the slider below.') 
        dtlayout.addWidget(dtlabel)
        dtlayout.addStretch(1)
        
        self.dtlabelupdating = QtWidgets.QDateTimeEdit() #QLabel(plotdate.strftime('%Y-%m-%d %H:%M:00'))
        self.dtlabelupdating.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dtlabelupdating.setDateTime(QDateTime(2020, 4, 15,6,0))
        #self.dtlabelupdating.setKeyboardTracking(False)
        self.dtlabelupdating.dateTimeChanged.connect(self.handledatetimebox)
        dtlayout.addWidget(self.dtlabelupdating)
        #rightbar_layout.addLayout(dtlayout)
        
        ####### TABVIEW
        
        right_tab_widget = QTabWidget()
        
        # Create the first tab layout
        rightbar_tab_layout1 = QVBoxLayout()
        #rightbar_tab_layout1.setAlignment(Qt.AlignTop)       
        
        # parameters
        
        self.paramlabels = []
        self.paramsliders = []
        
        for i in range(11):
            if i == 0:
                self.paramsliders.append(slider)
                self.paramlabels.append(dtlabel)
            else:
                rightbar_tab_layout1.addWidget(emptylabel)
                hlayout = QHBoxLayout()
                label = QLabel(params[i])
                hlayout.addWidget(label)
                hlayout.addStretch(1)
                updatelabel = QLabel('{}: {} {}'.format(variables[i], inits[i], units[i]))
                hlayout.addWidget(updatelabel)
                self.paramlabels.append(updatelabel)
                rightbar_tab_layout1.addLayout(hlayout)
            
                slider = QSlider(Qt.Horizontal)
                slider.setRange(int(mins[i]/resolutions[i]),int(maxs[i]/resolutions[i]))
                slider.setValue(int(inits[i]/resolutions[i]))
                self.paramsliders.append(slider)
                rightbar_tab_layout1.addWidget(slider)
        
        for slider in self.paramsliders:
            slider.valueChanged.connect(self.plot_mesh)
            
        rightbar_tab_widget1 = QWidget()
        rightbar_tab_widget1.setLayout(rightbar_tab_layout1)
        right_tab_widget.addTab(rightbar_tab_widget1, "General")
            
        
        # Create the first tab layout
        rightbar_tab_layout2 = QVBoxLayout()
        rightbar_tab_layout2.setAlignment(Qt.AlignTop)
        
        # magparameters
        
        self.magparamlabels = []
        self.magparamsliders = []
        
        for i in range(3):
            rightbar_tab_layout2.addWidget(emptylabel)
            hlayout = QHBoxLayout()
            label = QLabel(magparams[i])
            hlayout.addWidget(label)
            hlayout.addStretch(1)
            updatelabel = QLabel('{}: {} {}'.format(magvariables[i], maginits[i], magunits[i]))
            hlayout.addWidget(updatelabel)
            self.magparamlabels.append(updatelabel)
            rightbar_tab_layout2.addLayout(hlayout)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(magmins[i]/magresolutions[i]),int(magmaxs[i]/magresolutions[i]))
            slider.setValue(int(maginits[i]/magresolutions[i]))
            self.magparamsliders.append(slider)
            rightbar_tab_layout2.addWidget(slider)
        
        for slider in self.magparamsliders:
            slider.valueChanged.connect(self.plot_mesh)
        
        
        
        rightbar_tab_widget2 = QWidget()
        rightbar_tab_widget2.setLayout(rightbar_tab_layout2)
        right_tab_widget.addTab(rightbar_tab_widget2, "Magnetic Field")
        
        right_tab_widget.setTabToolTip(0, 'Modify general parameters of the model.') 
        right_tab_widget.setTabToolTip(0, 'Modify magnetic field parameters of the model.') 
        # Add the tab widget to the main layout
        #rightbar_layout.setAlignment(Qt.AlignTop)
        
        rightbar_layout.addLayout(dtlayout)
        rightbar_layout.addWidget(emptylabel)
        rightbar_layout.addWidget(right_tab_widget) #, 4)
        rightbar_layout.addStretch(1)
        
        vlayout_buttons = QHBoxLayout()
        self.resetbutton = QPushButton('Reset Parameters')
        self.resetbutton.setToolTip('This resets all parameters and their ranges.') 
        self.resetbutton.clicked.connect(self.reset)
        vlayout_buttons.addStretch(1)
        vlayout_buttons.addWidget(self.resetbutton)
        rightbar_layout.addLayout(vlayout_buttons)
        

        # save / load
        
        #sidebar_layout.addStretch(1)
        

        # add the sidebar to the main layout
        main_layout.addLayout(rightbar_layout)
        

        # show the main window
        self.show()
        
    def update_canvas(self):
        self.runndiff = False
        # get the selected font size from the combobox
        self.font_size = int(self.font_size_combobox.currentText())
        selected_date = self.calendar.selectedDate().toPyDate()
        selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
        self.plotdate = dt.datetime.combine(selected_date, selected_time)
        
        
        #update font size everywhere
        
        # set the font size of all relevant widgets
        font = QFont()
        font.setPointSize(self.font_size)
    
        for label in self.findChildren(QLabel):
            label.setFont(font)
        for label in self.findChildren(QSlider):
            label.setFont(font)
        for label in self.findChildren(QComboBox):
            label.setFont(font)
        for label in self.findChildren(QCheckBox):
            label.setFont(font)
        self.calendar.setFont(font)

        # create a new subfigure for each checkbox that is checked
        self.images = []
        
        self.checkedsc = []
        checkeddet = []
        currentdets = [self.instr_combobox_sta.currentText(),self.instr_combobox_stb.currentText(),self.instr_combobox_soho.currentText()]
        
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.checkedsc.append(spacecrafts[i])
                checkeddet.append(currentdets[i])
                
        for i, sc in enumerate(self.checkedsc):
            image = load_image(sc, checkeddet[i], self.plotdate, self.runndiff)
            self.images.append(image)
    
        
        ########self._bg = fig.canvas.copy_from_bbox(self.fig.bbox)
        # redraw the canvas
        self.fig.canvas.draw()
        self.plot_mesh()
        
    def handledatetimebox(self):
        
        selected_date = self.calendar.selectedDate().toPyDate()
        selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
        plotdate = dt.datetime.combine(selected_date, selected_time)
        minlaunchtime = plotdate - datetime.timedelta(hours = self.paramsliders[0].maximum()/10)
        if self.dtlabelupdating.dateTime().toPyDateTime() < minlaunchtime:
            self.paramsliders[0].setValue(int(0))
        else:
            deltatt = plotdate - self.dtlabelupdating.dateTime().toPyDateTime()
            self.paramsliders[0].setValue(int(deltatt.total_seconds()/3600*10))
        
    def tab_changed(self):
        self.side_tab_widget.setCurrentIndex(self.middle_tab_widget.currentIndex())
    
    def observer_changed(self):
        self.currentobserver = self.fitobservers[self.observer_combobox.currentIndex()]
        _, _, starttime, endtime, _, self.insitudatapath = self.currentobserver
        self.startselect.setDateTime(QDateTime.fromString(starttime.strftime('%Y-%m-%d %H:%M:00'),'yyyy-M-d hh:mm:ss'))
        self.endselect.setDateTime(QDateTime.fromString(endtime.strftime('%Y-%m-%d %H:%M:00'),'yyyy-M-d hh:mm:ss'))
        
    def plot_mesh(self):
        # clear the existing subfigures
        self.fig.clf()
        self.subplots =[]
        self.insitusubplots =[]
        self.runndiff = False
        sender = self.sender() 
        self.iparams_list = []
        self.magiparams_list = []
        self.starttime = self.startselect.dateTime().toPyDateTime()
        self.endtime = self.endselect.dateTime().toPyDateTime()
        
        for i, label in enumerate(self.paramlabels):
            if i == 0:
                self.dt_val.setText("\u0394 t: {} h".format(self.paramsliders[i].value()/10))
                selected_date = self.calendar.selectedDate().toPyDate()
                selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
                plotdate = dt.datetime.combine(selected_date, selected_time)
                launchtime = plotdate - datetime.timedelta(hours = self.paramsliders[i].value()/10)
                self.dtlabelupdating.setDateTime(QDateTime.fromString(launchtime.strftime('%Y-%m-%d %H:%M:00'),'yyyy-M-d hh:mm:ss'))
            else:
                
                if resolutions[i] == 1:
                    val = int(self.paramsliders[i].value()*resolutions[i])
                elif resolutions[i] == 0.1:
                    val = float("{:.1f}".format(self.paramsliders[i].value()*resolutions[i]))
                elif resolutions[i] == 0.01:
                    val = float("{:.2f}".format(self.paramsliders[i].value()*resolutions[i]))
                self.iparams_list.append(val)
                label.setText('{}: {} {}'.format(variables[i], val, units[i]))
                
        for i, label in enumerate(self.magparamlabels):
            if magresolutions[i] == 1:
                val = int(self.magparamsliders[i].value()*magresolutions[i])
            elif magresolutions[i] == 0.1:
                val = float("{:.1f}".format(self.magparamsliders[i].value()*magresolutions[i]))
            elif magresolutions[i] == 0.01:
                val = float("{:.2f}".format(self.magparamsliders[i].value()*magresolutions[i]))
            self.magiparams_list.append(val)
            label.setText('{}: {} {}'.format(magvariables[i], val, magunits[i]))

               
        for i, image in enumerate(self.images):
            self.subplots.append(self.fig.add_subplot( 1, len(self.checkedsc),i+1, projection=image))
            image.plot(axes= self.subplots[-1], cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if self.runndiff else None)
            ctype = self.subplots[-1].wcs.wcs.ctype
            self.subplots[-1].set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=self.font_size)
            self.subplots[-1].set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=self.font_size)
            self.subplots[-1].tick_params(axis='x', labelsize=self.font_size)
            self.subplots[-1].tick_params(axis='y', labelsize=self.font_size)
            self.subplots[-1].title.set_size(self.font_size+2)
            
        
        iparams = self.get_iparams()
               
        timedelta = self.paramsliders[0].value()/10
        if self.corebox.isChecked():
            t_snap = datetime.datetime(2020,4,15,6)
            dt_0 = t_snap - datetime.timedelta(hours = timedelta)
            mesh = go.py3dcore_mesh_sunpy(dt_0, t_snap,iparams)
            for i, (image, subplot) in enumerate(zip(self.images, self.subplots)):
               # if len(self.meshplots) <= i:
                style = '-'
                params = dict(lw=0.5)
                p = subplot.plot_coord(mesh.T, color='blue', scalex=False, scaley=False, **params)[0]
                p2 = subplot.plot_coord(mesh, color='blue', scalex=False, scaley=False, **params)[0]
            self.fig.canvas.draw()
            
        self.fig2.clf()
        if self.observer_combobox.currentText() != "":
            observer_obj = custom_observer(self.insitudatapath)
            t, b = observer_obj.get([self.starttime,self.endtime], "mag", reference_frame="HEEQ", as_endpoints=True)
            pos = observer_obj.trajectory(t, reference_frame="HEEQ")
            if self.currentobserver[0] == 'solo':
                obs_title = 'Solar Orbiter'
            elif self.currentobserver[0] == 'PSP':
                obs_title = 'Parker Solar Probe'
                
            self.insituax = self.fig2.add_subplot(1,1,1)
            self.insituax.plot(t, np.sqrt(np.sum(b**2, axis=1)), "k", alpha=0.5, lw=3, label ='Btotal')
            self.insituax.plot(t, b[:, 0], "r", alpha=1, lw=2, label ='Br')
            self.insituax.plot(t, b[:, 1], "g", alpha=1, lw=2, label ='Bt')
            self.insituax.plot(t, b[:, 2], "b", alpha=1, lw=2, label ='Bn')
            
            self.fig2.suptitle("Magnetic Field In Situ Data - "+obs_title)
            
            date_form = mdates.DateFormatter("%h %d %H")
            self.insituax.xaxis.set_major_formatter(date_form)
            
            self.insituax.set_ylabel("B [nT]")
            self.insituax.set_xticks(t[::len(t)//10])
            self.insituax.set_xticklabels([date.strftime('%b %d %H') for date in t[::len(t)//10]], rotation=25, ha='right')

            
            
            if self.legendbox.isChecked():
                self.insituax.legend(loc='lower right')
            if self.fitlinesbox.isChecked():
                t_fit = self.currentobserver[1]
                for _ in t_fit:
                    self.insituax.axvline(x=_, lw=2, alpha=0.25, color="k", ls="--")
                      
            if self.syntheticbox.isChecked():
                    
                model_obj = py3dcore_h4c.ToroidalModel(dt_0, **iparams) # model gets initialized
                model_obj.generator()
                #model_obj = fp.returnfixedmodel(self.filename, fixed_iparams_arr='mean')
                outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
                outa[outa==0] = np.nan
                self.insituax.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=3)
                self.insituax.plot(t, outa[:, 0], "r", alpha=0.5, linestyle='dashed', lw=3)
                self.insituax.plot(t, outa[:, 1], "g", alpha=0.5, linestyle='dashed', lw=3)
                self.insituax.plot(t, outa[:, 2], "b", alpha=0.5, linestyle='dashed', lw=3)
                
            self.fig2.canvas.draw()
        

        else:
            return
        
    def load(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Pickle File", "", "Pickle Files (*.pkl *.pickle *.p)", options=options)
        if fileName:
            self.filename = fileName
            # read from pickle file
            file = open(fileName, "rb")
            data = pickle.load(file)
            file.close()
            observers = data["observers"]
            self.fitobservers = observers
            self.observer_combobox.clear()
            for obs in observers:
                self.observer_combobox.addItem(obs[0])
            t_0 = data["dt_0"]
            deltat = 2
            model_objt = data["model_obj"]
            iparams_arrt = model_objt.iparams_arr    
            meanparams = np.mean(model_objt.iparams_arr, axis=0)
            maxparams = np.mean(model_objt.iparams_arr, axis=0) + np.std(model_objt.iparams_arr, axis=0)
            minparams = np.mean(model_objt.iparams_arr, axis=0) - np.std(model_objt.iparams_arr, axis=0)
            
            
            self.paramsliders[0].setValue(int(deltat/resolutions[0]))
            
            self.paramsliders[1].setValue(int(meanparams[1]/resolutions[1]))
            self.paramsliders[1].setRange(int(minparams[1]/resolutions[1]),int(maxparams[1]/resolutions[1]))
            
            self.paramsliders[2].setValue(int(meanparams[2]/resolutions[2]))
            self.paramsliders[2].setRange(int(minparams[2]/resolutions[2]),int(maxparams[2]/resolutions[2]))
            
            self.paramsliders[3].setValue(int(meanparams[3]/resolutions[3]))
            self.paramsliders[3].setRange(int(minparams[3]/resolutions[3]),int(maxparams[3]/resolutions[3]))
            
            self.paramsliders[4].setValue(int(meanparams[4]/resolutions[4]))
            self.paramsliders[4].setRange(int(minparams[4]/resolutions[4]),int(maxparams[4]/resolutions[4]))
            
            self.paramsliders[5].setValue(int(meanparams[5]/resolutions[5]))
            self.paramsliders[5].setRange(int(minparams[5]/resolutions[5]),int(maxparams[5]/resolutions[5]))
            
            self.paramsliders[6].setValue(int(meanparams[6]/resolutions[6]))
            self.paramsliders[6].setRange(int(minparams[6]/resolutions[6]),int(maxparams[6]/resolutions[6]))
            
            self.paramsliders[7].setValue(int(meanparams[7]/resolutions[7]))
            self.paramsliders[7].setRange(int(minparams[7]/resolutions[7]),int(maxparams[7]/resolutions[7]))
            
            self.paramsliders[8].setValue(int(meanparams[9]/resolutions[8]))
            
            self.paramsliders[9].setValue(int(meanparams[12]/resolutions[9]))
            self.paramsliders[9].setRange(int(minparams[12]/resolutions[9]),int(maxparams[12]/resolutions[9]))
            
            self.paramsliders[10].setValue(int(meanparams[13]/resolutions[10]))
            self.paramsliders[10].setRange(int(minparams[13]/resolutions[10]),int(maxparams[13]/resolutions[10]))
            
            ## mag params
            
            self.magparamsliders[0].setValue(int(meanparams[8]/magresolutions[0]))
            self.magparamsliders[0].setRange(int(minparams[8]/magresolutions[0]),int(maxparams[8]/magresolutions[0]))
            self.magparamsliders[1].setValue(int(meanparams[10]/magresolutions[1]))
            self.magparamsliders[2].setValue(int(meanparams[11]/magresolutions[2]))
            self.magparamsliders[2].setRange(int(minparams[11]/magresolutions[2]),int(maxparams[11]/magresolutions[2]))
            
            self.corebox.setChecked(True)
            plotttdate = t_0 + datetime.timedelta(hours = deltat)
            date = QDate(plotttdate.year, plotttdate.month, plotttdate.day)
            self.calendar.setSelectedDate(date)
            if plotttdate.minute < 15:
                minute = 0
                hour = plotttdate.hour
            elif plotttdate.minute < 45:
                minute = 30
                hour = plotttdate.hour
            elif plotttdate.minute < 60:
                minute = 0
                hour = plotttdate.hour + 1
            time = QTime(hour,minute)
            self.time_combobox.setCurrentText(time.toString("h:mm AP"))

    def save(self):
        # Get file name and type from user
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("JPG files (*.jpg);;PDF files (*.pdf);;PNG files (*.png)")
        file_dialog.setDefaultSuffix('png')
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        if file_dialog.exec_() == QFileDialog.Accepted:
            file_name = file_dialog.selectedFiles()[0]
        else:
            return

        # Get canvas pixmap and save to file
        pixmap = self.canvas.grab()
        pixmap.save(file_name)
        
    def save(self):
        # Get file name and type from user
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("JPG files (*.jpg);;PDF files (*.pdf);;PNG files (*.png)")
        file_dialog.setDefaultSuffix('png')
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        if file_dialog.exec_() == QFileDialog.Accepted:
            file_name = file_dialog.selectedFiles()[0]
        else:
            return

        # Get canvas pixmap and save to file
        pixmap = self.canvas.grab()
        pixmap.save(file_name)
        
    def reset(self):
        
        for i, slider in enumerate(self.paramsliders):
            slider.setRange(int(mins[i]/resolutions[i]),int(maxs[i]/resolutions[i]))
            slider.setValue(int(inits[i]/resolutions[i]))
            
        for i, slider in enumerate(self.magparamsliders):
            slider.setRange(int(magmins[i]/magresolutions[i]),int(magmaxs[i]/magresolutions[i]))
            slider.setValue(int(maginits[i]/magresolutions[i]))
        
    def get_iparams(self):
        model_kwargs = {
            "ensemble_size": int(1), #2**17
            "iparams": {
                "cme_longitude": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[0],
                    "maximum": 360,
                    "minimum": 0
                },
                "cme_latitude": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[1],
                    "maximum": 90,
                    "minimum": -90
                },
                "cme_inclination": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[2],
                    "maximum": 360,
                    "minimum": 0
                }, 
                "cme_diameter_1au": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[3],
                    "maximum": 5,
                    "minimum": 0.05
                }, 
                "cme_aspect_ratio": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[4],
                    "maximum": 6,
                    "minimum": 1
                },
                "cme_launch_radius": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[5],
                    "maximum": 100,
                    "minimum": 5
                },
                "cme_launch_velocity": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[6],
                    "maximum": 1500,
                    "minimum": 400
                },
                "t_factor": {
                    "distribution": "fixed",
                    "default_value": self.magiparams_list[0],
                    "maximum": 250,
                    "minimum": -250
                },
                "magnetic_decay_rate": {
                    "distribution": "fixed",
                    "default_value": self.magiparams_list[1],
                    "maximum": 2,
                    "minimum": 1
                },
                
                "magnetic_field_strength_1au": {
                    "distribution": "fixed",
                    "default_value": self.magiparams_list[2],
                    "maximum": 50,
                    "minimum": 5
                },
                
                "cme_expansion_rate": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[7],
                    "maximum": 4,
                    "minimum": 0.3
                }, 
                "background_drag": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[8],
                    "maximum": 4,
                    "minimum": 0.2
                }, 
                "background_velocity": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[9],
                    "maximum": 1000,
                    "minimum": 100
                } 
            }
        }
        return model_kwargs


def main():
    qapp = QtWidgets.QApplication(sys.argv)
    app = py3dcoreGUI()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()
    
    
    
