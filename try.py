from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("PyQt5 Application with TabWidget")
        
        # Create the main widget and layout
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the first tab widget
        tab_widget1 = QTabWidget(self)
        main_layout.addWidget(tab_widget1)
        
        # Create the first tab
        tab1 = QWidget(self)
        tab_widget1.addTab(tab1, "Tab 1")
        
        # Create the second tab
        tab2 = QWidget(self)
        tab_widget1.addTab(tab2, "Tab 2")
        
        # Set the visibility of the tabBar to False
        tab_widget1.tabBar().setVisible(False)
        
        # Create the second tab widget
        tab_widget2 = QTabWidget(self)
        main_layout.addWidget(tab_widget2)
        
        # Create the third tab
        tab3 = QWidget(self)
        tab_widget2.addTab(tab3, "Tab 3")
        
        # Create the fourth tab
        tab4 = QWidget(self)
        tab_widget2.addTab(tab4, "Tab 4")
        
        # Connect the currentChanged signals of the two tab widgets
        tab_widget1.currentChanged.connect(self.on_tab1_changed)
        tab_widget2.currentChanged.connect(self.on_tab2_changed)
        
        # Set the layout of the main widget
        main_widget.setLayout(main_layout)
    
    # Slot for when the first tab widget is changed
    def on_tab1_changed(self, index):
        # Switch to the corresponding tab in the second tab widget
        self.centralWidget().layout().itemAt(1).widget().setCurrentIndex(index)
    
    # Slot for when the second tab widget is changed
    def on_tab2_changed(self, index):
        # Switch to the corresponding tab in the first tab widget
        self.centralWidget().layout().itemAt(0).widget().setCurrentIndex(index)
        
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
    
    
    
    
    
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
        right_tab_widget.setTabToolTip(1, 'Modify magnetic field parameters of the model.') 
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