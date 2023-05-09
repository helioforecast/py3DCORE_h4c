from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QToolBar, QTabWidget, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window title and size
        self.setWindowTitle("PyQt5 Application with Toolbar and Canvas Figure")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        hbox = QHBoxLayout()
        central_widget.setLayout(hbox)
        
        # Create toolbar and tabs
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(16, 16))
        hbox.addWidget(toolbar)
        
        tab_widget = QTabWidget()
        hbox.addWidget(tab_widget)
        
        # Create two tabs with two sliders each
        for i in range(2):
            tab = QWidget()
            tab_widget.addTab(tab, f"Tab {i+1}")
            
            vbox = QVBoxLayout()
            tab.setLayout(vbox)
            
            for j in range(2):
                slider = QSlider(Qt.Horizontal)
                vbox.addWidget(slider)
        
        # Create little arrow to show/hide toolbar
        toolbar_toggle = toolbar.addAction(QIcon("arrow.png"), "Show/Hide Toolbar")
        toolbar_toggle.setCheckable(True)
        toolbar_toggle.setChecked(True)
        toolbar_toggle.triggered.connect(lambda checked: toolbar.setVisible(checked))
        
        # Create canvas figure in middle of window
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        hbox.addWidget(canvas)
        
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()