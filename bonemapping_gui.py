import sys
import numpy as np
import bonemapping_functions as fx
import pyvista as pv
from pathlib import Path

# Setting the Qt bindings for QtPy
import os

os.environ["QT_API"] = "pyside2"
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from pyvistaqt import QtInteractor, MainWindow


class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        main_layout = QtWidgets.QHBoxLayout()
        # create sidebar layout
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(250)

        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.setSpacing(10)

        # initialize filler variables
        self.cdb_archive = None
        self.dicom_data = None

        load_cdb_button = QtWidgets.QPushButton('Load TetMesh')
        load_cdb_button.clicked.connect(self.load_cdb_file)
        self.cdb_filename_label = QtWidgets.QLabel()
        self.cdb_filename_label.setMargin(0)
        self.cdb_filename_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        load_dicom_button = QtWidgets.QPushButton('Load Dicom')
        load_dicom_button.clicked.connect(self.load_dicom_file)
        self.dicom_directory_label = QtWidgets.QLabel()
        self.dicom_directory_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        view_meshes_button = QtWidgets.QPushButton('View Loaded Meshes')
        view_meshes_button.clicked.connect(self.visualize_meshes)

        # add controls to rotate
        rotate_dicom_button = QtWidgets.QPushButton('Rotate DICOM 180° Around Z')
        rotate_dicom_button.released.connect(self.rotate_dicom)

        [sidebar_layout.addWidget(widget) for widget in
         [load_cdb_button, self.cdb_filename_label, load_dicom_button, self.dicom_directory_label, view_meshes_button, rotate_dicom_button]]
        sidebar.setLayout(sidebar_layout)
        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        self.plotter.show_axes()
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.frame)
        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        # self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # # allow adding a sphere
        # meshMenu = mainMenu.addMenu('Mesh')
        # self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        # self.add_sphere_action.triggered.connect(self.add_sphere)
        # meshMenu.addAction(self.add_sphere_action)

        if show:
            self.show()

    def load_cdb_file(self):
        load_dialog = QtWidgets.QFileDialog()
        filename, _ = load_dialog.getOpenFileName(filter='*.cdb')
        if filename:
            self.cdb_filename_label.setText(Path(filename).stem)
            self.cdb_archive = fx.load_cdb_archive(str(filename))

    def load_dicom_file(self):
        load_dialog = QtWidgets.QFileDialog()
        folder = load_dialog.getExistingDirectory()
        if folder:
            self.dicom_directory_label.setText(Path(folder).stem)
            self.dicom_data = fx.DicomScan(Path(folder))

    def visualize_meshes(self):
        self.plotter.clear()
        if self.cdb_archive:
            self.plotter.add_mesh(self.cdb_archive.grid, color='white', label='Tet Mesh')
        if self.dicom_data:
            self.plotter.add_volume(fx.downsample_dicom_volume(self.dicom_data, threshold_value=1000),
                                    scalars='HU', name='Downsampled DICOM')
            # self.plotter.add_volume(self.dicom_data, scalars='HU', name='DICOM', cmap='bone')
        self.plotter.reset_camera()

    def rotate_dicom(self):
        self.dicom_data = fx.rotate_pyvista_grid(self.dicom_data)  # defaults are 180° around Z
        if len(self.plotter.children()) > 0:
            self.visualize_meshes()

    # todo map_dicom_to_tetmesh
    # todo save to .inp file.
    # todo allow for bone density to be calculated from HU after all of this. Tack on a function?


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())
