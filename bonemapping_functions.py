from pathlib import Path
from ansys.mapdl import reader as pymapdl_reader
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from trimesh.transformations import rotation_matrix
import numpy as np
from loguru import logger
import numexpr as ne
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

#


def load_cdb_archive(cdb_filepath):
    # import cdb file
    return pymapdl_reader.Archive(cdb_filepath, read_parameters=True)


def load_dicom_file(dicom_directory):
    reader = pv.DICOMReader(dicom_directory)
    return reader.read()


def downsample_dicom_volume(large_dicom, resampled_grid_size=(50, 50, 100), do_threshold=True, threshold_value=500):
    '''Useful to create a downsampled dicom grid for faster viewing.'''
    if do_threshold:
        logger.debug(f'Thresholding dicom with value of {threshold_value}')
        # clip data using threshold to remove uninteresting data.
        clipped = large_dicom.clip_scalar(scalars="DICOMImage", value=threshold_value, invert=False)
        # create downsampled grid
        logger.debug('Downsampling dicom')
        subset = pv.create_grid(clipped, dimensions=resampled_grid_size)
    else:
        logger.debug('Downsampling dicom without thresholding')
        subset = pv.create_grid(large_dicom, dimensions=resampled_grid_size)
    return subset.sample(large_dicom)


def rotate_pyvista_grid(grid, angle=180, rotation_axis=(0, 0, 1)):
    rot_matrix = rotation_matrix(angle, rotation_axis, (0, 0, 0))
    return grid.transform(rot_matrix, inplace=False)


# show and prompt user if rotation needs to happen
# p = pv.Plotter()
# p.add_volume(resampled_subset)
# p.add_mesh(mesh.grid, color='white')
# # p.add_mesh(rotated_mesh, color='blue')
# p.show_axes()
# p.view_xy()
# # p.show_grid()
# p.show()


class FeaMesh(pv.UnstructuredGrid):

    def __init__(self, grid: pv.UnstructuredGrid, dicom_data: pv.UniformGrid):
        super(FeaMesh, self).__init__(grid)
        self.ct_data = dicom_data
        self.hu = dicom_data.get_array('DICOMImage')
        self.shaped_hu = self.hu.reshape(dicom_data.dimensions)
        self.ct_spacing = dicom_data.spacing
        self.ct_bounds = dicom_data.bounds

    def _assign_material_properties(self, n_integration_steps, rho_qct_fx, rho_ash_fx, modulus_fx):
        step = 1.0 / n_integration_steps
        # establish natural coordinates
        perfect_natural_coord, shape_fx_values = self._get_natural_coordinates(step)
        # py_abq_nodes = [0, 1, 2, 3, 01, 12, 20, 04, 31, 32] matches vtk/pyvista
        # pv._vtk.VTK_QUADRATIC_TETRA
        # todo calculate modulus/density from shaped_hu using passed equations/params
        shaped_modulus = self.shaped_hu * 17

        for elem_id in self.get_array('ansys_elem_num'):  # iterate thru element ids
            elem_pts = self.cell_points(elem_id)
            jacobian = np.ones((4, 4))
            nc4 = perfect_natural_coord.T*4
            n_naturals = len(perfect_natural_coord)
            # get jacobians at all naturals for this element
            nc_J = np.zeros((4, 4, n_naturals))

            nc_J[0] = nc4 - 1
            nc_J[1] = nc4[[1, 0, 1, 0]]
            nc_J[2] = nc4[[2, 2, 0, 1]]
            nc_J[3] = nc4[[3, 3, 3, 2]]
            nc_P = np.zeros((4, 4, 3))
            nc_P[0] = elem_pts[:4]
            nc_P[1] = elem_pts[[4, 4, 5, 7]]
            nc_P[2] = elem_pts[[6, 5, 6, 8]]
            nc_P[3] = elem_pts[[7, 8, 9, 9]]

            jacobians = np.ones((10, 4, 4))
            jacobians[:,1:, :] = np.einsum('jec,jen->nce', nc_P, nc_J)
            det_jacobians = np.linalg.det(jacobians)

            # estimate volume of element from jacobians
            volume = det_jacobians.sum()/6/n_naturals
            # find co-ordinate for each iteration using shape functions
            interpolation_coordinates = np.sum(elem_pts*shape_fx_values, axis=1)

            x, y, z = [np.arange(self.ct_bounds[2 * idx], self.ct_bounds[2 * idx + 1] + step, step) for idx, step in
                       enumerate(self.ct_spacing)]
            self.ct_interp = RegularGridInterpolator((x, y, z), shaped_modulus)
            # for each co-ordinate find CT co-ordinate
            interpolated_ct_data = self.ct_interp(interpolation_coordinates)
            # todo integrate function?
            # todo trilinear interpolation in dicom using RegularGridInterpolate or interpn from scipy.interpolate
            # UniformGrid.find_containing_cell()


    @staticmethod
    def _get_natural_coordinates(step):
        natural_coords = np.zeros((10, 4))
        shape_values = np.zeros((10, 10, 1))
        count = 0
        for t in np.arange(step / 2., 1, step):
            for s in np.arange(step / 2., 1 - t, step):
                for r in np.arange(step / 2., 1 - s - t, step):
                    l = 1 - r - s - t
                    # calculate shape functions
                    # https: // www.sciencedirect.com / topics / engineering / tetrahedron - element
                    w = np.array([[(2 * l - 1) * l],
                                  [(2 * r - 1) * r],
                                  [(2 * s - 1) * s],
                                  [(2 * t - 1) * t],
                                  [4 * l * r],
                                  [4 * r * s],
                                  [4 * l * s],
                                  [4 * l * t],
                                  [4 * r * t],
                                  [4 * s * t]])
                    natural_coords[count] = [l, r, s, t]
                    shape_values[count] = w
                    count += 1
        return natural_coords, shape_values

        # l, r, s, t, w could all be calculated once ahead of time.

    # todo assign_material_properties(self, dicom_mesh). Easier with numexpr/numba, I think

    # todo interpolate scalar in the tet mesh
    # todo build NN tree for CT data.
    # todo _refine/bin material properties


# todo define_material_equations from params

def write_ansys_inp_file():
    # spacing is 13 spaces
    # todo Header
    # todo Nodes (N,node_id,             x_coord,             y_coord,             z_coord

    # todo material
    # MP,EX,mat_id,             22842.01821243
    # MP,NUXY,mat_id,             0.30000000
    # MP,DENS,mat_id,             1.44491135

    # ET,2,187 # todo element type

    # TYPE, 2 $ MAT, mat_id $ REAL, 0  # element type ptr? material id.
    # EN,             72201,             13409,             652,             4740,             5219,             30600,             30596,             59516,             61769 # node ids
    # EMORE,             30597,              59514 # more node ids
    # CM, TYPE2 - REAL0 - MAT1, ELEM  # element block (part?) name
    # todo footer

    # ESEL, ALL

    # FINISH

    pass


# todo save to .inp file.
if __name__ == '__main__':
    folder = Path(r'C:\Users\Npyle1\OneDrive - DJO LLC\active_projects\Bone Density Paper')
    filename = str(folder / '0408_S5.cdb')
    dicom_dir = folder / 's5'
    file_path = folder / '0408_S5_wmm_FEMAP_ABQ.inp'

    dicom_data = load_dicom_file(dicom_dir)
    tetmesh = load_cdb_archive(filename)
    mesh = FeaMesh(tetmesh.grid, dicom_data)


