from pathlib import Path
from ansys.mapdl import reader as pymapdl_reader
import pyvista as pv
from trimesh.transformations import rotation_matrix
import numpy as np
from loguru import logger
import numexpr as ne

ne.set_num_threads(8)
ne.use_vml = True
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import pydicom
import operator
from functools import reduce
import yaml
from datetime import datetime
from itertools import chain, repeat


def load_cdb_archive(cdb_filepath):
    # import cdb file
    return pymapdl_reader.Archive(cdb_filepath, read_parameters=True)


def downsample_dicom_volume(large_dicom, resampled_grid_size=(50, 50, 100), do_threshold=True, threshold_value=500):
    '''Useful to create a downsampled dicom grid for faster viewing.'''
    if do_threshold:
        logger.debug(f'Thresholding dicom with value of {threshold_value}')
        # clip data using threshold to remove uninteresting data.
        clipped = large_dicom.clip_scalar(scalars="HU", value=threshold_value, invert=False)
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


def indices_merged_arr(arr):
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n + 1,), dtype=arr.dtype)
    for i in range(n):
        out[..., i] = grid[i]
    out[..., -1] = arr
    out.shape = (-1, n + 1)
    # out[:, :2] = out[:, (1, 0)]
    return out


class DicomScan(pv.UniformGrid):
    """Container using pydicom to load and create a pyvista grid"""

    def __init__(self, directory):
        self.dicom_dir: Path = None
        self.voxel_size = None
        self.rescale_int = None
        self.rescale_slope = None
        self.row_cosine = None
        self.slice_thickness = None
        self.slice_increment = None
        self.pixel_spacing = None
        self.hu_data = None
        self.grid_shape = None
        self.col_cosine = None
        self.read_xyzhu_from_dcm(directory, False)
        grid = pv.UniformGrid(dims=self.grid_shape, origin=self.origin, spacing=self.voxel_size)
        grid.point_data['HU'] = self.hu_data[:, -1]
        super(DicomScan, self).__init__(grid)

    def _sorted_dicom_files(self):
        '''Iterates through dicom files in directory and returns a sorted tuple of file paths by Z'''
        files = (pydicom.dcmread(fname, specific_tags=['ImagePositionPatient'], stop_before_pixels=True) for fname in self.dicom_dir.iterdir())
        return (Path(file.filename) for file in sorted(files, key=lambda s: s.ImagePositionPatient[2]))

    def read_xyzhu_from_dcm(self, dicom_dir, skip_scouts=False):
        '''Reads dicom xyz, HU, and other parameters into class. Origin'''
        self.dicom_dir = Path(dicom_dir)
        files = (pydicom.dcmread(file) for file in self._sorted_dicom_files())

        if skip_scouts:
            # skip files with no SliceLocation (eg scout views)
            slices = []
            skipcount = 0
            for f in files:
                if hasattr(f, 'SliceLocation'):
                    slices.append(f)
                else:
                    skipcount += 1

            print("skipped, no SliceLocation: {}".format(skipcount))

            # ensure remaining are in the correct order
            slices = sorted(slices, key=lambda s: s.SliceLocation)
        else:
            slices = tuple(files)

        # create 3D array
        img_shape = list(slices[0].pixel_array.shape)
        img_shape.append(len(slices))
        self.grid_shape = img_shape
        self.hu_data = np.zeros((reduce(operator.mul, img_shape), 4))

        # fill 2D array with the image data from the files
        # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
        self.pixel_spacing = np.array(slices[0].PixelSpacing, dtype=float)  # center-center distance of pixels
        slice_increment_set = {slices[i + 1].ImagePositionPatient[2] - s.ImagePositionPatient[2] for i, s in
                               enumerate(slices[:-1])}
        # todo consider raising a warning unequally spaced data
        self.slice_increment = tuple(slice_increment_set)[0] if len(slice_increment_set) == 1 else \
            slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
        self.slice_thickness = np.array(slices[0].SliceThickness, dtype=float)  # center-center distance of pixels
        self.row_cosine = np.array(slices[0].ImageOrientationPatient[:3], dtype=float)
        self.col_cosine = np.array(slices[0].ImageOrientationPatient[-3:], dtype=float)
        self.rescale_slope = float(slices[0].RescaleSlope)
        self.rescale_int = float(slices[0].RescaleIntercept)
        grid_to_xyz = np.zeros((4, 4))
        grid_to_xyz[-1, -1] = 1
        grid_to_xyz[:-1, 0] = self.row_cosine * self.pixel_spacing[0]
        grid_to_xyz[:-1, 1] = self.col_cosine * self.pixel_spacing[1]
        self.voxel_size = np.array([*self.pixel_spacing, self.slice_increment])
        idx_vector = np.array([0, 0, 0, 1]).reshape(-1, 1)
        zero_col = np.c_[[0] * img_shape[0] ** 2]
        one_col = np.c_[[1] * img_shape[0] ** 2]

        for i, s in tqdm(enumerate(slices), desc='Pulling xyz and HU data', total=len(slices)):
            img2d = s.pixel_array
            slice_origin = np.array(s.ImagePositionPatient, dtype=float)  # Top left corner
            grid_to_xyz[:-1, -1] = slice_origin
            if i == 0:
                self.origin = slice_origin
            slice_hu_data = indices_merged_arr(ne.evaluate("img * slope + intercept",
                                                           local_dict={'img': img2d, 'slope': self.rescale_slope,
                                                                       'intercept': self.rescale_int}))  # x_idx, y_idx, HU
            xyz_one = np.concatenate([slice_hu_data[:, :-1], zero_col, one_col], axis=1) @ grid_to_xyz.T
            xyz_one[:, -1] = slice_hu_data[:, -1]
            # slice_hu_data[:, :2] = slice_hu_data[:, :2] * pixel_spacing + slice_origin[:-1]
            self.hu_data[i * img2d.size:(i + 1) * img2d.size] = xyz_one

        # hu_df = vaex.from_arrays(x=hu_data[:, 0], y=hu_data[:, 1], z=hu_data[:, 2], HU=hu_data[:, 3])
        # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
        self.voxel_size = np.array([*self.pixel_spacing, self.slice_increment])


class FeaMesh(pv.UnstructuredGrid):

    def __init__(self, grid: pv.UnstructuredGrid, dicom_data: pv.UniformGrid, params: dict):
        super(FeaMesh, self).__init__(grid)
        self.ct_data = dicom_data
        self.hu = dicom_data.get_array('HU')
        self.ct_spacing = dicom_data.spacing
        self.ct_bounds = dicom_data.bounds
        self.params = params
        if self.params['integration']['method'] == 'E':
            self._interpolate_modulus()
        elif self.params['integration']['method'] == 'HU':
            self._interpolate_hu()
        else:
            raise ValueError('Integration method has to be either E or HU')
        self._refine_materials()
        self.poisson = params['CT_Calibration']['poisson']

    def _interpolate_hu(self):
        """
        This function uses trilinear interpolation on shape function generated coordinates to assign a HU to
        each element.
        :return:
        """
        step = 1.0 / self.params['integration']['steps']
        # establish natural coordinates
        perfect_natural_coord, shape_fx_values = self._get_natural_coordinates(step)
        # py_abq_nodes = [0, 1, 2, 3, 01, 12, 20, 04, 31, 32] matches vtk/pyvista
        # pv._vtk.VTK_QUADRATIC_TETRA
        # todo adapt to multiple input element shapes
        # using numexpr to evaluate dynamically built string expressions on arrays.
        HU = self.hu
        # HU = np.array([0, 5, 10, 5, 10, 15, 10, 15, 20, 5, 10, 15, 10, 15, 20, 15, 20,
        #                25, 10, 15, 20, 15, 20, 25, 20, 25, 30])
        shaped_hu = HU.reshape(self.ct_data.dimensions, order='F')
        x, y, z = [np.arange(self.ct_bounds[2 * idx], self.ct_bounds[2 * idx + 1] + step, step) for idx, step in
                   enumerate(self.ct_spacing)]
        # shaped_modulus = modulus.reshape([3, 3, 3])
        #  test_coords = [-1, 0, 1]
        #  ct_interp = RegularGridInterpolator((test_coords, test_coords, test_coords), shaped_modulus)
        ct_interp = RegularGridInterpolator((x, y, z), shaped_hu)
        elem_pts_arr = self.points[self.cells.reshape(-1, 11)][:, 1:][:, np.newaxis, ...]
        # find co-ordinate for each iteration using shape functions
        # test_HU = np.array([0, 5, 10, 5, 10, 15, 10, 15, 20, 5, 10, 15, 10, 15, 20, 15, 20,
        #                                         25, 10, 15, 20, 15, 20, 25, 20, 25, 30])
        # test_moduli = np.array([0.3253933519291822, 0.46225376439520105, 0.6140291170289474, 0.46225376439520105, 0.6140291170289474, 0.7793257648849838, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.46225376439520105, 0.6140291170289474, 0.7793257648849838, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.7793257648849838, 0.9570775611694978, 1.1464347387199867, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.7793257648849838, 0.9570775611694978, 1.1464347387199867, 0.9570775611694978, 1.1464347387199867, 1.3466993724191563])
        # test_moduli = test_moduli.reshape([3, 3, 3])
        # interpn(([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]), test_moduli, interpolation_coordinates)
        # interpolation_coordinates_arr = np.sum(elem_pts_arr * shape_fx_values, axis=2)
        interpolation_coordinates_arr = ne.evaluate("sum(elem_pts_arr * shape_fx_values, axis=2)")
        # for each co-ordinate, interpolate ct value
        interp_result = ct_interp(interpolation_coordinates_arr)
        n_naturals = perfect_natural_coord.shape[0]
        interpolated_hu = ne.evaluate("sum(interp_result/ n_naturals, axis=1)")
        # calculate modulus from hu
        rho_qct_fx_HU_str = ' + '.join(
            [f'{coef}*(HU**{idx})' for idx, coef in enumerate(self.params['CT_Calibration']['ct_coefs'])])
        rho = ne.evaluate(rho_qct_fx_HU_str, local_dict={'HU': interpolated_hu})
        if self.params['CT_Calibration']['apply_ash_correction']:
            rho_ash_fx_RQCT_str = ' + '.join([f'{coef}*(rho**{idx})' for idx, coef in
                                              enumerate(self.params['CT_Calibration']['ash_correction_coefs'])])
            rho = ne.evaluate(rho_ash_fx_RQCT_str)
        modulus_coefs = self.params['CT_Calibration']['modulus_coefs']
        modulus_fx_RHO_str = f'{modulus_coefs[0]}+{modulus_coefs[1]}*rho**{self.params["CT_Calibration"]["modulus_exponent"]}'
        min_density = self.params['CT_Calibration']['min_rho']
        rho = ne.evaluate(
            "where(rho<=min_density, min_density, rho)")  # eliminate any negative densities for full dicom array
        calculated_modulus = ne.evaluate(modulus_fx_RHO_str)
        if self.params['integration']['apply_elasticity_bounds']:
            calculated_modulus = np.clip(calculated_modulus, float(self.params['CT_Calibration']['min_modulus_value']),
                                         float(self.params['CT_Calibration']['max_modulus_value']))
        self.interpolated_moduli = calculated_modulus

    def _interpolate_modulus(self):
        """
        This function uses trilinear interpolation on shape function generated coordinates to assign a modulus to
        each element.
        :return:
        """
        step = 1.0 / self.params['integration']['steps']
        # establish natural coordinates
        perfect_natural_coord, shape_fx_values = self._get_natural_coordinates(step)
        # py_abq_nodes = [0, 1, 2, 3, 01, 12, 20, 04, 31, 32] matches vtk/pyvista
        # pv._vtk.VTK_QUADRATIC_TETRA
        # todo adapt to multiple input element shapes
        # using numexpr to evaluate dynamically built string expressions on arrays.
        HU = self.hu
        # HU = np.array([0, 5, 10, 5, 10, 15, 10, 15, 20, 5, 10, 15, 10, 15, 20, 15, 20,
        #                25, 10, 15, 20, 15, 20, 25, 20, 25, 30])
        rho_qct_fx_HU_str = ' + '.join(
            [f'{coef}*(HU**{idx})' for idx, coef in enumerate(self.params['CT_Calibration']['ct_coefs'])])
        rho = ne.evaluate(rho_qct_fx_HU_str)
        if self.params['CT_Calibration']['apply_ash_correction']:
            rho_ash_fx_RQCT_str = ' + '.join([f'{coef}*(rho**{idx})' for idx, coef in
                                              enumerate(self.params['CT_Calibration']['ash_correction_coefs'])])
            rho = ne.evaluate(rho_ash_fx_RQCT_str)
        modulus_coefs = self.params['CT_Calibration']['modulus_coefs']
        modulus_fx_RHO_str = f'{modulus_coefs[0]}+{modulus_coefs[1]}*rho**{self.params["CT_Calibration"]["modulus_exponent"]}'
        min_density = self.params['CT_Calibration']['min_rho']
        rho = ne.evaluate(
            "where(rho<=min_density, min_density, rho)")  # eliminate any negative densities for full dicom array
        modulus = ne.evaluate(modulus_fx_RHO_str)
        if self.params['integration']['apply_elasticity_bounds']:
            modulus = np.clip(modulus, float(self.params['CT_Calibration']['min_modulus_value']),
                              float(self.params['CT_Calibration']['max_modulus_value']))
        x, y, z = [np.arange(self.ct_bounds[2 * idx], self.ct_bounds[2 * idx + 1] + step, step) for idx, step in
                   enumerate(self.ct_spacing)]
        shaped_modulus = modulus.reshape(self.ct_data.dimensions, order='F')
        # shaped_modulus = modulus.reshape([3, 3, 3])
        #  test_coords = [-1, 0, 1]
        #  ct_interp = RegularGridInterpolator((test_coords, test_coords, test_coords), shaped_modulus)
        ct_interp = RegularGridInterpolator((x, y, z), shaped_modulus)

        # todo I think I can get rid of this for loop
        elem_pts_arr = self.points[self.cells.reshape(-1, 11)][:, 1:][:, np.newaxis, ...]
        # find co-ordinate for each iteration using shape functions
        # test_HU = np.array([0, 5, 10, 5, 10, 15, 10, 15, 20, 5, 10, 15, 10, 15, 20, 15, 20,
        #                                         25, 10, 15, 20, 15, 20, 25, 20, 25, 30])
        # test_moduli = np.array([0.3253933519291822, 0.46225376439520105, 0.6140291170289474, 0.46225376439520105, 0.6140291170289474, 0.7793257648849838, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.46225376439520105, 0.6140291170289474, 0.7793257648849838, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.7793257648849838, 0.9570775611694978, 1.1464347387199867, 0.6140291170289474, 0.7793257648849838, 0.9570775611694978, 0.7793257648849838, 0.9570775611694978, 1.1464347387199867, 0.9570775611694978, 1.1464347387199867, 1.3466993724191563])
        # test_moduli = test_moduli.reshape([3, 3, 3])
        # interpn(([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]), test_moduli, interpolation_coordinates)
        # interpolation_coordinates_arr = np.sum(elem_pts_arr * shape_fx_values, axis=2)
        interpolation_coordinates_arr = ne.evaluate("sum(elem_pts_arr * shape_fx_values, axis=2)")
        # for each co-ordinate, interpolate moduli
        interp_result = ct_interp(interpolation_coordinates_arr)
        n_naturals = perfect_natural_coord.shape[0]
        self.interpolated_moduli = ne.evaluate("sum(interp_result/ n_naturals, axis=1)")

    def _bin_modulus(self):
        '''This functions creates and bins material properties based on passed parameters to reduce the number of
        materials created by the mapping process. '''
        min_E = float(self.params['CT_Calibration']['min_modulus_value'])
        max_E = float(self.params['CT_Calibration']['max_modulus_value'])
        max_E = max_E if max_E < self.interpolated_moduli.max() else self.interpolated_moduli.max()
        E_bin_size = self.params['CT_Calibration']['modulus_bin_size']
        E_bin_method = self.params['CT_Calibration']['modulus_bin_grouping_method']
        possible_bins = np.arange(max_E, min_E - E_bin_size, -E_bin_size)
        assigned_bins = possible_bins[np.digitize(self.interpolated_moduli, possible_bins)]
        used_bins = possible_bins[np.in1d(possible_bins, assigned_bins)]
        self.binned_moduli = np.zeros_like(used_bins)
        self.material_mapping = list()
        for idx, bin_edge in enumerate(used_bins):
            # find elements in this material bin
            element_indices = np.where(assigned_bins == bin_edge)[0]
            bin_moduli = self.interpolated_moduli[element_indices]
            self.material_mapping.append(element_indices)
            self.binned_moduli[idx] = bin_moduli.max() if E_bin_method == 'max' else bin_moduli.mean()

        self.n_materials = len(used_bins)

    def _backcalculate_density(self):
        '''This function calculates the density from the assigned modulus by reversing the passed relationship
        between modulus and density '''
        modulus_coefs = self.params['CT_Calibration']['modulus_coefs']
        rho_ash = ne.evaluate('((mod-coef0)/coef1)**(1/mod_exp)',
                              local_dict={'mod': self.binned_moduli,
                                          'coef0': modulus_coefs[0],
                                          'coef1': modulus_coefs[1],
                                          'mod_exp': self.params['CT_Calibration']['modulus_exponent']})
        if self.params['CT_Calibration']['output_rho_qct']:
            # reverse rho_ash_correction
            ash_coefs = self.params['CT_Calibration']['ash_correction_coefs']
            rho_qct_fx_RASH_str = f'(rho_ash - {ash_coefs[0]})/{ash_coefs[1]}'
            self.binned_density = ne.evaluate(rho_qct_fx_RASH_str)
        else:
            self.binned_density = rho_ash
        self.binned_density = np.clip(self.binned_density, self.params['CT_Calibration']['min_rho'],
                                      self.binned_density.max())

    def _refine_materials(self):
        """This function uses preset parameters to group materials in bins, based on modulus gap value. Without this
        all elements would have independent material properties """
        self._bin_modulus()
        self._backcalculate_density()

    @staticmethod
    def _get_natural_coordinates(step):
        """Method to create natural coordinates for a TET10 element"""
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

    def frequency_table(self):
        table = np.zeros((self.binned_density.size, 3))
        table[:, 0] = self.binned_density
        table[:, 1] = self.binned_moduli
        table[:, 2] = [elem_map.size for elem_map in self.material_mapping]
        return table


def write_ansys_inp_file(mesh: FeaMesh, file_path: Path):
    """This function writes the passed mesh to ansys .inp format at the given file_path location. Currently assumes a
    TET10 mesh """
    assert file_path.suffix == '.inp', 'File suffix has to be .inp'
    # spacing is 13 spaces
    sep = ' ' * 13
    # Header
    header_lines = (r'/TITLE,',
                    fr'/COM, Generated By Python {str(datetime.now())}',
                    r'/PREP7',
                    '')
    # Nodes (N,node_id,             x_coord,             y_coord,             z_coord
    nd_ids = mesh.get_array('ansys_node_num')
    node_str_gen = (f"N,{nd_id},{sep}{nd_coord[0]},{sep}{nd_coord[1]},{sep}{nd_coord[2]}" for
                    nd_id, nd_coord in zip(nd_ids, mesh.points.round(6)))
    moduli_str = (fr"MP,EX,{mat_idx + 1},{sep}{ex}" for mat_idx, ex in enumerate(mesh.binned_moduli.round(8)))
    poisson_str = (fr"MP,NUXY,{mat_idx + 1},{sep}{nuxy}" for mat_idx, nuxy in
                   enumerate(repeat(mesh.poisson, mesh.binned_moduli.shape[0])))
    dens_str = (fr"MP,DENS,{mat_idx + 1},{sep}{dens}" for mat_idx, dens in enumerate(mesh.binned_density.round(8)))
    skipline = repeat('', mesh.binned_moduli.shape[0])
    matr_str_gen = (entry for material_block in zip(moduli_str, poisson_str, dens_str, skipline) for entry in
                    material_block)
    # MP,EX,mat_id,             22842.01821243
    # MP,NUXY,mat_id,             0.30000000
    # MP,DENS,mat_id,             1.44491135

    # ET,2,187 # element type
    element_type_entry = ('', '', 'ET,2,187')
    # element node and material mapping
    elem_ids = mesh.get_array('vtkOriginalCellIds')
    element_nodes_data = list()
    for mat_idx, element_set in enumerate(mesh.material_mapping):
        element_nodes_data.append('')
        element_nodes_data.append(f'TYPE, 2 $ MAT, {mat_idx + 1} $ REAL, 0')
        for elem_idx in element_set:
            el_id = elem_ids[elem_idx]
            element_nodes_data.append(
                f'EN,{sep}{el_id},{sep}{("," + sep).join(map(str, nd_ids[mesh.cell_point_ids(el_id)[:-2]]))}')
            element_nodes_data.append(
                f'EMORE,{sep}{("," + sep).join(map(str, nd_ids[mesh.cell_point_ids(el_id)[-2:]]))}')
        element_nodes_data.append(f'CM, TYPE2-REAL0-MAT{mat_idx + 1}, ELEM')

    # TYPE, 2 $ MAT, mat_id $ REAL, 0  # element type ptr? material id.
    # EN,             72201,             13409,             652,             4740,             5219,             30600,             30596,             59516,             61769 # node ids
    # EMORE,             30597,              59514 # more node ids
    # CM, TYPE2-REAL0-MAT1, ELEM  # element block (part?) name
    footer_lines = ('',
                    '',
                    'ESEL, ALL',
                    '',
                    'FINISH')
    # ESEL, ALL

    # FINISH

    write_content = chain(header_lines, node_str_gen, [''], matr_str_gen, element_type_entry, element_nodes_data,
                          footer_lines)
    with open(file_path, 'w') as f:
        f.write('\n'.join(write_content))


if __name__ == '__main__':
    # load data and parameters in
    tetmesh_cdb_file = Path(r'C:\Users\Npyle1\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / '0408_S5.cdb'
    dicom_dir = Path(r'C:\Users\Npyle1\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / '0408_s5'
    # params_yaml_file = Path(r'C:\Users\Npyle1\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / 'params.yaml'
    params_yaml_file = Path('params.yaml')
    output_tetmesh = Path(r'C:\Users\Npyle1\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / (tetmesh_cdb_file.stem + '_MM.inp')
    with open(params_yaml_file) as f:
        parameters = yaml.safe_load(f)

    # dicom_data = load_dicom_file(dicom_dir)
    dicom_data = DicomScan(dicom_dir)
    tetmesh = load_cdb_archive(str(tetmesh_cdb_file))
    mesh = FeaMesh(tetmesh.grid, dicom_data, parameters)
    write_ansys_inp_file(mesh, output_tetmesh)
