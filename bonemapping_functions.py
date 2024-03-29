from pathlib import Path
from ansys.mapdl import reader as pymapdl_reader
import pyvista as pv
from trimesh.transformations import rotation_matrix
import numpy as np
from loguru import logger
import numexpr as ne

ne.set_num_threads(8)
ne.use_vml = False
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import pydicom
import operator
from functools import reduce, partial
import yaml
from datetime import datetime
from itertools import chain, repeat
import SimpleITK as sitk
import trio


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


# def indices_merged_arr(arr):
def indices_merged_arr(arr):
    n = arr.ndim
    grid = np.ogrid[tuple(map(slice, arr.shape))]
    out = np.empty(arr.shape + (n + 1,), dtype=arr.dtype)
    for i in range(n):
        out[..., i] = grid[i]
    out[..., -1] = arr
    out.shape = (-1, n + 1)
    # out[:, :2] = out[:, (1, 0)]
    # return out[:, :-1], out[:, -1]
    return out[:, :-1], out[:, -1]


class DicomScan(pv.ImageData):
    """Container using pydicom to load and create a pyvista grid"""

    def __init__(self, directory, data_orientation=None):
        self.dicom_dir: Path = None
        self.data_patient_orientation = data_orientation
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
        self.read_xyzhu_from_dcm_sitk(directory, False)
        grid = pv.ImageData(dims=self.grid_shape, origin=self.origin, spacing=self.voxel_size)
        grid.point_data['HU'] = self.hu_data[:, -1]
        super(DicomScan, self).__init__(grid)

    def _sorted_dicom_files(self):
        '''Iterates through dicom files in directory and returns a sorted tuple of file paths by Z'''
        files = (pydicom.dcmread(fname, specific_tags=['ImagePositionPatient'], stop_before_pixels=True) for fname in
                 self.dicom_dir.iterdir())
        return (Path(file.filename) for file in sorted(files, key=lambda s: s.ImagePositionPatient[2]))

    def read_xyzhu_from_dcm_sitk(self, dicom_dir, skip_scouts=False):
        self.dicom_dir = Path(dicom_dir)

        reader = sitk.ImageSeriesReader()

        dicom_names = reader.GetGDCMSeriesFileNames(str(self.dicom_dir))
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        image = reader.Execute()
        for k in reader.GetMetaDataKeys(0):
            v = reader.GetMetaData(0, k)
            logger.debug(f'({k}) = = "{v}"')
        # dicom default: x is + to left, y is + posterior, z is + superior (https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037)
        if self.data_patient_orientation:
            image = sitk.DICOMOrient(image, self.data_patient_orientation)
        arr = sitk.GetArrayFromImage(image)
        self.voxel_size = image.GetSpacing()
        self.grid_shape = image.GetSize()
        self.origin = image.GetOrigin()
        grid_to_xyz = np.zeros((4, 4))
        grid_to_xyz[:-1, :-1] = np.reshape(image.GetDirection(), (3, -1)) * np.transpose(image.GetSpacing())
        grid_to_xyz[-1, :-1] = self.origin

        kij, hu = indices_merged_arr(arr)
        ijk1 = np.ones((kij.shape[0], 4))
        ijk1[:, :-1] = kij[:, [1, 2, 0]]

        xyz1 = ijk1 @ grid_to_xyz
        xyz1[:, -1] = hu
        self.hu_data = xyz1

    def read_xyzhu_from_dcm_original(self, dicom_dir, skip_scouts=False):
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

            logger.debug("skipped, no SliceLocation: {}".format(skipcount))

            # ensure remaining are in the correct order
            slices = sorted(slices, key=lambda s: s.SliceLocation)
        else:
            slices = tuple(files)

        slice0 = slices[0]
        # create 3D array
        img_shape = [*(slice0.pixel_array.shape), len(slices)]
        self.grid_shape = img_shape
        self.hu_data = np.multiply(np.ones((reduce(operator.mul, img_shape), 4)), [0, 0, 0, 1])

        # fill 2D array with the image data from the files
        # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
        self.pixel_spacing = np.array(slice0.PixelSpacing, dtype=float)  # center-center distance of pixels
        slice_increment_set = {slices[i + 1].ImagePositionPatient[2] - s.ImagePositionPatient[2] for i, s in
                               enumerate(slices[:-1])}
        if len(slice_increment_set) > 1: logger.warning(
            f'There are {len(slice_increment_set)} different spacings between slices.')
        self.slice_increment = tuple(slice_increment_set)[0] if len(slice_increment_set) == 1 else \
            slices[1].ImagePositionPatient[2] - slice0.ImagePositionPatient[2]
        self.slice_thickness = float(slice0.SliceThickness)  # center-center distance of pixels
        self.row_cosine = np.array(slice0.ImageOrientationPatient[:3], dtype=float)
        self.col_cosine = np.array(slice0.ImageOrientationPatient[-3:], dtype=float)
        self.rescale_slope = float(slice0.RescaleSlope)
        self.rescale_int = float(slice0.RescaleIntercept)
        grid_to_xyz = np.zeros((4, 4))
        grid_to_xyz[-1, -1] = 1
        grid_to_xyz[:-1, 0] = self.row_cosine * self.pixel_spacing[0]
        grid_to_xyz[:-1, 1] = self.col_cosine * self.pixel_spacing[1]
        self.voxel_size = np.array([*self.pixel_spacing, self.slice_increment])
        idx_vector = np.array([0, 0, 0, 1]).reshape(-1, 1)

        for i, s in tqdm(enumerate(slices), desc='Pulling xyz and HU data', total=len(slices)):
            img2d = s.pixel_array
            grid_to_xyz[:-1, -1] = s.ImagePositionPatient
            if i == 0:
                self.origin = grid_to_xyz[:-1, -1]
            df = self.hu_data[i * img2d.size:(i + 1) * img2d.size]
            df[:, :2], slice_hu_arr = indices_merged_arr(ne.evaluate("img * slope + intercept",
                                                                     local_dict={'img': img2d,
                                                                                 'slope': self.rescale_slope,
                                                                                 'intercept': self.rescale_int}))  # (x_idx, y_idx), HU
            xyz_one = df @ grid_to_xyz.T
            xyz_one[:, -1] = slice_hu_arr
            self.hu_data[i * img2d.size:(i + 1) * img2d.size] = xyz_one

        # hu_df = vaex.from_arrays(x=hu_data[:, 0], y=hu_data[:, 1], z=hu_data[:, 2], HU=hu_data[:, 3])
        # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032

    def read_xyzhu_from_dcm_hybrid(self, dicom_dir, skip_scouts=False):  # slower than original
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

            logger.debug("skipped, no SliceLocation: {}".format(skipcount))

            # ensure remaining are in the correct order
            slices = sorted(slices, key=lambda s: s.SliceLocation)
        else:
            slices = tuple(files)

        slice0 = slices[0]
        # create 3D array
        img_shape = [*(slice0.pixel_array.shape), len(slices)]
        self.grid_shape = img_shape

        # fill 2D array with the image data from the files
        # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
        self.pixel_spacing = np.array(slice0.PixelSpacing, dtype=float)  # center-center distance of pixels
        slice_increment_set = {slices[i + 1].ImagePositionPatient[2] - s.ImagePositionPatient[2] for i, s in
                               enumerate(slices[:-1])}
        if len(slice_increment_set) > 1: logger.warning(
            f'There are {len(slice_increment_set)} different spacings between slices.')
        self.slice_increment = tuple(slice_increment_set)[0] if len(slice_increment_set) == 1 else \
            slices[1].ImagePositionPatient[2] - slice0.ImagePositionPatient[2]
        self.slice_thickness = float(slice0.SliceThickness)  # center-center distance of pixels
        self.row_cosine = np.array(slice0.ImageOrientationPatient[:3], dtype=float)
        self.col_cosine = np.array(slice0.ImageOrientationPatient[-3:], dtype=float)
        self.rescale_slope = float(slice0.RescaleSlope)
        self.rescale_int = float(slice0.RescaleIntercept)
        self.origin = slice0.ImagePositionPatient

        grid_to_xyz = np.zeros((4, 4))
        grid_to_xyz[-1, -1] = 1
        grid_to_xyz[:-1, 0] = self.row_cosine * self.pixel_spacing[0]
        grid_to_xyz[:-1, 1] = self.col_cosine * self.pixel_spacing[1]
        grid_to_xyz[:-1, 2] = np.cross(self.row_cosine, self.col_cosine) * self.slice_increment
        grid_to_xyz[-1, :-1] = self.origin

        self.voxel_size = np.array([*self.pixel_spacing, self.slice_increment])

        all_arr = np.concatenate([s.pixel_array[np.newaxis, ...] for s in slices])
        kij, hu = indices_merged_arr(ne.evaluate("img * slope + intercept",
                                                 local_dict={'img': all_arr, 'slope': self.rescale_slope,
                                                             'intercept': self.rescale_int}))
        ijk1 = np.ones((kij.shape[0], 4))
        ijk1[:, :-1] = kij[:, [1, 2, 0]]

        xyz1 = ijk1 @ grid_to_xyz
        xyz1[:, -1] = hu
        self.hu_data = xyz1


class FeaMesh(pv.UnstructuredGrid):

    def __init__(self, grid: pv.UnstructuredGrid, dicom_data: pv.ImageData, params: dict):
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
        ct_interp = RegularGridInterpolator((x, y, z), shaped_hu, bounds_error=True)
        if np.all(self.celltypes == pv.CellType.QUADRATIC_TETRA):
            pts_per_cell = 10
        elif np.all(self.celltypes == pv.CellType.TETRA):
            pts_per_cell = 4
        # establish natural coordinates
        perfect_natural_coord, shape_fx_values = self._get_natural_tet_coordinates(step, pts_per_cell)
        # py_abq_nodes = [0, 1, 2, 3, 01, 12, 20, 04, 31, 32] matches vtk/pyvista
        # pv._vtk.VTK_QUADRATIC_TETRA
        elem_pts_arr = self.points[self.cells.reshape(-1, pts_per_cell + 1)][:, 1:][:, np.newaxis, ...]
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
        perfect_natural_coord, shape_fx_values = self._get_natural_tet_coordinates(step)
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
        ct_interp = RegularGridInterpolator((x, y, z), shaped_modulus, bounds_error=True)
        if np.all(self.celltypes == pv.CellType.QUADRATIC_TETRA):
            pts_per_cell = 10
        elif np.all(self.celltypes == pv.CellType.TETRA):
            pts_per_cell = 4
        # establish natural coordinates
        perfect_natural_coord, shape_fx_values = self._get_natural_tet_coordinates(step, pts_per_cell)
        # py_abq_nodes = [0, 1, 2, 3, 01, 12, 20, 04, 31, 32] matches vtk/pyvista
        # pv._vtk.VTK_QUADRATIC_TETRA
        # todo adapt to multiple input element shapes
        elem_pts_arr = self.points[self.cells.reshape(-1, pts_per_cell + 1)][:, 1:][:, np.newaxis, ...]
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
        calculated_modulus = ne.evaluate("sum(interp_result/ n_naturals, axis=1)")
        if self.params['integration']['apply_elasticity_bounds']:
            calculated_modulus = np.clip(calculated_modulus, float(self.params['CT_Calibration']['min_modulus_value']),
                                         float(self.params['CT_Calibration']['max_modulus_value']))
        self.interpolated_moduli = calculated_modulus

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
        self.binned_density = np.clip(self.binned_density,
                                      self.params['CT_Calibration']['min_rho'],
                                      self.binned_density.max())

    def _refine_materials(self):
        """This function uses preset parameters to group materials in bins, based on modulus gap value. Without this
        all elements would have independent material properties """
        self._bin_modulus()
        self._backcalculate_density()
        self._check_for_merged_materials()

    def _check_for_merged_materials(self):
        # check min density merge
        min_density_merged_mask = self.binned_density == self.params['CT_Calibration']['min_rho']
        min_density_merged_indices = np.argwhere(min_density_merged_mask).ravel()
        if min_density_merged_indices.size > 1:
            # get merged element index array
            merged_element_indices = np.array(
                list(chain.from_iterable([self.material_mapping[idx] for idx in min_density_merged_indices])))
            # get rid of material rows that were merged into one
            cutoff_idx = min_density_merged_indices[0] + 1
            self.binned_moduli = self.binned_moduli[:cutoff_idx]
            self.binned_density = self.binned_density[:cutoff_idx]
            self.material_mapping = self.material_mapping[:cutoff_idx]
            self.material_mapping[-1] = merged_element_indices

    @staticmethod
    def _get_natural_tet_coordinates(step, nodes_per_element=10):
        """Method to create natural coordinates for an n_noded element"""
        natural_coord_lists = []
        # natural_coords = np.zeros((nodes_per_element, 4))
        shape_value_lists = []
        # shape_value = np.zeros((nodes_per_element, nodes_per_element, 1))
        count = 0
        # l, r, s, t are normalized coordinates, (L1, L2, L3, L4 in bme.hu reference)
        # going from 0.0 at a vertex to 1.0 at the opposite sie or face
        for t in np.arange(step / 2., 1, step):
            for s in np.arange(step / 2., 1 - t, step):
                for r in np.arange(step / 2., 1 - s - t, step):
                    l = 1 - r - s - t
                    # calculate shape functions
                    # https: // www.sciencedirect.com / topics / engineering / tetrahedron - element
                    # https: // www.mm.bme.hu / ~gyebro / files / ans_help_v182 / ans_thry / thy_shp8.html
                    if nodes_per_element == 10:
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
                    else:
                        w = np.array([[l],
                                      [r],
                                      [s],
                                      [t]])
                    # natural_coords[count] = [l, r, s, t]
                    # shape_values[count] = w
                    natural_coord_lists.append([l, r, s, t])
                    shape_value_lists.append(w)
                    count += 1
        natural_coords = np.array(natural_coord_lists)
        shape_values = np.array(shape_value_lists)
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
            cell = mesh.get_cell(el_id)
            element_nodes_data.append(
                f'EN,{sep}{el_id},{sep}{("," + sep).join(map(str, nd_ids[cell.point_ids[:8]]))}')
            if cell.n_points > 8:
                element_nodes_data.append(
                    f'EMORE,{sep}{("," + sep).join(map(str, nd_ids[cell.point_ids[8:16]]))}')
            if cell.n_points > 16:
                element_nodes_data.append(
                    f'EMORE,{sep}{("," + sep).join(map(str, nd_ids[cell.point_ids[16:]]))}')
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
    original_data = False
    if original_data:
        tetmesh_file = Path(
            r'D:\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / '0408_S5.cdb'
        dicom_dir = Path(
            r'D:\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / '0408_s5'
        params_yaml_file = Path('verif_params.yaml')
        output_tetmesh = Path(
            r'D:\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / (
                                 tetmesh_file.stem + '_MM.inp')
    else:
        p3_dir = Path(r'D:\OneDrive - DJO LLC\Documents\Matchpoint Drive Data\p3. Segmented CT Scans')
        tetmesh_file = p3_dir / 'Segmented Scans\DAU14\DAU14-UKU-QUG/uku_quq_vol.vtk'
        dicom_dir = p3_dir / 'Segmented Scans\DAU14\DAU14-UKU-QUG\ScalarVolume_11'
        params_yaml_file = p3_dir / 'Segmented Scans\DAU14\DAU14-UKU-QUG/verif_params.yaml'
        output_tetmesh = Path(
            r'D:\OneDrive - DJO LLC\active_projects\Bone Density Paper\Bonemat Dev Tests') / (
                                 tetmesh_file.stem + '_MM.inp')
    with open(params_yaml_file) as f:
        parameters = yaml.safe_load(f)

    # dicom_data = load_dicom_file(dicom_dir)
    # todo add the option of loading image from different sources (nrrds, h5, etc)
    dicom_data = DicomScan(dicom_dir, data_orientation='LPS')
    if tetmesh_file.suffix == '.cdb':
        tetmesh = load_cdb_archive(str(tetmesh_file)).grid
    else:
        tetmesh = pv.read(tetmesh_file)
        tetmesh = tetmesh.extract_cells_by_type(pv.CellType.TETRA)
    tf = np.array(
        '1.000000 0.000000 0.000000 141.066692 0.000000 1.000000 0.000000 7.536217 0.000000 0.000000 1.000000 -46.385058 0.000000 0.000000 0.000000 1.000000'.split(
            ' ')).astype(float).reshape((-1, 4))
    inverted_tf = np.linalg.inv(tf)
    untransformed_tetmesh = tetmesh.transform(inverted_tf, inplace=False)
    pv.global_theme.color_cycler = 'default'
    pl = pv.Plotter()
    pl.add_volume(dicom_data, cmap='bone', opacity=[0, 0, 0, 0.0, 0.3, 0.6, 1])
    pl.add_mesh(tetmesh)
    pl.add_mesh(untransformed_tetmesh)
    pl.show()

    # mesh = FeaMesh(untransformed_tetmesh, dicom_data, parameters)
    # write_ansys_inp_file(mesh, output_tetmesh)
