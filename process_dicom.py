import operator
from functools import reduce

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pydicom
# import vaex
from tqdm import tqdm
import general

pio.renderers.default = 'browser'


def read_xyzhu_from_dcm(dicom_dir, skip_scouts=True):
    # load the DICOM files
    files = []

    for fname in dicom_dir.iterdir():
        # print("loading: {}".format(fname))
        files.append(pydicom.dcmread(fname))

    print("file count: {}".format(len(files)))

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
        slices = files

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    hu_data = np.zeros((reduce(operator.mul, img_shape), 4))

    # fill 2D array with the image data from the files
    pixel_spacing = np.array(slices[0].PixelSpacing, dtype=np.float)  # center-center distance of pixels
    slice_thickness = np.array(slices[0].SliceThickness, dtype=np.float)  # center-center distance of pixels
    row_cosine = np.array(slices[0].ImageOrientationPatient[:3], dtype=np.float)
    col_cosine = np.array(slices[0].ImageOrientationPatient[-3:], dtype=np.float)
    rescale_slope = float(files[0].RescaleSlope)
    rescale_int = float(files[0].RescaleIntercept)
    grid_to_xyz = np.zeros((4, 4))
    grid_to_xyz[-1, -1] = 1
    grid_to_xyz[:-1, 0] = row_cosine * pixel_spacing[0]
    grid_to_xyz[:-1, 1] = col_cosine * pixel_spacing[1]

    idx_vector = np.array([0, 0, 0, 1]).reshape(-1, 1)
    zero_row = np.r_[[0] * img_shape[0] ** 2]
    one_row = np.r_[[1] * img_shape[0] ** 2]

    for i, s in tqdm(enumerate(slices), desc='Pulling xyz and HU data', total=len(slices)):
        img2d = s.pixel_array
        slice_origin = np.array(s.ImagePositionPatient, dtype=np.float)  # Top left corner
        grid_to_xyz[:-1, -1] = slice_origin
        slice_hu_data = general.indices_merged_arr(img2d * rescale_slope + rescale_int)  # x_idx, y_idx, HU
        xyz_one = grid_to_xyz @ np.vstack((slice_hu_data[:, :-1].T, zero_row, one_row))
        xyz_one[-1, :] = slice_hu_data[:, -1]
        # slice_hu_data[:, :2] = slice_hu_data[:, :2] * pixel_spacing + slice_origin[:-1]
        hu_data[i * img2d.size:(i + 1) * img2d.size] = xyz_one.T

    # hu_df = vaex.from_arrays(x=hu_data[:, 0], y=hu_data[:, 1], z=hu_data[:, 2], HU=hu_data[:, 3])
    # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032

    return hu_data, np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness]), np.array([*img2d.shape, len(slices)])


def plot_xyzhu(df):
    fig = go.Figure(data=go.Volume(
        x=df.x,
        y=df.y,
        z=df.z,
        value=df.HU,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=30,  # needs to be a large number for good volume rendering
    ))

    fig.update_layout(scene_aspectmode='data')
    fig.show(threaded=True)
    return fig

