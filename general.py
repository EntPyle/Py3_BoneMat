from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress


# import vaex


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


def xyz_from_voxel_arr(voxel_arr, pixel_spacing, slice_thickness, origin, row_cosine=(1, 0, 0), col_cosine=(0, 1, 0)):
    # create transformation matrix for index coordinates to spatial coordinates
    row_cs_argmax = np.abs(row_cosine).argmax()
    col_cs_argmax = np.abs(col_cosine).argmax()
    row_cosine = row_cosine * -1 if row_cosine[row_cs_argmax] < 0 else row_cosine
    col_cosine = col_cosine * -1 if col_cosine[col_cs_argmax] < 0 else col_cosine
    argmax_sum = row_cs_argmax + col_cs_argmax
    if argmax_sum == 1:  # xy
        slice_idx = 2
        slice_cosine = np.cross(row_cosine, col_cosine) if row_cs_argmax == 0 else np.cross(col_cosine, row_cosine)
    elif argmax_sum == 2:  # zx
        slice_idx = 1
        slice_cosine = np.cross(row_cosine, col_cosine) if row_cs_argmax == 2 else np.cross(col_cosine, row_cosine)
    else:  # yz
        slice_idx = 0
        slice_cosine = np.cross(row_cosine, col_cosine) if row_cs_argmax == 1 else np.cross(col_cosine, row_cosine)
    scr_trans = np.zeros((3, 3))
    scr_trans[:, row_cs_argmax] = row_cosine * pixel_spacing[0]
    scr_trans[:, col_cs_argmax] = col_cosine * pixel_spacing[1]
    scr_trans[:, slice_idx] = slice_cosine * slice_thickness
    transform = np.eye(4)
    transform[:-1, :-1] = scr_trans
    transform[:-1, -1] = origin
    rcsHU = indices_merged_arr(voxel_arr)  # slice_idx, column_idx, row_idx, HU
    # xyzHU = np.column_stack((origin + np.transpose(scr_trans @ scrHU[:, :-1].T), scrHU[:, -1]))
    ones = np.ones(len(rcsHU))
    xyz1 = np.column_stack((rcsHU[:, :-1], ones))
    xyzHU = np.transpose(transform @ xyz1.T)
    xyzHU[:, -1] = rcsHU[:, -1]
    # size_check = np.array([idx_dist * step for idx_dist, step in
    #           zip(voxel_arr.shape, (pixel_spacing[0]*row_cosine[0], pixel_spacing[1]*col_cosine[1], slice_thickness*slice_cosine[-1]))])
    # if np.allclose(xyzHU[:,:-1].ptp(axis=0), size_check, atol=6) is False:
    #     raise ValueError(f'One of xyz range was not close to the bounding box check. Difference: {np.subtract(xyzHU[:,:-1].ptp(axis=0), size_check)}')

    # df = vaex.from_arrays(x=xyzHU[:, 0], y=xyzHU[:, 1], z=xyzHU[:, 2], GV=xyzHU[:, 3])
    df = pd.DataFrame(xyzHU, columns=['x', 'y', 'z', 'HU'])
    # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032

    return df


def get_mode(data, n_modes=1):
    return Counter(data).most_common(n_modes)


def transform_coordinates(coordinate_array: np.ndarray, transformation_matrix: np.ndarray):
    xyz1 = np.ones_like(coordinate_array)
    xyz1[:, :-1] = coordinate_array[:, :-1]
    return np.transpose(transformation_matrix @ xyz1.T)


def flatten(list2d):
    return list(chain(*list2d))


def bland_altman_plot(data1, data2, data1_name='A', data2_name='B', subgroups=None, plotly_template='none',
                      annotation_offset=0.05, plot_trendline='subgroups+all', show_mean_std_box=False, n_sd=1.96, *args,
                      **kwargs):
    # todo resume here making BA plot from tidy_density
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.nanmean([data1, data2], axis=0)
    diff = (data1 - data2) - data2  # Difference between data1 and data2
    md = np.nanmean(diff)  # Mean of the difference
    sd = np.nanstd(diff, axis=0)  # Standard deviation of the difference

    fig = go.Figure()

    if subgroups is None:
        fig.add_trace(go.Scatter(x=mean, y=diff, mode='markers', **kwargs))
    else:
        for group_name in np.unique(subgroups):
            group_mask = np.where(np.array(subgroups) == group_name)
            fig.add_trace(
                go.Scatter(x=mean[group_mask], y=diff[group_mask], mode='markers', name=str(group_name),
                           legendgroup=str(group_name), **kwargs))
    if 'all' in plot_trendline.split('+'):
        slope, intercept, r_value, p_value, std_err = linregress(mean, diff)
        trendline_x = np.linspace(mean.min(), mean.max(), 10)
        fig.add_trace(go.Scatter(x=trendline_x, y=slope * trendline_x + intercept,
                                 name='All Trendline',
                                 mode='lines',
                                 line=dict(
                                     width=4,
                                     dash='dot')))
    if 'subgroups' in plot_trendline.split('+'):
        for group_name in np.unique(subgroups):
            group_mask = np.where(np.array(subgroups) == group_name)
            slope, intercept, r_value, p_value, std_err = linregress(mean[group_mask], diff[group_mask])
            trendline_x = np.linspace(mean[group_mask].min(), mean[group_mask].max(), 10)
            fig.add_trace(go.Scatter(x=trendline_x, y=slope * trendline_x + intercept,
                                     name=f'{group_name} Trendline',
                                     mode='lines',
                                     legendgroup='Sub Trendlines',
                                     line=dict(
                                         width=4,
                                         dash='dot')))

    if show_mean_std_box:
        fig.add_shape(
            # Line Horizontal
            type="line",
            xref="paper",
            x0=0,
            y0=md,
            x1=1,
            y1=md,
            line=dict(
                # color="Black",
                width=6,
                dash="dashdot",
            ),
            name=f'Mean {round(md, 2)}',
        )
        fig.add_shape(
            # borderless Rectangle
            type="rect",
            xref="paper",
            x0=0,
            y0=md - n_sd * sd,
            x1=1,
            y1=md + n_sd * sd,
            line=dict(
                color="SeaGreen",
                width=2,
            ),
            fillcolor="LightSkyBlue",
            opacity=0.4,
            name=f'±{n_sd} Standard Deviations'
        )
        fig.update_layout(annotations=[dict(
            x=1,
            y=md,
            xref="paper",
            yref="y",
            text=f"Mean {round(md, 2)}",
            showarrow=True,
            arrowhead=7,
            ax=50,
            ay=0
        ),
                          dict(
                              x=1,
                              y=n_sd * sd + md + annotation_offset,
                              xref="paper",
                              yref="y",
                              text=f"+{n_sd} SD",
                              showarrow=False,
                              arrowhead=0,
                              ax=0,
                              ay=-20
                          ),
                          dict(
                              x=1,
                              y=md - n_sd * sd + annotation_offset,
                              xref="paper",
                              yref="y",
                              text=f"-{n_sd} SD",
                              showarrow=False,
                              arrowhead=0,
                              ax=0,
                              ay=20
                          ),
                          dict(
                              x=1,
                              y=md + n_sd * sd - annotation_offset,
                              xref="paper",
                              yref="y",
                              text=f"{round(md + n_sd * sd, 2)}",
                              showarrow=False,
                              arrowhead=0,
                              ax=0,
                              ay=20
                          ),
                          dict(
                              x=1,
                              y=md - n_sd * sd - annotation_offset,
                              xref="paper",
                              yref="y",
                              text=f"{round(md - n_sd * sd, 2)}",
                              showarrow=False,
                              arrowhead=0,
                              ax=0,
                              ay=20
                          )
        ])

    # Edit the layout
    fig.update_layout(title=f'Bland-Altman Plot for {data1_name} and {data2_name}',
                      xaxis_title=f'Average of {data1_name} and {data2_name}',
                      yaxis_title=f'{data1_name} Minus {data2_name}',
                      template=plotly_template,
                      )
    return fig
