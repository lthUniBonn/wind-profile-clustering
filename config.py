# -*- coding: utf-8 -*-
"""Configuration file for wind resource analysis.

Attributes:
    start_year (int): Process the wind data starting from this year - in four-digit format.
    final_year (int): Process the wind data up to this year - in four-digit format.
    era5_data_dir (str): Directory path for reading era5 data files.
    model_level_file_name_format (str): Target name of the wind data files. Python's format() is used to fill in year
        and month at placeholders.
    surface_file_name_format (str): Target name of geopotential and surface pressure data files. Python's format() is
        used to fill in year and month at placeholders.

.FILL
.
.


.. _L137 model level definitions:
    https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

"""

# --------------------------- GENERAL
use_data_opts = ['DOWA', 'LIDAR', 'ERA5']
use_data = use_data_opts[0]


# See plots interactively abs_wind_pc_relative_diff_vs_velocity_ranges_20_m_lat_52.85_lon_3.43_DOWA_2010_2017.pdf- don't save plots directly as pdf to result_dir
plots_interactive = False
result_dir = "/cephfs/user/s6lathim/clustering_results/" + use_data + "/"
validation_type_opts = ['full_training_full_test', 'cut_training_full_test', 'cut_training_cut_test']
validation_type = validation_type_opts[1]

height_range_name_opts = ['DOWA_height_range', 'lin_height_range']
height_range_name = height_range_name_opts[0]

do_normalize_data = True
if do_normalize_data:
    result_dir = result_dir + validation_type + '/' + height_range_name + '/'
else:
    result_dir = result_dir + 'no_norm/' + validation_type + '/' + height_range_name + '/'

make_result_subdirs = True


start_year = 2010
final_year = 2017

# Single location processing
latitude = 0
longitude = 0
# TODO: get dowa indices by lat/lon - for now: DOWA loc indices used for both

# --------------------------- DOWA
# data contains years 2008 to 2017
DOWA_data_dir = "/cephfs/user/s6lathim/DOWA/"
# "/home/mark/WindData/DOWA/"  # '/media/mark/LaCie/DOWA/'

location = {'i_lat': 110, 'i_lon': 55}


# --------------------------- ERA5
# General settings.
era5_data_dir = '/cephfs/user/s6lathim/ERA5Data-redownload/'
model_level_file_name_format = "{:d}_europe_{:d}_130_131_132_133_135.nc"  # 'ml_{:d}_{:02d}.netcdf'
surface_file_name_format = "{:d}_europe_{:d}_152.nc"  # 'sfc_{:d}_{:02d}.netcdf'
era5_grid_size = 1.  # 0.25
# Processing settings
read_model_level_up_to = 112
height_range = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160., 180.,
                200., 220., 250., 300., 500., 600.]
# Test linearized height range (ERA5 only)
#height_range = [70.,  100., 140., 170., 200., 240., 270., 300., 340., 370.,
#                400., 440., 470., 500., 540., 570., 600.]