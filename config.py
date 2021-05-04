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
# TODO update docstring

# --------------------------- GENERAL
# Don't save plots directly as pdf to result_dir
plots_interactive = False

n_clusters = 8 # default: 8
n_pcs = 5 # default: 5

location_type = 'mmc' #TODO multiple locations/data info change




# ----------------------------------------------------------------
# -------------------------------- DATA - config input/output
# ----------------------------------------------------------------

# --------------------------- TIME --------------------------------------
start_year = 2010
final_year = 2017

# --------------------------- LOCATION ----------------------------------
# Single location processing
#latitude = 0
#longitude = 0
# TODO: get dowa indices by lat/lon - for now: DOWA loc indices used for both
location = {'i_lat': 110, 'i_lon': 55}

# --------------------------- DATASET INPUT ------------------------------
# Choose dataset
use_data_opts = ['DOWA', 'LIDAR', 'ERA5']
use_data = use_data_opts[0]

# --------------------------- DOWA
# DOWA data contains years 2008 to 2017
DOWA_data_dir = "/cephfs/user/s6lathim/DOWA/"
# "/home/mark/WindData/DOWA/"  # '/media/mark/LaCie/DOWA/'

# --------------------------- ERA5
# General settings
era5_data_dir = '/cephfs/user/s6lathim/ERA5Data/' #'-redownload/'
model_level_file_name_format = "{:d}_europe_{:d}_130_131_132_133_135.nc"  # 'ml_{:d}_{:02d}.netcdf'
surface_file_name_format = "{:d}_europe_{:d}_152.nc"  # 'sfc_{:d}_{:02d}.netcdf'
era5_grid_size = 0.25 #1.  
# Processing settings
read_model_level_up_to = 112

# --------------------------- OUTPUT -------------------------------------
# --------------------------- RESULT DIR
result_dir = "/cephfs/user/s6lathim/clustering_results/" + use_data + "/"

# --------------------------- FILE SUFFIX
# Get actual lat/lon from chosen DOWA indices - change this in the future to the other way around in read_data?# TODO
#from read_data.dowa import lats_dowa_grid, lons_dowa_grid
lat = 52.85 # lats_dowa_grid[location['i_lat'], location['i_lon']]
lon = 3.43 # lons_dowa_grid[location['i_lat'], location['i_lon']]

data_info = '_lat_{:2.2f}_lon_{:2.2f}_{}_{}_{}'.format(lat, lon, use_data, start_year, final_year)
# round to grid spacing in ERA5 data # TODO 
latitude = round(lat/era5_grid_size)*era5_grid_size
longitude = round(lon/era5_grid_size)*era5_grid_size
if use_data == 'ERA5':
    
    data_info = '_lat_{:2.2f}_lon_{:2.2f}_{}_{}_{}_grid_{}'.format(lat, lon, use_data, start_year, final_year, era5_grid_size)


# --------------------------- CLUSTERING OUTPUT
cluster_config = '{}{}'.format(n_clusters, location_type)

file_name_profiles = result_dir + 'cluster_wind_profile_shapes_{}{}.csv'.format(cluster_config, data_info)
file_name_freq_distr = result_dir + 'freq_distribution_{}{}.pickle'.format(cluster_config, data_info)
cut_wind_speeds_file = result_dir + 'cut_in_out_{}{}.pickle'.format(cluster_config, data_info)
# Mark # cut_wind_speeds_file = '/home/mark/Projects/quasi-steady-model-sandbox/wind_resource/cut_in_out_8mmc.pickle'




# ----------------------------------------------------------------
# -------------------------------- VALIDATION - sample config 
# ----------------------------------------------------------------
# PCA/ Clustering sample settings
validation_type_opts = ['full_training_full_test', 'cut_training_full_test', 'cut_training_cut_test']
validation_type = validation_type_opts[1] # default: 1

# Height range settings
# DOWA height range
height_range = [10.,  20.,  40.,  60.,  80., 100., 120., 140., 150., 160., 180.,
                200., 220., 250., 300., 500., 600.]
# Test linearized height range (ERA5 only)
#height_range = [70.,  100., 140., 170., 200., 240., 270., 300., 340., 370.,
#                400., 440., 470., 500., 540., 570., 600.]

height_range_name_opts = ['DOWA_height_range'] #, 'lin_height_range']
height_range_name = height_range_name_opts[0] # default: 0

# Normalization settings preprocessing
do_normalize_data = True # default: True

# Validation output directories
if do_normalize_data:
    result_dir_validation = result_dir + validation_type + '/' + height_range_name + '/'
else:
    result_dir_validation = result_dir + 'no_norm/' + validation_type + '/' + height_range_name + '/'

make_result_subdirs = True