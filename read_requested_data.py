#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
from os.path import join as path_join

from config import use_data, start_year, final_year, year_final_month,\
                   DOWA_data_dir, locations, \
                   era5_data_dir, model_level_file_name_format, latitude_ds_file_name, era5_data_input_format,\
                   surface_file_name_format, read_model_level_up_to, height_range #TODO import only location or locations

from era5_ml_height_calc import compute_level_heights

import dask
# only as many threads as requested CPUs | only one to be requested, more threads don't seem to be used
dask.config.set(scheduler='synchronous')


def read_raw_data(start_year, final_year, year_final_month=12):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        tuple of Dataset, ndarray, ndarray, ndarray, and ndarray: Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid, model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    if era5_data_input_format == 'loc_box':
        # match locations to loc-boxes? faster? TODO
        ds = read_ds_loc_boxes(start_year, final_year, year_final_month=12, n_boxes=21)
    elif era5_data_input_format == 'single_loc':
        ds = read_ds_single_loc_files()
    elif era5_data_input_format == 'monthly':
        # Construct the list of input NetCDF files
        ml_files = []
        sfc_files = []
        for y in range(start_year, final_year+1):
            for m in range(1, year_final_month+1):
                ml_files.append(path_join(era5_data_dir, model_level_file_name_format.format(y, m)))
                sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
        # Load the data from the NetCDF files.
        ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values  # Model level numbers.
    hours = ds['time'].values

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive. Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level


def read_ds_loc_boxes(start_year, final_year, year_final_month=12, n_boxes=21):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        Dataset: Reading object of multiple wind data (netCDF) files

    """
    # Construct the list of input NetCDF files
    ml_files = []
    sfc_files = []
    ml_loc_box_file_name = 'loc-box/' + model_level_file_name_format + '000{:02d}.nc'
    for y in range(start_year, final_year+1):
        for m in range(1, year_final_month+1):
            for i_box in range(n_boxes):
                ml_files.append(path_join(era5_data_dir, ml_loc_box_file_name.format(y, m, i_box)))
            sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))
    # Load the data from the NetCDF files.
    ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=True)
    return ds


def read_ds_single_loc_files():
    """"Read ERA5 wind data from location wise files.

    Returns:
        Dataset: Reading object of multiple wind data (netCDF) files

    """
    # Construct the list of input NetCDF files
    from config import i_locations
    data_files = []
    #Add only relevant locations to the ds
    for i_lat, i_lon in i_locations:
            data_files.append(latitude_ds_file_name.format(i_lat=i_lat, i_lon=i_lon))
    # Load the data from the NetCDF files.
    ds = xr.open_mfdataset(data_files, decode_times=True)
    return ds

def get_wind_data_era5(heights_of_interest, locations=[(40,1)], start_year=2010, final_year=2010, max_level=112, era5_data_input_format='monthly'):
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year, year_final_month=year_final_month)
    i_highest_level = list(levels).index(max_level)
    
    # Convert lat/lon lists to indices
    lats, lons = (list(lats), list(lons))
    
    i_locs = [(lats.index(lat), lons.index(lon)) for lat,lon in locations]

    v_req_alt_east = np.zeros((len(hours)*len(i_locs), len(heights_of_interest))) #TODO will this be too large? 
    v_req_alt_north = np.zeros((len(hours)*len(i_locs), len(heights_of_interest)))
    
    #TODO possible in parallel? I/O cap anyways? not best way for connected areas? -- test 
    
    # ds define for each location box 
    
    for i, i_loc in enumerate(i_locs):
        i_lat, i_lon = i_loc
        # Extract wind data for single location
        v_levels_east = ds['u'][:, i_highest_level:, i_lat, i_lon].values
        v_levels_north = ds['v'][:, i_highest_level:, i_lat, i_lon].values
    
        t_levels = ds['t'][:, i_highest_level:, i_lat, i_lon].values #TODO test -- beter to call values later? or all together at beginning?
        q_levels = ds['q'][:, i_highest_level:, i_lat, i_lon].values
    
        try:
            surface_pressure = ds.variables['sp'][:, i_lat, i_lon].values
        except KeyError:
            surface_pressure = np.exp(ds.variables['lnsp'][:, i_lat, i_lon].values)
        
        # Calculate model level height
        level_heights, density_levels = compute_level_heights(levels,
                                                              surface_pressure,
                                                              t_levels,
                                                              q_levels)
        # Determine wind at altitudes of interest by means of interpolating the raw wind data.
        v_req_alt_east_loc = np.zeros((len(hours), len(heights_of_interest)))  # Interpolation results array.
        v_req_alt_north_loc = np.zeros((len(hours), len(heights_of_interest)))
        

        for i_hr in range(len(hours)):
            if not np.all(level_heights[i_hr, 0] > heights_of_interest):
                raise ValueError("Requested height ({:.2f} m) is higher than height of highest model level."
                                 .format(level_heights[i_hr, 0]))
            v_req_alt_east_loc[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                v_levels_east[i_hr, ::-1])
            v_req_alt_north_loc[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                 v_levels_north[i_hr, ::-1])
        v_req_alt_east[len(hours)*i:len(hours)*(i+1), :] = v_req_alt_east_loc
        v_req_alt_north[len(hours)*i:len(hours)*(i+1), :] = v_req_alt_north_loc

    wind_data = { #TODO This could get too large for a large number of locations - better use an xarray structure here? 
        'wind_speed_east': v_req_alt_east,
        'wind_speed_north': v_req_alt_north,
        'n_samples': len(hours)*len(i_locs),
        'n_samples_per_loc': len(hours),
        'datetime': ds['time'].values,
        'altitude': heights_of_interest,
        'years': (start_year, final_year),
        'locations':locations
    }
    ds.close()  # Close the input NetCDF file.

    return wind_data


def get_wind_data():
    if use_data == 'DOWA':
        import os
        #HDF5 library has been updated (1.10.1) (netcdf uses HDF5 under the hood)
        #file system does not support the file locking that the HDF5 library uses.
        #In order to read your hdf5 or netcdf files, you need set this environment variable :
        os.environ["HDF5_USE_FILE_LOCKING"]="FALSE" # check - is this needed? if yes - where set, needed for era5? FIX
        from read_data.dowa import read_data
        wind_data = read_data({'mult_coords':locations}, DOWA_data_dir) 

        # Use start_year to final_year data only
        hours = wind_data['datetime']
        start_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(start_year))
        end_date = np.datetime64('{}-01-01T00:00:00.000000000'.format(final_year+1))

        start_idx = list(hours).index(start_date)
        end_idx = list(hours).index(end_date)
        data_range = range(start_idx, end_idx + 1)

        for key in ['wind_speed_east', 'wind_speed_north', 'datetime']:
            wind_data[key] = wind_data[key][data_range]
        wind_data['n_samples'] = len(data_range)
        wind_data['years'] = (start_year, final_year)

        print(len(hours))
        print(len(wind_data['wind_speed_east']), wind_data['n_samples'])
    elif use_data == 'LIDAR':
        from read_data.fgw_lidar import read_data
        wind_data = read_data()

    elif use_data in ['ERA5', 'ERA5_1x1']:
        wind_data = get_wind_data_era5(height_range, locations=locations, start_year=start_year, final_year=final_year,
                                       max_level=read_model_level_up_to,  era5_data_input_format=era5_data_input_format)
    else:
        raise ValueError("Wrong data type specified: {} - no option to read data is executed".format(use_data))

    return wind_data

