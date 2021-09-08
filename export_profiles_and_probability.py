import pandas as pd
import pickle
import numpy as np

import sys
import getopt

from config_clustering import n_clusters, n_pcs, file_name_profiles, cut_wind_speeds_file, file_name_freq_distr,\
    file_name_cluster_labels, file_name_cluster_pipeline
from read_requested_data import get_wind_data

n_wind_speed_bins = 100

def export_wind_profile_shapes(heights, prl, prp, output_file, ref_height=100.):
    assert output_file[-4:] == ".csv"
    df = pd.DataFrame({
        'height [m]': heights,
    })
    scale_factors = []
    for i, (u, v) in enumerate(zip(prl, prp)):
        w = (u**2 + v**2)**.5

        # Get normalised wind speed at reference height via linear interpolation
        w_ref = np.interp(ref_height, heights, w)
        # Scaling factor such that the normalised absolute wind speed at the reference height is 1 
        sf = 1/w_ref
        dfi = pd.DataFrame({
            'u{} [-]'.format(i+1): u*sf,
            'v{} [-]'.format(i+1): v*sf,
            'scale factor{} [-]'.format(i+1): sf,
        })
        df = pd.concat((df, dfi), axis=1)

        scale_factors.append(sf)
    df.to_csv(output_file, index=False, sep=";")
    return scale_factors


def export_frequency_distribution(cut_wind_speeds_file, output_file, labels_full, normalisation_wind_speeds, n_samples,
                                  normalisation_wind_speed_scaling, n_wind_speed_bins=100, write_output=True):
    cut_wind_speeds = pd.read_csv(cut_wind_speeds_file)
    freq_2d = np.zeros((n_clusters, n_wind_speed_bins))
    v_bin_limits = np.zeros((n_clusters, n_wind_speed_bins+1))
    for i_c in range(n_clusters):
        v = np.linspace(cut_wind_speeds['vw_100m_cut_in'][i_c], cut_wind_speeds['vw_100m_cut_out'][i_c],
                        n_wind_speed_bins+1)
        v_bin_limits[i_c, :] = v
        sf = normalisation_wind_speed_scaling[i_c]  # Re-scaling to make the normalisation winds used in the clustering
        # procedure consistent with the wind property used for characterizing the cut-in and cut-out wind speeds, i.e.
        # the wind speed at 100 m height.
        for j, (v0, v1) in enumerate(zip(v[:-1], v[1:])):
            samples_in_bin = (labels_full == i_c) & (normalisation_wind_speeds/sf >= v0) & \
                             (normalisation_wind_speeds/sf < v1) # Denormalised assigned cluster wind speed at 100m, for each sample
            freq_2d[i_c, j] = np.sum(samples_in_bin) / n_samples * 100.

    distribution_data = {'frequency': freq_2d, 'wind_speed_bin_limits': v_bin_limits}

    if write_output:
        with open(output_file, 'wb') as f:
            pickle.dump(distribution_data, f, protocol=2)
        
    return freq_2d, v_bin_limits


def location_wise_frequency_distribution(locations, n_wind_speed_bins, labels, n_samples, n_samples_per_loc, scale_factors, normalisation_values):
    if len(locations) > 1:
        distribution_data = {'frequency': np.zeros((len(locations), n_clusters, n_wind_speed_bins)), 
                             'wind_speed_bin_limits': np.zeros((len(locations), n_clusters, n_wind_speed_bins+1)),
                             'locations': locations
                                  }
        for i, loc in enumerate(locations):
            
            distribution_data['frequency'][i, :, :], distribution_data['wind_speed_bin_limits'][i, :, :] = export_frequency_distribution(
                cut_wind_speeds_file, file_name_freq_distr, labels[n_samples_per_loc*i: n_samples_per_loc*(i+1)],
                normalisation_values[n_samples_per_loc*i: n_samples_per_loc*(i+1)], n_samples_per_loc,
                scale_factors, n_wind_speed_bins=n_wind_speed_bins, write_output=False)
        
        with open(file_name_freq_distr, 'wb') as f:
            pickle.dump(distribution_data, f, protocol=2)

    
    else:
        freq_2d, v_bin_limits = export_frequency_distribution(cut_wind_speeds_file, file_name_freq_distr, labels,
                                      normalisation_values, n_samples,
                                      scale_factors)
            
    
def interpret_input_args():
    make_profiles, make_freq_distr = (False, False)
    if len(sys.argv) > 1:  # User input was given
        help = """
        python export_profiles_and_probability.py                  : run clustering, save both profiles and frequency distributions
        python export_profiles_and_probability.py -p               : run clustering, save new profiles
        python export_profiles_and_probability.py -f               : export frequency distributions 
        python export_profiles_and_probability.py -h               : display this help
        """
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hpf", ["help", "profiles", "frequency"])
        except getopt.GetoptError:  # User input not given correctly, display help and end
            print(help)
            sys.exit()
        for opt, arg in opts:
            if opt in ("-h", "--help"):  # Help argument called, display help and end
                print(help)
                sys.exit()
            elif opt in ("-p", "--profiles"): 
                make_profiles = True
            elif opt in ("-f", "--frequency"):  
                make_freq_distr = True
    else:
        make_profiles = True
        make_freq_distr = True

    return make_profiles, make_freq_distr


if __name__ == '__main__':
    # Read program parameters 
    make_profiles, make_freq_distr = interpret_input_args()
    
    import time
    since = time.time()
    
    if not make_profiles and make_freq_distr:
        print('Exporting frequency distribution only')
        profiles_file = pd.read_csv(file_name_profiles, sep=";")
        scale_factors = []
        for i in range(n_clusters):
            scale_factors.append(profiles_file['scale factor{} [-]'.format(i+1)][0])
        with open(file_name_cluster_labels, 'rb') as f:labels_file = pickle.load(f)
        labels = labels_file['labels [-]']
        n_samples = len(labels)
        normalisation_values = labels_file['normalisation value [-]']
        locations = labels_file['locations']
        n_samples_per_loc = labels_file['n_samples_per_loc']
        
        location_wise_frequency_distribution(locations, n_wind_speed_bins, labels, n_samples, n_samples_per_loc, scale_factors, normalisation_values)
        time_elapsed = time.time() - since
        print('Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    elif make_profiles:  
        print('Perform full clustering algorithm')
        from wind_profile_clustering import cluster_normalized_wind_profiles_pca, predict_cluster
        from preprocess_data import preprocess_data
    
        data = get_wind_data()
        time_elapsed = time.time() - since
        print('Input read - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        processed_data = preprocess_data(data)
        
        time_elapsed = time.time() - since
        print('Data preprocessed - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
        res = cluster_normalized_wind_profiles_pca(processed_data['training_data'], n_clusters, n_pcs=n_pcs)
        prl, prp = res['clusters_feature']['parallel'], res['clusters_feature']['perpendicular']

        time_elapsed = time.time() - since
        print('Clustering trained - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        processed_data_full = preprocess_data(data, remove_low_wind_samples=False)
        time_elapsed = time.time() - since
        print('Preprocessed full data - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        labels, frequency_clusters = predict_cluster(processed_data_full['training_data'], n_clusters,
                                                     res['data_processing_pipeline'].predict, res['cluster_mapping'])
        time_elapsed = time.time() - since
        print('Predicted full data - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # write cluster labels to file
        cluster_info_dict = {
            'n clusters': n_clusters,
            'n samples': len(labels),
            'n pcs': n_pcs,
            'labels [-]': labels,
            'cluster_mapping': res['cluster_mapping'],
            'normalisation value [-]': processed_data_full['normalisation_value'], 
            'locations': processed_data_full['locations'],
            'n_samples_per_loc': processed_data_full['n_samples_per_loc']
            }
        pickle.dump(cluster_info_dict, open(file_name_cluster_labels, 'wb'))

        # write mapping to file
        pipeline = res['data_processing_pipeline']
        pickle.dump(pipeline, open(file_name_cluster_pipeline, 'wb'))
        # read as:
        #with open(file_name_cluster_pipeline, 'rb') as f:pipeline = pickle.load(f)
        #labels_load_unsorted = pipeline.predict(processed_data_full['training_data'])
        
        scale_factors = export_wind_profile_shapes(data['altitude'], prl, prp, file_name_profiles)
                
        if make_freq_distr:
            location_wise_frequency_distribution(processed_data_full['locations'], n_wind_speed_bins, labels, 
                                            processed_data_full['n_samples'], 
                                            processed_data_full['n_samples_per_loc'], scale_factors, 
                                            processed_data_full['normalisation_value'])
        time_elapsed = time.time() - since
        print('Output written END - Time lapsed: ', '\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
