import pandas as pd
import pickle
import numpy as np

import sys
import getopt

from config import n_clusters, n_pcs, file_name_profiles, cut_wind_speeds_file, file_name_freq_distr
from read_requested_data import get_wind_data

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
                                  normalisation_wind_speed_scaling, n_wind_speed_bins=100):
    with open(cut_wind_speeds_file, 'rb') as f:
        cut_wind_speeds = pickle.load(f)

    freq_2d = np.zeros((n_clusters, n_wind_speed_bins))
    v_bin_limits = np.zeros((n_clusters, n_wind_speed_bins+1))
    for i_c in range(n_clusters):
        v = np.linspace(cut_wind_speeds[i_c+1]['v_cut_in_100m'], cut_wind_speeds[i_c+1]['v_cut_out_100m'],
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

    with open(output_file, 'wb') as f:
        pickle.dump(distribution_data, f, protocol=2)

def interpret_input_args():
    make_profiles, make_freq_distr = (False, False)
    if len(sys.argv) > 1:  # User input was given
        help = """
        python export_profiles_and_probability.py                  : run clustering, save both profiles and frequency distributions
        python export_profiles_and_probability.py -p               : run clustering, save new profiles
        python export_profiles_and_probability.py -f               : run clustering, save new frequency distributions 
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

    from wind_profile_clustering import cluster_normalized_wind_profiles_pca, predict_cluster
    from preprocess_data import preprocess_data

    data = get_wind_data()
    processed_data = preprocess_data(data)

    res = cluster_normalized_wind_profiles_pca(processed_data['training_data'], n_clusters, n_pcs=n_pcs)
    prl, prp = res['clusters_feature']['parallel'], res['clusters_feature']['perpendicular']

    processed_data_full = preprocess_data(data, remove_low_wind_samples=False)
    labels, frequency_clusters = predict_cluster(processed_data_full['training_data'], n_clusters,
                                                 res['data_processing_pipeline'].predict, res['cluster_mapping'])

    if make_profiles:
        scale_factors = export_wind_profile_shapes(data['altitude'], prl, prp, file_name_profiles)
    else:
        profiles_file = pd.read_csv(file_name_profiles, sep=";")
        scale_factors = []
        for i in range(n_clusters):
            scale_factors.append(profiles_file['scale factor{} [-]'.format(i+1)])
            
    if make_freq_distr:
        export_frequency_distribution(cut_wind_speeds_file, file_name_freq_distr, labels,
                                      processed_data_full['normalisation_value'], processed_data_full['n_samples'],
                                      scale_factors)
