from sklearn.decomposition import PCA
import numpy as np
import numpy.ma as ma

import matplotlib as mpl

from config import plots_interactive, result_dir

if not plots_interactive:
    mpl.use('Pdf')
import matplotlib.pyplot as plt

from read_requested_data import get_wind_data
from wind_profile_clustering import cluster_normalized_wind_profiles_pca


def get_weights_from_heights(heights):
    """Calculate weights for mean weighted by height difference covered (half to top and to bottom, starting at 0).

    Parameters
    ----------
    heights : list
        Height levels in the data.

    Returns
    -------
    weights : list
        Weights for the respective heights.

    """
    if heights[0] != 0:
        test_heights = np.insert(heights, 0, 0, axis=0)
    else:
        test_heights = heights

    height_diffs_half = np.diff(test_heights)/2
    weights = [height_diffs_half[idx] + height_diffs_half[idx+1] for idx in range(len(height_diffs_half)-1)]
    weights.append(height_diffs_half[-1]*2)
    if heights[0] == 0:
        weights.insert(0, height_diffs_half[0])
    return weights


def plot_diff_pdf(data, wind_orientation, diff_type, output_file_name='diff_pdf_height.pdf', title=''):
    """Plot pdf of differences.

    Parameters
    ----------
    data : list
        Sample data for one height/diff type/wind_orientation.
    wind_orientation : string
        Evaluated wind orientation (parallel, perpendicualar, absolute).
    diff_type : string
        Evaluated differences (relative, absolute).
    output_file_name : string, optional
        Path to save pdf. The default is 'diff_pdf_height.pdf'.
    title : string, optional
        Plot title. The default is ''.

    Returns
    -------
    None.

    """
    if diff_type == 'relative':
        plt.hist(data, bins=100, density=True, range=(-0.5, 0.5))  # TODO remove again?
    else:
        plt.hist(data, bins=100, density=True)
    plt.grid()
    if diff_type == 'absolute':
        plt.xlabel('{} diff for v {} in m/s'.format(diff_type, wind_orientation))
    else:
        plt.xlabel('{} diff for v {}'.format(diff_type, wind_orientation))
    plt.ylabel('frequency')
    txt_x_pos = plt.xlim()[1]*0.1
    plt.text(txt_x_pos, plt.ylim()[1]*0.9, 'mean: {:.2E} +- {:.2E}'.format(np.mean(data), np.std(data)), color='r')
    plt.title(title)
    if not plots_interactive:
        plt.savefig(output_file_name)
        # Clear plots after saving, otherwise plotted on top of each other
        plt.cla()
        plt.clf()


def plot_height_vs_diffs(heights, wind_orientation, diff_type, n_pcs, plot_info,
                         pc_mean, pc_std,
                         cluster_mean=[], cluster_std=[], n_clusters=0):
    """Plot differences for all heights. Optional: Include Clustering differences for comparison.

    Parameters
    ----------
    heights : list
        Height levels in the data.
    wind_orientation : string
        Evaluated wind orientation (parallel, perpendicualar, absolute).
    diff_type : string
        Evaluated differences (relative, absolute).
    n_pcs : int
        Number of principal components evaluated.
    plot_info : string
        Info on data/location to be included in file naming.
    pc_mean : array
        PC difference sample mean.
    pc_std : array
        PC difference sample standard deviation.
    cluster_mean : array, optional
        Cluster difference sample mean. The default is [].
    cluster_std : array, optional
        Cluster difference sample standard deviation. The default is [].
    n_clusters : int, optional
        Number of clusters chosen in the analysis. The default is 0 - meaning no clustering.

    Returns
    -------
    None.

    """
    y = heights
    pc = plt.errorbar(pc_mean, y, xerr=pc_std, fmt='+')
    txt_x_pos = plt.xlim()[1]*0.1
    plt.text(txt_x_pos, y[-1]-(y[-1]-y[-2])*0.3, 'mean: {:.2E} +- {:.2E}'.format(np.mean(pc_mean), np.mean(pc_std)), color='b')
    # Calculate weighted mean depending on height difference covered (half difference to top and to bottom, starting at 0)
    weights = get_weights_from_heights(heights)
    plt.text(txt_x_pos, y[-1]-(y[-1]-y[-2])*0.7, 'w-mean: {:.2E} +- {:.2E}'.format(
        np.average(pc_mean, weights=weights), np.average(pc_std, weights=weights)), color='b')

    if n_clusters > 0:
        cluster = plt.errorbar(cluster_mean, y, xerr=cluster_std, fmt='+', alpha=0.5)
        plt.text(txt_x_pos, y[-1]-(y[-1]-y[-2])*1.3, 'mean: {:.2E} +- {:.2E}'.format(np.mean(cluster_mean), np.mean(cluster_std)),
                 color='r')
        plt.text(txt_x_pos, y[-1]-(y[-1]-y[-2])*1.7, 'w-mean: {:.2E} +- {:.2E}'.format(
            np.average(cluster_mean, weights=weights), np.average(cluster_std, weights=weights)),
            color='r')
        plt.legend((pc, cluster), ('pc', 'cluster'), loc='center right')

    plt.grid()
    if diff_type == 'absolute':
        plt.xlabel('{} diff for v {} in m/s'.format(diff_type, wind_orientation))
    else:
        plt.xlabel('{} diff for v {}'.format(diff_type, wind_orientation))
    plt.ylabel('height in m')
    plt.title('{} difference {} wind data with {} pcs'.format(diff_type, wind_orientation, n_pcs))

    if not plots_interactive:
        if len(cluster_mean) != 0:
            plt.savefig(result_dir + '{}_wind_{}_cluster_{}_diff_vs_height_{}_pcs'.format(
                wind_orientation, n_clusters, diff_type, n_pcs) + plot_info + '.pdf')
        else:
            plt.savefig(result_dir + '{}_wind_{}_diff_vs_height_{}_pcs'.format(
                wind_orientation, diff_type, n_pcs) + plot_info + '.pdf')

        # Clear plots after saving, otherwise plotted on top of each other
        plt.cla()
        plt.clf()


def diff_original_vs_reco(original, reco, data_is_normalised=False, sample_norm=[]):
    """Assess differences between original and fit reconstruction.

    Parameters
    ----------
    original : array
        Original data before fit.
    reco : array
        Reconstructed wind data via fit.
    data_is_normalised : BOOL, optional
        Data needs to be denormalised for absolute values. The default is False.
    sample_norm : array, optional
        Norm-factors for each sample, used to denormalise data by multiplication. The default is [].

    Returns
    -------
    diffs : dict
        Dictionary of difference types containing the respective resulting difference for each sample.

    """
    # Absolute difference:
    absolute_difference_normalised = reco - original
    if data_is_normalised:
        absolute_difference = absolute_difference_normalised * sample_norm[:, np.newaxis]
    else:
        absolute_difference = absolute_difference_normalised

    # Relative difference:
    # mask div by 0 in relative difference (explicitly coded, ma.masked_values(data, 0) for some data (low values?) masks all)
    original_data_masked = ma.array(original)
    original_data_masked[original == 0] = ma.masked

    relative_difference = absolute_difference_normalised/original_data_masked

    # Average over all samples - mean and standard deviation:
    diffs = {
            'absolute': absolute_difference,
            'relative': relative_difference
            }

    return diffs


def get_diffs_reco(original, abs_wind_speed, reco, n_altitudes, sample_norm=[],
                   wind_type_eval_only=['abs']):
    """Evaluate precision of reconstructed data versus original data.

    Parameters
    ----------
    original : array
        Original data (parallel and perpendicular wind speed) before fit.
    abs_wind_speed : array
        Original data (absolute wind speed) before fit.
    reco : array
        Reconstructed wind data (parallel and perpendicular wind speed) via fit.
    n_altitudes : int
        Number of height levels in the data.
    sample_norm : array, optional
        Norm-factors for each sample, used to denormalise data by multiplication. The default is [].
    wind_type_eval_only : list, optional
        Only evaluate selected wind orientations (parallel, perpendicular, abs). The default is ['abs'].

    Returns
    -------
    diffs_sample_mean : dict
        Dictionary of difference types containing the respective resulting difference mean&standard deviation.
    diffs : dict
        Dictionary of difference types containing the respective resulting difference for each sample.


    """
    # Also compare absolute wind speeds:
    # Calc resulting reco absolute wind speed, denormalise, as original absolute wind speed is also not normalised
    reco_abs_wind_speed = sample_norm[:, np.newaxis] * (reco[:, :n_altitudes]**2 +
                                                        reco[:, n_altitudes:]**2)**.5

    # Average over all samples - mean and standard deviation:
    diffs = {'parallel': diff_original_vs_reco(original[:, :n_altitudes], reco[:, :n_altitudes], sample_norm=sample_norm,
                                               data_is_normalised=True),
             'perpendicular': diff_original_vs_reco(original[:, n_altitudes:], reco[:, n_altitudes:], sample_norm=sample_norm,
                                                    data_is_normalised=True),
             'abs': diff_original_vs_reco(abs_wind_speed, reco_abs_wind_speed, data_is_normalised=False)
             }

    diffs_sample_mean = {}

    for wind_orientation in diffs:
        if wind_orientation not in wind_type_eval_only:
            continue
        diffs_sample_mean[wind_orientation] = {}
        for diff_type, val in diffs[wind_orientation].items():
            diffs_sample_mean[wind_orientation][diff_type] = (np.mean(val, axis=0), np.std(val, axis=0))

    return diffs_sample_mean, diffs


def eval_all_pcs(n_features, eval_pcs, wind_data, data_info, n_clusters=0, eval_heights=[], wind_type_eval_only=['abs']):
    """Fully evaluate a clustering analysis for a range of principal components.

    Parameters
    ----------
    n_features : int
        Maximal number of principal components to be analysed.
    eval_pcs : list
        Number of principal components to be analysed in detail.
    wind_data : array
        Wind speed data for parallel/perpendicular winds for an altitude range.
    data_info : string
        Location/data source information to be included in output file names.
    n_clusters : int, optional
        Number of clusters to be formed in the analysis. The default is 0 - meaning no clustering.
    eval_heights : list, optional
        Heights to be evaluated in detail. The default is [].
    wind_type_eval_only : list, optional
        Only evaluate selected wind orientations (parallel, perpendicular, abs). The default is ['abs'].

    Returns
    -------
    n_pc_dependence : dict
        Dictionary of difference types containing the respective resulting difference mean&standard deviation
        for the range of pcs analysed.
    """
    heights = wind_data['altitude']
    n_altitudes = len(heights)

    for n in range(n_features):
        n = n+1

        # ---- Principal component analysis
        # Do pca and back transformation
        pca = PCA(n_components=n)
        training_data_pc = pca.fit_transform(wind_data['training_data'])
        print("Components reduced from {} to {}.".format(wind_data['training_data'].shape[1], pca.n_components_))

        training_data_back_pc = pca.inverse_transform(training_data_pc)

        pc_differences, pc_full_diffs = get_diffs_reco(wind_data['training_data'], wind_data['wind_speed'],
                                                       training_data_back_pc, n_altitudes,
                                                       sample_norm=wind_data['normalisation_value'],
                                                       wind_type_eval_only=wind_type_eval_only)

        # ---- Clustering
        if n_clusters > 0:
            # Run clustering with n_clusters and n_pcs
            res = cluster_normalized_wind_profiles_pca(wind_data['training_data'], n_clusters, n_pcs=n)
            # Normalised parallel and perpendicular cluster wind speeds for each of the n_clusters
            prl, prp = res['clusters_feature']['parallel'], res['clusters_feature']['perpendicular']
            labels = res['sample_labels']
            cluster_reco_data = np.zeros(wind_data['training_data'].shape)
            # Fill reco data array with assigned cluster wind profile
            for i_cluster in range(n_clusters):
                mask_cluster = labels == i_cluster
                cluster_reco_data[mask_cluster, :n_altitudes] = prl[i_cluster, :][np.newaxis, :]
                cluster_reco_data[mask_cluster, n_altitudes:] = prp[i_cluster, :][np.newaxis, :]
            # Find differences between original and clustering reconstructed data
            cluster_differences, cluster_full_diffs = get_diffs_reco(wind_data['training_data'], wind_data['wind_speed'],
                                                                     cluster_reco_data, n_altitudes,
                                                                     sample_norm=wind_data['normalisation_value'],
                                                                     wind_type_eval_only=wind_type_eval_only)

        # ---- Plotting - pdfs and height evaluation
        if n in eval_pcs:
            # Define heights to be plotted - no heights given, evaluate all
            if eval_heights == []:
                eval_heights = heights
            for wind_orientation in pc_differences:
                if wind_orientation not in wind_type_eval_only:
                    continue

                for diff_type, val in pc_differences[wind_orientation].items():
                    pc_val = pc_full_diffs[wind_orientation][diff_type]
                    pc_mean, pc_std = val
                    if n_clusters > 0:
                        cluster_val = cluster_full_diffs[wind_orientation][diff_type]
                        cluster_mean, cluster_std = cluster_differences[wind_orientation][diff_type]

                    # Plot PDFs: Distribution of differences for each height.
                    for height_idx, height in enumerate(heights):
                        if height not in eval_heights:
                            continue
                        if n_clusters > 0:
                            plot_diff_pdf(cluster_val[:, height_idx], wind_orientation, diff_type,
                                          output_file_name=(result_dir + '{}_wind_{}_cluster_{}_diff_pdf_height_{}_{}_pcs'.format(
                                              wind_orientation, n_clusters, diff_type, height, n) + data_info + '.pdf'),
                                          title='cluster {} difference {} wind data with {} pcs at {} m'.format(
                                              diff_type, wind_orientation, n, height))
                        else:
                            plot_diff_pdf(pc_val[:, height_idx], wind_orientation, diff_type,
                                          output_file_name=(result_dir + '{}_wind_{}_diff_pdf_height_{}_{}_pcs'.format(
                                              wind_orientation, diff_type, height, n) + data_info + '.pdf'),
                                          title='{} difference {} wind data with {} pcs at {} m'.format(
                                              diff_type, wind_orientation, n, height))
                    # Plot height vs differences
                    if n_clusters > 0:
                        plot_height_vs_diffs(heights, wind_orientation, diff_type, n, data_info,
                                             pc_mean, pc_std,
                                             cluster_mean=cluster_mean, cluster_std=cluster_std,
                                             n_clusters=n_clusters)
                    else:
                        plot_height_vs_diffs(heights, wind_orientation, diff_type, n, data_info,
                                             pc_mean, pc_std)

        # ---- Fill result dictionary with n pc analysis results
        if n == 1:
            n_pc_dependence = {}
            for wind_orientation in pc_differences:
                n_pc_dependence[wind_orientation] = {}
                for diff_type in pc_differences[wind_orientation]:
                    n_pc_dependence[wind_orientation][diff_type] = np.zeros((n_features, 2, len(wind_data['altitude'])))
        for wind_orientation in pc_differences:
            for diff_type, val in pc_differences[wind_orientation].items():
                if n_clusters > 0:
                    n_pc_dependence[wind_orientation][diff_type][n-1, :, :] = cluster_differences[wind_orientation][diff_type]
                else:
                    n_pc_dependence[wind_orientation][diff_type][n-1, :, :] = val

    return n_pc_dependence


def evaluate_pc_analysis(wind_data, data_info, eval_pcs=[5, 6, 7, 21], eval_heights=[100, 300, 500],
                         eval_clusters=[], eval_n_pc_up_to=-1):
    """Fully evaluate multiple clustering analyses for a range of principal components.

    Parameters
    ----------
    wind_data : array
        Wind speed data for parallel/perpendicular winds for an altitude range.
    data_info : string
        Location/data source information to be included in output file names.
    eval_pcs : list
        Number of principal components to be analysed in detail. The default is [5, 6, 7, 21].
    eval_heights : list, optional
        Heights to be evaluated in detail. The default is [100, 300, 500].
    eval_clusters : list, optional
        Numbers of clusters to be evaluated. The default is [].
    eval_n_pc_up_to : int, optional
        Maximal number of principal components to be evaluated. The default is -1 - meaning all pcs.

    Returns
    -------
    None.

    """
    n_features = wind_data['training_data'].shape[1]
    print('Total of {} features in the principal component analysis'.format(n_features))

    if eval_n_pc_up_to < 1:
        eval_n_pc_up_to = n_features

    # pc evaluation for #pc up to eval_n_pc_up_to
    print('pc only:')
    n_pc_dependence = eval_all_pcs(eval_n_pc_up_to, eval_pcs, wind_data, data_info, eval_heights=eval_heights, n_clusters=0)

    # cluster evaluation for #pc up to eval_n_pc_up_to
    if len(eval_clusters) != 0:
        n_cluster_dependence = {}
        for cluster_idx, n_clusters in enumerate(eval_clusters):
            print('evaluate {} cluster analysis'.format(n_clusters))
            n_cluster_dependence[n_clusters] = eval_all_pcs(eval_n_pc_up_to, eval_pcs, wind_data, data_info,
                                                            eval_heights=eval_heights, n_clusters=n_clusters)

    # Compare to representation by sample mean
    sample_mean = np.mean(wind_data['wind_speed'], axis=0)
    print(sample_mean)
    sample_mean_dict = {
        'abs': {
            'absolute': (np.mean(wind_data['wind_speed']-sample_mean, axis=0),
                         np.std(wind_data['wind_speed']-sample_mean, axis=0)),
            'relative': (np.mean((wind_data['wind_speed']-sample_mean)/wind_data['wind_speed'], axis=0),
                         np.std((wind_data['wind_speed']-sample_mean)/wind_data['wind_speed'], axis=0))
            }
        }

    # Plot dependence of mean differences for each height depending on the number of pcs
    for height_idx, height in enumerate(wind_data['altitude']):
        if height not in eval_heights:
            continue

        for wind_orientation in n_pc_dependence:
            for diff_type in n_pc_dependence[wind_orientation]:
                x = np.array(range(eval_n_pc_up_to-2)) + 3
                y_pc = n_pc_dependence[wind_orientation][diff_type][2:, 0, height_idx]
                dy_pc = n_pc_dependence[wind_orientation][diff_type][2:, 1, height_idx]

                plt.xlabel('# pcs')
                if diff_type == 'absolute':
                    plt.ylabel('{} diff for v {} in m/s'.format(diff_type, wind_orientation))
                else:
                    plt.ylabel('{} diff for v {}'.format(diff_type, wind_orientation))
                plt.title('{} diff of v {} at {} m'.format(diff_type, wind_orientation, height))

                if len(eval_clusters) == 0:
                    # Plot detailed number of pcs dependence for only pc analysis
                    plt.errorbar(x, y_pc, yerr=dy_pc, fmt='+', color='b')
                    if wind_orientation in sample_mean_dict:
                        plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.8, 'mean: {:.2E} +- {:.2E}'.format(
                            sample_mean_dict[wind_orientation][diff_type][0][height_idx],
                            sample_mean_dict[wind_orientation][diff_type][1][height_idx]))
                    plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.7, '#pc=1: {:.2E} +- {:.2E}'.format(
                        n_pc_dependence[wind_orientation][diff_type][0, 0, height_idx],
                        n_pc_dependence[wind_orientation][diff_type][0, 1, height_idx]))
                    plt.text(plt.xlim()[1]*0.55, plt.ylim()[1]*0.6, '#pc=2: {:.2E} +- {:.2E}'.format(
                        n_pc_dependence[wind_orientation][diff_type][1, 0, height_idx],
                        n_pc_dependence[wind_orientation][diff_type][1, 1, height_idx]))
                    plt.savefig(result_dir + '{}_wind_{}_diff_vs_number_of_pcs_{}_m'.format(
                                wind_orientation, diff_type, height) + data_info + '.pdf')
                    # Clear plots after saving, otherwise plotted on top of each other
                    plt.cla()
                    plt.clf()
                else:
                    # Plot number of pcs dependence comparing all analyses with number of clusters given in eval_clusters
                    plot_dict = {}
                    for n_clusters in n_cluster_dependence:
                        y = n_cluster_dependence[n_clusters][wind_orientation][diff_type][2:, 0, height_idx]
                        dy = n_cluster_dependence[n_clusters][wind_orientation][diff_type][2:, 1, height_idx]
                        plot_dict[n_clusters] = plt.errorbar(x, y, yerr=dy, fmt='+', alpha=0.5)
                    legend_list = [plot_item for key, plot_item in plot_dict.items()]
                    legend_names = ['{} clusters'.format(key) for key, plot_item in plot_dict.items()]
                    pc = plt.errorbar(x, y_pc, yerr=dy_pc, fmt='+', color='b')
                    legend_list.insert(0, pc)
                    legend_names.insert(0, 'pc only')
                    plt.legend(legend_list, legend_names)
                    plt.savefig(result_dir + '{}_wind_cluster_{}_diff_vs_number_of_pcs_{}_m'.format(
                                wind_orientation, diff_type, height) + data_info + '.pdf')
                    # Clear plots after saving, otherwise plotted on top of each other
                    plt.cla()
                    plt.clf()


if __name__ == '__main__':
    wind_data, data_info = get_wind_data()
    from preprocess_data import preprocess_data
    wind_data = preprocess_data(wind_data)

    # Evaluate performance of pcs and clusters
    evaluate_pc_analysis(wind_data, data_info, eval_pcs=[3, 5, 7, 21], eval_heights=[20, 100, 300, 500, 600],
                         eval_clusters=[8], eval_n_pc_up_to=7)

    if plots_interactive:
        plt.show()
