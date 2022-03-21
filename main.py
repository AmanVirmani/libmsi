from msi_file_names import msi_filenames
from he_file_names import he_filenames
from dataloader_kp import dataloader_kp
from nmf_kp import nmf_kp
import numpy as np
from data_preformatter_kp import data_preformatter_kp
from svm_kp import svm_kp


if __name__ == "__main__":

    save_data = 0

    path_to_compact_msi_dataset = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data.npy"
    compact_msi_data = np.load(path_to_compact_msi_dataset, allow_pickle=True)[()]
    saved_nmf_filename = "D:/msi_project_data/saved_outputs/nmf_outputs/truncated_combined_dataset_[1, 1, 1, 1, 1, 1, 1, 1, 0]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
    nmf_kp_object = nmf_kp(compact_msi_data, saved_nmf_filename=saved_nmf_filename)
    data_preformatter_kp_object = data_preformatter_kp(nmf_kp_object)

    image_patch_reject_percentage = 25
    image_patch_window_size = 30
    image_patch_overlap = 0
    segregate_data_training_percentage = 80
    segregate_data_random_select = 1
    preformatting_pipeline_filename_prefix = 'data_preformatted_pipelined_'

    # data_preformatter_kp_object.data_preformatting_pipeline(image_patch_reject_percentage=image_patch_reject_percentage,
    #                                                         image_patch_window_size=image_patch_window_size,
    #                                                         image_patch_overlap=image_patch_overlap,
    #                                                         segregate_data_training_percentage=segregate_data_training_percentage,
    #                                                         segregate_data_random_select=segregate_data_random_select,
    #                                                         save_data=save_data,
    #                                                         preformatting_pipeline_filename_prefix=preformatting_pipeline_filename_prefix)

    saved_segregated_data_filename = "D:/msi_project_data/saved_outputs/segregated_data/data_preformatted_pipelined_segregated_data_trianing_percentage_80_random_select_1_nmf__ver_1.npy"
    svm_kp_object = svm_kp(saved_segregated_data_filename=saved_segregated_data_filename)

    c_range = np.linspace(1, 200, 50)  # c_range=np.logspace(-4, 5, 20)
    gamma_range = np.linspace(0.01, 2, 50)  # gamma_range=np.logspace(-4, 5, 20)
    parameter_grid_dict = {'C': c_range, 'gamma': gamma_range, 'kernel': ['rbf']}
    grid_search_cv_count = 5
    grid_search_filename_prefix = 'test_grid_svm_'
    grid_search_results, best_param_dict=svm_kp_object.grid_search_cv_svm(parameter_grid_dict=parameter_grid_dict, cv_count=grid_search_cv_count, save_data=save_data, filename_prefix=grid_search_filename_prefix)
