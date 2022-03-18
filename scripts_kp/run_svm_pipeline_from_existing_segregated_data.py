from nmf_kp import nmf_kp
import numpy as np
from data_preformatter_kp import data_preformatter_kp
from svm_kp import svm_kp

save_data = 1

########################################################
## Load the saved segregated data file and initialize the svm_kp object with it.
saved_segregated_data_filename = "D:/msi_project_data/saved_outputs/segregated_data/data_preformatted_pipelined_segregated_data_trianing_percentage_80_random_select_1_nmf__ver_1.npy"
svm_kp_object = svm_kp(saved_segregated_data_filename=saved_segregated_data_filename)

########################################################
## Run the gridsearch svm code
c_range = np.linspace(1, 200, 5)  # c_range=np.logspace(-4, 5, 20)
gamma_range = np.linspace(0.01, 2, 5)  # gamma_range=np.logspace(-4, 5, 20)
parameter_grid_dict = {'C': c_range, 'gamma': gamma_range, 'kernel': ['rbf']}
grid_search_cv_count = 5
grid_search_filename_prefix = 'test_grid_svm_'
grid_search_results, best_param_dict = svm_kp_object.grid_search_cv_svm(parameter_grid_dict=parameter_grid_dict,
                                                                        cv_count=grid_search_cv_count,
                                                                        save_data=save_data,
                                                                        filename_prefix=grid_search_filename_prefix)
print(best_param_dict)

########################################################
## Run SVM only once using given parameters
svm_kernel = 'rbf'
svm_max_iter = -1
svm_tol = 1e-10
svm_c = 1.5  # inverse regularization
svm_gamma = 0.154  # Could be 'scale' or, 'auto', or can give a number
svm_probability_enable = False
svm_shrinking = True
one_time_svc_results_object = svm_kp_object.perform_svm_once(svm_kernel=svm_kernel, svm_max_iter=svm_max_iter,
                                                             svm_tol=svm_tol, svm_c=svm_c, svm_gamma=svm_gamma,
                                                             svm_probability_enable=svm_probability_enable,
                                                             svm_shrinking=svm_shrinking)


