

import libmsi
import numpy as np
import random as rd
import os
from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.preprocessing import minmax_scale
from itertools import combinations
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearnex import patch_sklearn
patch_sklearn()

rd.seed(0)

class svm_kp:

    def __init__(self, data_preformatter_object=None, saved_segregated_data_filename=None):

        """
        @brief Initializes the svm_kp class
            Usage example:
                            from svm_kp import svm_kp
                            saved_segregated_data_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/segregated_data/version_2_segregated_data_trianing_percentage_80_random_select_1.npy'

                            svm_kp_object = svm_kp(data_preformatter_object = data_preformatter_kp_object)

                            OR

                            svm_kp_object = svm_kp(saved_segregated_data_filename = saved_segregated_data_filename)

        @param saved_segregated_data_filename THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved segregated_data_dict instead of the segregated_data_dict inside the data_preformatter_kp_object
             that came in through the initialization of this class. However, either this variable, or the 'data_preformatter_kp_object' argument
             must be provided.

        @param data_preformatter_object This is the object that has the segregated training and testing data
            including all other necessary background metadata. However, either this variable, or the 'saved_segregated_data_filename' argument
             must be provided.
        """

        if (saved_segregated_data_filename is not None) and (saved_segregated_data_filename != ''):
            self.segregated_data_dict_recovered = np.load(saved_segregated_data_filename, allow_pickle=True)[()]
            self.data_preformatter_object = None
            self.saved_segregated_data_filename = saved_segregated_data_filename

        elif (data_preformatter_object is not None):
            self.data_preformatter_object = data_preformatter_object
            self.segregated_data_dict_recovered = self.data_preformatter_object.segregated_data_dict
            self.saved_segregated_data_filename = saved_segregated_data_filename

    def perform_svm_once(self, svm_kernel='rbf', svm_max_iter=-1, svm_tol=1e-4, svm_c=1, svm_gamma=0.2, svm_probability_enable=False, svm_shrinking=True):

        """
        @brief Perform svm once using a given set of parameters.
            Example usage:
                            svm_kernel='rbf'
                            svm_max_iter=-1
                            svm_tol=1e-10
                            svm_c= 1.5 # inverse regularization
                            svm_gamma=0.154 # Could be 'scale' or, 'auto', or can give a number
                            svm_probability_enable=False
                            svm_shrinking=True
                            one_time_svc_results_object = svm_kp_object.perform_svm_once(svm_kernel=svm_kernel, svm_max_iter=svm_max_iter, svm_tol=svm_tol, svm_c=svm_c, svm_gamma=svm_gamma, svm_probability_enable=svm_probability_enable, svm_shrinking=svm_shrinking)

        @param svm_kernel This can be 'rbf', 'linear', etc
        @param svm_c C parameter
        @param svm_max_iter Can set to -1 to keep iterating until the tolerance criteria is met I think. See sklearn documentation
        @param svm_tol Tolerance after which SVM stops I think. Read sklearn documentation
        @param svm_gamma The gamma hyperparameter for 'rbf' kernel svm.
        @param svm_shrinking See sklearn documentation
        @param svm_probability_enable See sklearn documentation
        @return: An svm results object
        """

        segregated_data_dict_recovered = self.segregated_data_dict_recovered

        x_train = segregated_data_dict_recovered['x_train']
        y_train = segregated_data_dict_recovered['y_train']
        x_test = segregated_data_dict_recovered['x_test']
        y_test = segregated_data_dict_recovered['y_test']

        my_svc = SVC(kernel=svm_kernel, max_iter=svm_max_iter, tol=svm_tol, C=svm_c, gamma=svm_gamma, probability=svm_probability_enable, shrinking=svm_shrinking)
        my_svc.fit(x_train, np.squeeze(y_train))

        ### Train set predictions

        train_predictions = my_svc.predict(x_train)
        print('Training Set accuracy: ', np.sum(train_predictions == np.squeeze(y_train)) / y_train.shape[0])

        train_class_0_predictions = train_predictions[np.squeeze(y_train == 0)]
        train_class_1_predictions = train_predictions[np.squeeze(y_train == 1)]

        train_class_0_prediction_accuracy = np.sum(train_class_0_predictions == np.squeeze(y_train[y_train == 0])) / train_class_0_predictions.shape[0]
        train_class_1_prediction_accuracy = np.sum(train_class_1_predictions == np.squeeze(y_train[y_train == 1])) / train_class_1_predictions.shape[0]

        print('training class 0 prediction accuracy: ', train_class_0_prediction_accuracy)
        print('training class 1 prediction accuracy: ', train_class_1_prediction_accuracy)

        ### Test set predictions
        predictions = my_svc.predict(x_test)
        print('Testing set accuracy: ', np.sum(predictions == np.squeeze(y_test)) / y_test.shape[0])

        class_0_predictions = predictions[np.squeeze(y_test == 0)]
        class_1_predictions = predictions[np.squeeze(y_test == 1)]

        class_0_prediction_accuracy = np.sum(class_0_predictions == np.squeeze(y_test[y_test == 0])) / class_0_predictions.shape[0]
        class_1_prediction_accuracy = np.sum(class_1_predictions == np.squeeze(y_test[y_test == 1])) / class_1_predictions.shape[0]

        print('class 0 prediction accuracy: ', class_0_prediction_accuracy)
        print('class 1 prediction accuracy: ', class_1_prediction_accuracy)

        self.one_time_svc_results = my_svc
        return my_svc

    def grid_search_cv_svm(self, parameter_grid_dict='', cv_count=5, save_data=0, filename_prefix=''):

        """
        @brief Perform svm multiple times to automatically decide the best hyperparameters.
            Usage example:

                            c_range=np.linspace(1, 20, 2)        # c_range=np.logspace(-4, 5, 20)
                            gamma_range=np.linspace(0.01, 2, 2)  # gamma_range=np.logspace(-4, 5, 20)
                            parameter_grid_dict = {'C': c_range,  'gamma': gamma_range, 'kernel': ['rbf']}
                            grid_search_cv_count=5
                            grid_search_filename_prefix=''
                            grid_search_results, best_param_dict=svm_kp_object.grid_search_cv_svm(parameter_grid_dict=parameter_grid_dict, cv_count=grid_search_cv_count, save_data=save_data, filename_prefix=grid_search_filename_prefix)


        @param parameter_grid_dict: The parameters oover which the gridsearch should be carried out.
        @param cv_count: The number of cross validation iterations to run.
        @param save_data: Save the results if set to 1. Do not save if set to 0.
        @param filename_prefix: Only required if 'save_data' argument is 1. Defines the prefix of the saved filename
        @return: Returns the grid search results in a pandas dataframe, and also the best parameters that gave the best accuracy
        """

        segregated_data_dict_recovered = self.segregated_data_dict_recovered

        x_train = segregated_data_dict_recovered['x_train']
        y_train = segregated_data_dict_recovered['y_train']
        training_perc = segregated_data_dict_recovered['training_percentage']
        reject_perc = segregated_data_dict_recovered['reject_percentage_used_in_image_patch']
        window = segregated_data_dict_recovered['window_size_used_in_image_patch']
        used_image_patch_filename = segregated_data_dict_recovered['used_image_patch_filename']
        used_3d_datagrid_filename = segregated_data_dict_recovered['used_3d_datagrid_filename_for_image_patch']
        used_dim_reduction_technique = segregated_data_dict_recovered['used_dim_reduction_technique']

        my_svm_object = SVC()

        my_grided_classifier = GridSearchCV(my_svm_object, parameter_grid_dict, cv=cv_count)

        my_grided_classifier.fit(x_train, np.squeeze(y_train))

        results = my_grided_classifier.cv_results_

        grid_search_dict = {'Used segregated_data_filename': self.saved_segregated_data_filename,
                            'used_x_train': x_train, 'used_y_train': y_train,
                            'data_segregation_training_set_percentage': training_perc,
                            'image_patch_rejection_percentage': reject_perc,
                            'image_patch_window_size': window,
                            'used_3d_datagrid_filename': used_3d_datagrid_filename,
                            'my_grid_search_object': my_grided_classifier,
                            'used_dim_reduction_technique': used_dim_reduction_technique,
                            'global_path_name': segregated_data_dict_recovered['global_path_name']}

        if save_data == 1:
            folder_name = segregated_data_dict_recovered['global_path_name']+'/saved_outputs/grid_search_svm/'
            file_name = filename_prefix + 'grid_search_svm_' + used_dim_reduction_technique + '_' + 'window_' + str(window) + '_reject_perc_' + str(reject_perc) + '_training_perc_' + str(training_perc) + '_ver_1.npy'

            for count in range(2, 20):
                dir_list = os.listdir(folder_name)
                if file_name in dir_list:
                    file_name = file_name[0:-5] + str(count) + '.npy'

                else:
                    #             print("File exists. New run number: "+ str(count-1))
                    break

            np.save(folder_name+file_name, grid_search_dict, 'dtype=object')


        results_in_dataframe = pd.DataFrame.from_dict(my_grided_classifier.cv_results_)
        self.grid_search_cv_svm_results = results_in_dataframe

        print("The best parameters are %s with a score of %0.2f"
              % (my_grided_classifier.best_params_, my_grided_classifier.best_score_))

        best_param_dict_out = {'best_paras': my_grided_classifier.best_params_,
                               'best_cv_score': my_grided_classifier.best_score_}

        self.grid_search_cv_svm_best_param_dict = best_param_dict_out
        return results_in_dataframe, best_param_dict_out

