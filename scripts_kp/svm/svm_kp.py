import multiprocessing

import libmsi_kp
import numpy as np
import random as rd
import os
from sklearn.preprocessing import minmax_scale
from itertools import combinations
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import copy
from collections import Counter
import pickle

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearnex import patch_sklearn

patch_sklearn()


rd.seed(0)


class svm_kp:
    def __init__(
        self,
        dataloader_kp_object=None,
        data_preformatter_object=None,
        saved_segregated_data_filename=None,
    ):

        """
        @brief Initializes the svm_kp class
            Usage example:
                            from svm_kp import svm_kp
                            saved_segregated_data_filename='/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/segregated_data/version_2_segregated_data_trianing_percentage_80_random_select_1.npy'

                            svm_kp_object = svm_kp(data_preformatter_object = data_preformatter_kp_object)

                            OR

                            svm_kp_object = svm_kp(saved_segregated_data_filename = saved_segregated_data_filename)

                            OR
                            dataloader_kp_object = double_compact_msi_kp_object
                            svm_kp_object = svm_kp(dataloader_kp_object=dataloader_kp_object, saved_segregated_data_filename = saved_segregated_data_filename)



        @param saved_segregated_data_filename THis is an optional parameter, which, if provided as a path, will cause this piece of
            code to use that saved segregated_data_dict instead of the segregated_data_dict inside the data_preformatter_kp_object
             that came in through the initialization of this class. However, either this variable, or the 'data_preformatter_kp_object' argument
             must be provided.

        @param data_preformatter_object This is the object that has the segregated training and testing data
            including all other necessary background metadata. However, either this variable, or the 'saved_segregated_data_filename' argument
             must be provided.

        @param dataloader_kp_object: This is also an optional parameter. This can accept either a double_compact version, a compact version, or the full version of a saved dataloader_kp class object
        """

        if (saved_segregated_data_filename is not None) and (
            saved_segregated_data_filename != ""
        ):
            self.segregated_data_dict_recovered = np.load(
                saved_segregated_data_filename, allow_pickle=True
            )[()]
            self.data_preformatter_object = None
            self.saved_segregated_data_filename = saved_segregated_data_filename

        elif data_preformatter_object is not None:
            self.data_preformatter_object = data_preformatter_object
            self.segregated_data_dict_recovered = (
                self.data_preformatter_object.segregated_data_dict
            )
            self.saved_segregated_data_filename = saved_segregated_data_filename

        if (dataloader_kp_object is not None) and (dataloader_kp_object != ""):
            self.data_preformatter_object = data_preformatter_object
            self.saved_segregated_data_filename = saved_segregated_data_filename
            self.dataloader_kp_object = dataloader_kp_object
        else:
            self.data_preformatter_object = data_preformatter_object
            self.saved_segregated_data_filename = saved_segregated_data_filename
            self.dataloader_kp_object = dataloader_kp_object

    def perform_svm_once(
        self,
        svm_kernel="rbf",
        svm_max_iter=-1,
        svm_tol=1e-4,
        svm_c=1,
        svm_gamma=0.2,
        svm_probability_enable=False,
        svm_shrinking=True,
        custom_scoring_function=None,
    ):

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
                            custom_scoring_function = 'fbeta'
                            one_time_svc_results_object = svm_kp_object.perform_svm_once(svm_kernel=svm_kernel, svm_max_iter=svm_max_iter, svm_tol=svm_tol, svm_c=svm_c, svm_gamma=svm_gamma, svm_probability_enable=svm_probability_enable, svm_shrinking=svm_shrinking, custom_scoring_function=custom_scoring_function)

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

        x_train = segregated_data_dict_recovered["x_train"]
        y_train = segregated_data_dict_recovered["y_train"]
        x_test = segregated_data_dict_recovered["x_test"]
        y_test = segregated_data_dict_recovered["y_test"]

        my_svc = SVC(
            kernel=svm_kernel,
            max_iter=svm_max_iter,
            tol=svm_tol,
            C=svm_c,
            gamma=svm_gamma,
            probability=svm_probability_enable,
            shrinking=svm_shrinking,
        )
        my_svc.fit(x_train, np.squeeze(y_train))

        ### Train set predictions

        train_predictions = my_svc.predict(x_train)
        print(
            "Training Set accuracy: ",
            np.sum(train_predictions == np.squeeze(y_train)) / y_train.shape[0],
        )

        train_class_0_predictions = train_predictions[np.squeeze(y_train == 0)]
        train_class_1_predictions = train_predictions[np.squeeze(y_train == 1)]

        train_class_0_prediction_accuracy = (
            np.sum(train_class_0_predictions == np.squeeze(y_train[y_train == 0]))
            / train_class_0_predictions.shape[0]
        )
        train_class_1_prediction_accuracy = (
            np.sum(train_class_1_predictions == np.squeeze(y_train[y_train == 1]))
            / train_class_1_predictions.shape[0]
        )

        print(
            "training class 0 prediction accuracy: ", train_class_0_prediction_accuracy
        )
        print(
            "training class 1 prediction accuracy: ", train_class_1_prediction_accuracy
        )

        ### Test set predictions
        predictions = my_svc.predict(x_test)
        print(
            "Testing set accuracy: ",
            np.sum(predictions == np.squeeze(y_test)) / y_test.shape[0],
        )

        class_0_predictions = predictions[np.squeeze(y_test == 0)]
        class_1_predictions = predictions[np.squeeze(y_test == 1)]

        class_0_prediction_accuracy = (
            np.sum(class_0_predictions == np.squeeze(y_test[y_test == 0]))
            / class_0_predictions.shape[0]
        )
        class_1_prediction_accuracy = (
            np.sum(class_1_predictions == np.squeeze(y_test[y_test == 1]))
            / class_1_predictions.shape[0]
        )

        print("class 0 prediction accuracy: ", class_0_prediction_accuracy)
        print("class 1 prediction accuracy: ", class_1_prediction_accuracy)

        self.one_time_svc_results = my_svc
        return my_svc

    def grid_search_cv_svm(
        self,
        saved_grid_search_dict_filename="",
        parameter_grid_dict="",
        cv_count=5,
        save_data=0,
        filename_prefix="",
        custom_scoring_function=None,
        num_parallel_processes=1,
    ):

        """
        @brief Perform svm multiple times to automatically decide the best hyperparameters.
            Usage example:

                            c_range=np.linspace(1, 20, 2)        # c_range=np.logspace(-4, 5, 20)
                            gamma_range=np.linspace(0.01, 2, 2)  # gamma_range=np.logspace(-4, 5, 20)
                            parameter_grid_dict = {'C': c_range,  'gamma': gamma_range, 'kernel': ['rbf']}
                            grid_search_cv_count=5
                            grid_search_filename_prefix=''
                            custom_scoring_function='fbeta'
                            num_parallel_processes = 5
                            grid_search_results, best_param_dict, best_svm_estimator, best_scores_set, other_output_info = svm_kp_object.grid_search_cv_svm(parameter_grid_dict=parameter_grid_dict, cv_count=grid_search_cv_count, save_data=save_data, filename_prefix=grid_search_filename_prefix, custom_scoring_function=custom_scoring_function, num_parallel_processes=num_parallel_processes)

                            or:

                            saved_grid_search_dict_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/grid_search_svm/multi_trials_summary/trial_set_4/test_grid_svm_for_trial_num_38_grid_search_svm_nmf_score_func_fbeta_window_15_reject_perc_25_training_perc_80_ver_1.npy"
                            grid_search_results, best_param_dict, best_svm_estimator, best_scores_set, other_output_info = svm_kp_object.grid_search_cv_svm(saved_grid_search_dict_filename=saved_grid_search_dict_filename)


        @param saved_grid_search_dict_filename: If a path is given in this variable, do NOT calculate a grid search object anew. instead, simply load the saved grid search object's data from here.
        @param parameter_grid_dict: The parameters oover which the gridsearch should be carried out.
        @param cv_count: The number of cross validation iterations to run.
        @param save_data: Save the results if set to 1. Do not save if set to 0.
        @param filename_prefix: Only required if 'save_data' argument is 1. Defines the prefix of the saved filename
        @param custom_scoring_function: This argument can accept 'fbeta', 'custom_kp'. If nothing is given, or an empty string is passed, the standard scorer in svm class of sklearn will be used.
        @param num_parallel_processes: Set the number of processor cores you would like to parallely use
        @return: Returns the grid search results in a pandas dataframe, and also the best parameters that gave the best accuracy
        """

        if (saved_grid_search_dict_filename is not None) and (
            saved_grid_search_dict_filename != ""
        ):  ## Do NOT run the gridsearch analysis anew
            print("NOT running gridsearch anew. Loading a saved gridsearch result file")
            self.grid_search_dict = np.load(
                saved_grid_search_dict_filename, allow_pickle=True
            )[()]
            my_grided_classifier = self.grid_search_dict["my_grid_search_object"]
            custom_scorer_name = self.grid_search_dict["scorer_used"]
            self.segregated_data_dict_recovered = self.grid_search_dict[
                "segregated_data_dict_used"
            ]
            segregated_data_dict_recovered = self.segregated_data_dict_recovered
            x_train = self.segregated_data_dict_recovered["x_train"]
            y_train = self.segregated_data_dict_recovered["y_train"]
            x_test = self.segregated_data_dict_recovered["x_test"]
            y_test = self.segregated_data_dict_recovered["y_test"]

        else:  ## Run the gridsearch analysis anew
            segregated_data_dict_recovered = self.segregated_data_dict_recovered

            x_train = segregated_data_dict_recovered["x_train"]
            y_train = segregated_data_dict_recovered["y_train"]
            x_test = segregated_data_dict_recovered["x_test"]
            y_test = segregated_data_dict_recovered["y_test"]

            used_dim_reduction_technique = segregated_data_dict_recovered[
                "used_dim_reduction_technique"
            ]

            my_svm_object = SVC()

            if custom_scoring_function == "fbeta":
                custom_scorer_name = custom_scoring_function
                print(
                    "Using the ",
                    custom_scorer_name,
                    " score to calculate svm accuracies",
                )
                my_custom_scorer = make_scorer(fbeta_score, beta=1)
                my_grided_classifier = GridSearchCV(
                    my_svm_object,
                    parameter_grid_dict,
                    scoring=my_custom_scorer,
                    cv=cv_count,
                    n_jobs=num_parallel_processes,
                )
            elif custom_scoring_function == "kp_custom":
                custom_scorer_name = custom_scoring_function
                print(
                    "Using the ",
                    custom_scorer_name,
                    " score to calculate svm accuracies",
                )
                my_custom_scorer = make_scorer(self.custom_kp_score_func)
                my_grided_classifier = GridSearchCV(
                    my_svm_object,
                    parameter_grid_dict,
                    scoring=my_custom_scorer,
                    cv=cv_count,
                    n_jobs=num_parallel_processes,
                )
            else:
                custom_scorer_name = "standard_sklearn"
                print(
                    "Using the ",
                    custom_scorer_name,
                    " score to calculate svm accuracies",
                )
                my_grided_classifier = GridSearchCV(
                    my_svm_object,
                    parameter_grid_dict,
                    cv=cv_count,
                    n_jobs=num_parallel_processes,
                )

            my_grided_classifier.fit(x_train, np.squeeze(y_train))

            self.grid_search_dict = {
                "segregated_data_dict_used": segregated_data_dict_recovered,
                "my_grid_search_object": my_grided_classifier,  # Saving this along with a fully custom scorer may cause an overflow error. Try to use inbuilt custom scorers in sklearn.metrics library                                      'used_dim_reduction_technique': used_dim_reduction_technique,
                "global_path_name": segregated_data_dict_recovered["global_path_name"],
                "scorer_used": custom_scorer_name,
                "best_classifier_object_estimator": my_grided_classifier.best_estimator_,
            }

            if save_data == 1:
                folder_name = (
                    segregated_data_dict_recovered["global_path_name"]
                    + "/saved_outputs/grid_search_svm/"
                )
                file_name = (
                    filename_prefix
                    + "grid_search_svm_"
                    + used_dim_reduction_technique
                    + "_score_func_"
                    + custom_scorer_name
                    + "_unseen_test_data_"
                    + str(
                        segregated_data_dict_recovered[
                            "totally_unseen_test_data_enabled"
                        ]
                    )
                    + "_ver_1.npy"
                )

                for count in range(2, 20):
                    dir_list = os.listdir(folder_name)
                    if file_name in dir_list:
                        file_name = file_name[0:-5] + str(count) + ".npy"

                    else:
                        # print("File exists. New run number: "+ str(count-1))
                        break

                np.save(folder_name + file_name, self.grid_search_dict)

        #########################################
        ## Stuff for printing and returning

        results_in_dataframe = pd.DataFrame.from_dict(my_grided_classifier.cv_results_)
        self.grid_search_cv_svm_results = results_in_dataframe

        best_param_dict_out = {
            "best_paras": my_grided_classifier.best_params_,
            "best_cv_score": my_grided_classifier.best_score_,
            "scorer_used": custom_scorer_name,
        }

        self.grid_search_cv_svm_best_param_dict = best_param_dict_out
        self.best_svm_estimator = my_grided_classifier.best_estimator_

        print(
            "The best parameters are ",
            my_grided_classifier.best_params_,
            " with a score of ",
            my_grided_classifier.best_score_,
            " as measured from ",
            custom_scorer_name,
            " score function",
        )

        ########################################################################
        ## Calculate train and test accuracies off best estimator

        # For train data
        best_training_accuracy = my_grided_classifier.score(
            x_train, y_train
        )  ## Note: the gridded classifier object now actually resembles the best estimator by default

        # For test data
        best_testing_accuracy = my_grided_classifier.score(x_test, y_test)

        ########################################################################
        ## Calculate the confusion matrices for train and test data

        # For train data
        train_data_confusion_matrix = confusion_matrix(
            y_train, my_grided_classifier.predict(x_train)
        )

        # For test data
        test_data_confusion_matrix = confusion_matrix(
            y_test, my_grided_classifier.predict(x_test)
        )

        ########################################################################
        ## More stuff for printing and returning

        print(
            "\n###### \n Train data accuracy with best estimator ("
            + custom_scorer_name
            + "): "
            + str(best_training_accuracy)
            + " with confusion matrix: "
            + str(train_data_confusion_matrix)
        )
        print(
            "\n###### \n Test data accuracy with best estimator ("
            + custom_scorer_name
            + "): "
            + str(best_testing_accuracy)
            + " with confusion matrix: "
            + str(test_data_confusion_matrix)
        )

        self.best_scores_set = {
            "best_training_accuracy": best_training_accuracy,
            "best_testing_accuracy": best_testing_accuracy,
            "best_training_confusion_matrix": train_data_confusion_matrix,
            "best_testing_confusion_matrix": test_data_confusion_matrix,
        }

        ########################################################################
        ## Other output info will be arranged into a dictionary
        if segregated_data_dict_recovered["totally_unseen_test_data_enabled"] == 1:
            totally_unseen_test_data_datasets = segregated_data_dict_recovered[
                "totally_unseen_test_data_datasets"
            ]
        else:
            totally_unseen_test_data_datasets = "None"

        other_output_info = {
            "totally_unseen_test_data_datasets": totally_unseen_test_data_datasets
        }

        ########################################################################

        return (
            results_in_dataframe,
            best_param_dict_out,
            self.best_svm_estimator,
            self.best_scores_set,
            other_output_info,
        )

    def per_process_ordering_dim_reduced_component_combinations_for_svm_accuracy(
        self,
        this_process_combinations_list,
        saved_dim_reduced_filename,
        process_number,
        num_components_per_group,
        data_preformatting_parameter_dict,
        parameter_grid_dict_for_svm,
        my_shared_memory_container,
    ):

        from nmf_kp import nmf_kp
        from pca_kp import pca_kp
        from data_preformatter_kp import data_preformatter_kp

        original_dim_reduced_dict_recovered = np.load(
            saved_dim_reduced_filename, allow_pickle=True
        )[()]
        dim_reduced_dict_recovered = copy.deepcopy(original_dim_reduced_dict_recovered)
        used_dim_reduction_technique = original_dim_reduced_dict_recovered[
            "used_dim_reduction_technique"
        ]
        original_dim_reduced_outputs_recovered = original_dim_reduced_dict_recovered[
            "dim_reduced_outputs"
        ][0][0]
        original_num_dim_reduced_components = original_dim_reduced_outputs_recovered.shape[
            1
        ]

        combinations_list = this_process_combinations_list

        per_process_main_comparison_table = {
            "per_process_count_array": [],
            "per_process_component_set_removed": [],
            "per_process_residual_accuracy": [],
            "process_id": process_number,
        }
        for count, chosen_nmf_set in enumerate(combinations_list):
            print(
                "Now processing combination number ",
                count,
                " out of ",
                len(combinations_list),
                " combinations of process number " + str(process_number),
            )
            print("NMF components that will be removed: ", chosen_nmf_set)
            with open(
                self.dataloader_kp_object.global_path_name
                + "kp_code/order_"
                + used_dim_reduction_technique
                + "_to_svm_accuracy_sub_process_"
                + str(process_number)
                + "_result.txt",
                "a",
            ) as file_1:  # Screen output update
                file_1.write(
                    "Now processing combination number "
                    + str(count)
                    + " out of "
                    + str(len(combinations_list))
                    + " combinations of process number "
                    + str(process_number)
                    + ".\n"
                    + "NMF components that will be removed: "
                    + str(chosen_nmf_set)
                    + ".\n"
                )

            dim_reduced_outputs_modified = np.delete(
                original_dim_reduced_outputs_recovered, chosen_nmf_set, axis=1
            )
            dim_reduced_dict_recovered["dim_reduced_outputs"][0][
                0
            ] = dim_reduced_outputs_modified
            dim_reduced_dict_recovered[
                "num_dim_reduced_components"
            ] = dim_reduced_outputs_modified.shape[1]
            dim_reduced_object = eval(
                used_dim_reduction_technique
                + "_kp(self.dataloader_kp_object, custom_"
                + used_dim_reduction_technique
                + "_dict_from_memory=dim_reduced_dict_recovered)"
            )
            data_preformatter_kp_object = data_preformatter_kp(dim_reduced_object)
            (
                segregated_training_testing_dict,
                image_patches_unrolled_dict,
                datagrid_store_dict,
            ) = data_preformatter_kp_object.data_preformatting_pipeline(
                image_patch_reject_percentage=data_preformatting_parameter_dict[
                    "image_patch_reject_percentage"
                ],
                image_patch_window_size=data_preformatting_parameter_dict[
                    "image_patch_window_size"
                ],
                image_patch_overlap=data_preformatting_parameter_dict[
                    "image_patch_overlap"
                ],
                segregate_data_training_percentage=data_preformatting_parameter_dict[
                    "segregate_data_training_percentage"
                ],
                segregate_data_random_select=data_preformatting_parameter_dict[
                    "segregate_data_random_select"
                ],
                save_data=0,
                preformatting_pipeline_filename_prefix="",
                segregate_data_repeatable_random_values=1,
            )

            self.segregated_data_dict_recovered = segregated_training_testing_dict

            parameter_dict_in_dataframe, best_parameter_dict = self.grid_search_cv_svm(
                parameter_grid_dict=parameter_grid_dict_for_svm["parameter_grid_dict"],
                cv_count=parameter_grid_dict_for_svm["grid_search_cv_count"],
                save_data=0,
                filename_prefix="",
                custom_scoring_function=parameter_grid_dict_for_svm[
                    "custom_scoring_function"
                ],
                num_parallel_processes=1,
            )

            per_process_main_comparison_table["per_process_count_array"].append(count)
            per_process_main_comparison_table[
                "per_process_component_set_removed"
            ].append(chosen_nmf_set)
            per_process_main_comparison_table["per_process_residual_accuracy"].append(
                np.round(best_parameter_dict["best_cv_score"], 3)
            )

            custom_scorer_name = best_parameter_dict["scorer_used"]
            with open(
                self.dataloader_kp_object.global_path_name
                + "kp_code/order_"
                + used_dim_reduction_technique
                + "_to_svm_accuracy_sub_process_"
                + str(process_number)
                + "_result.txt",
                "a",
            ) as file_1:  # Screen output update
                file_1.write(
                    "The best parameters are "
                    + str(best_parameter_dict["best_paras"])
                    + " with a score of "
                    + str(np.round(best_parameter_dict["best_cv_score"], 3))
                    + " as measured from "
                    + custom_scorer_name
                    + " score function \n___________\n"
                )
                file_1.close()
            print("___________\n")

        my_shared_memory_container.put(per_process_main_comparison_table)

    def use_multiprocessing_to_order_grouped_dimensionality_reduced_components_according_to_importance_to_svm_classification(
        self,
        saved_dim_reduced_filename="",
        num_components_per_group=3,
        data_preformatting_parameter_dict="",
        parameter_grid_dict_for_svm="",
        save_data=1,
        filename_prefix="",
        saved_ordered_combinations_filename="",
        cutoff_percentage=0.02,
    ):

        """
        @brief Use parallel processing to quickly do backward elimination of NMF components in combinations of a set of (example: 3) components at a time to find out which sets of components are most influential towards classsification accuracy based on the scorer given in 'custom_score_function' argument inside the parameter_grid_dict_for_svm

                Example usage:
                                    path_to_double_compact_msi_dataset = "/mnt/sda/kasun/double_compact_msi_data_negative_mode_1_bin_size.pickle"
                                    double_compact_msi_data = pickle.load(open(path_to_double_compact_msi_dataset, 'rb'))  # Load the compact version of msi datadouble_compact_msi_kp_object
                                    svm_kp_object = svm_kp(dataloader_kp_object=double_compact_msi_data)
                                    saved_dim_reduced_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/double_redo_batch_mode_test_negative_mode_0_05_binsize_/double_redo_batch_mode_test_negative_mode_0_05_binsize_individual_dataset_[1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
                                    filename_prefix = 'nmf_neg_mode_0_05_order_of_combinations_of_3_'
                                    save_data = 1
                                    data_preformatting_parameter_dict = {'image_patch_reject_percentage': 25, 'image_patch_window_size': 30, 'image_patch_overlap': 0, 'segregate_data_training_percentage': 80, 'segregate_data_random_select': 1}
                                    parameter_grid_dict_for_svm = {'parameter_grid_dict': {'C': np.logspace(1, 200, 5), 'gamma': np.logspace(0.01, 2, 5), 'kernel': ['rbf']}, 'grid_search_cv_count': 5, 'custom_scoring_function': 'fbeta'}
                                    num_components_per_group = 3
                                    cutoff_percentage = 0.01
                                    svm_kp_object.use_multiprocessing_to_order_grouped_dimensionality_reduced_components_according_to_importance_to_svm_classification(saved_dim_reduced_filename, num_components_per_group=num_components_per_group, data_preformatting_parameter_dict=data_preformatting_parameter_dict, parameter_grid_dict_for_svm=parameter_grid_dict_for_svm, save_data=save_data, filename_prefix=filename_prefix, cutoff_percentage=cutoff_percentage)

                                    ################### OR: simply load an existing saved dictionary

                                    saved_ordered_combinations_filename = "/home/kasun/aim_hi_project_kasun/kp_libmsi/saved_outputs/nmf_outputs/double_redo_batch_mode_test_negative_mode_0_05_binsize_/nmf_neg_mode_0_05_order_of_combinations_of_3_double_redo_batch_mode_test_negative_mode_0_05_binsize_individual_dataset_[1, 1, 1, 1, 1, 1, 1, 1]_nmf_dict_max_iter_10000_tic_normalized_nmf_outputs_20.npy"
                                    svm_kp_object.use_multiprocessing_to_order_grouped_dimensionality_reduced_components_according_to_importance_to_svm_classification(saved_ordered_combinations_filename=saved_ordered_combinations_filename, cutoff_percentage=cutoff_percentage)

        @param saved_dim_reduced_filename: Path to a saved nmf or pca dictionary
        @param num_components_per_group: Find which combination of 'num_components_per_group' number of dim reduced components chosen from the total number of dim reduced components will be most important for classification, and sort these groups of combinations of components in the order in which they are important
        @param data_preformatting_parameter_dict: A dictionary containing the parameters for the data preformatting task to happen. This includes the following keys: image_patch_reject_percentage', 'image_patch_window_size', 'image_patch_overlap','segregate_data_training_percentage','segregate_data_random_select'
        @param parameter_grid_dict_for_svm: The dictionary containing parameters over which the gridsearch for svm should be carried out. The keys for this dictionary should include the 'grid_search_cv_count' i.e, the number of cross validation iterations to run, the 'parameter_grid_dict', i.e a dictionary containing the 'C', 'gamma', and 'kernel' parameters, and finally 'custom_scoring_function', i.e the scorer to be used in grid search.
        @param save_data: Whether the save the final dataframe that contains all the information from the different combinations of components
        @param filename_prefix: The prefix to use when saving the dataframe
        @param saved_ordered_combinations_filename: If this is given, we do not have to run the ordering analysis again. A saved, already ordered list will be printed.
        @cutoff_percentage: This determines what percentage of combinations from the most important and least important ends to be considered for populaton based most important combinations determination (maximum is 1, implying 100%)
        @return: A dataframe containing all the details of the order of importance of different groups of dim_reduced component combinations
        """

        num_cores_to_be_used = (
            multiprocessing.cpu_count() - 1
        )  # Leave one core out to be used as the main parent core.

        if not (saved_ordered_combinations_filename):
            from nmf_kp import nmf_kp
            from pca_kp import pca_kp
            from data_preformatter_kp import data_preformatter_kp

            original_dim_reduced_dict_recovered = np.load(
                saved_dim_reduced_filename, allow_pickle=True
            )[()]
            dim_reduced_dict_recovered = copy.deepcopy(
                original_dim_reduced_dict_recovered
            )
            used_dim_reduction_technique = original_dim_reduced_dict_recovered[
                "used_dim_reduction_technique"
            ]
            original_dim_reduced_outputs_recovered = original_dim_reduced_dict_recovered[
                "dim_reduced_outputs"
            ][
                0
            ][
                0
            ]
            original_num_dim_reduced_components = original_dim_reduced_outputs_recovered.shape[
                1
            ]
            num_components_to_select = num_components_per_group
            main_comparison_table = {
                "count_array": [],
                "component_set_removed": [],
                "residual_accuracy": [],
            }

            comb = combinations(
                np.arange(original_num_dim_reduced_components), num_components_to_select
            )  # Going to select 3 components out of 20 NMF components for example, and gonna repeat for all such combinations
            combinations_list = list(comb)
            num_combinations = len(combinations_list)

            #############################
            # combinations_list = combinations_list[0:35]
            #############################

            per_process_combinations_list_store = np.array_split(
                np.array(combinations_list), num_cores_to_be_used
            )

            print("Starting the ordering process via multiprocessing")

            my_shared_memory_container = multiprocessing.Queue()

            process_array = []
            for process_number, this_process_combinations_list in enumerate(
                per_process_combinations_list_store
            ):
                process_array.append(
                    multiprocessing.Process(
                        target=self.per_process_ordering_dim_reduced_component_combinations_for_svm_accuracy,
                        args=(
                            this_process_combinations_list,
                            saved_dim_reduced_filename,
                            process_number,
                            num_components_per_group,
                            data_preformatting_parameter_dict,
                            parameter_grid_dict_for_svm,
                            my_shared_memory_container,
                        ),
                    )
                )

            for i in process_array:
                i.start()

            retrieved_unordered_child_data_store = []
            for j in range(len(per_process_combinations_list_store)):
                retrieved_unordered_child_data_store.append(
                    my_shared_memory_container.get()
                )

            for i in process_array:
                i.join()

            ordered_child_data_store = [0] * len(retrieved_unordered_child_data_store)
            for this_data_dict in retrieved_unordered_child_data_store:
                this_data_dict_number = this_data_dict["process_id"]
                ordered_child_data_store[this_data_dict_number] = this_data_dict

            count = 0
            for this_data_dict in ordered_child_data_store:
                num_subloops_in_this_process = len(
                    this_data_dict["per_process_count_array"]
                )
                this_data_dict_number = this_data_dict["process_id"]
                for i in range(num_subloops_in_this_process):
                    main_comparison_table["component_set_removed"].append(
                        this_data_dict["per_process_component_set_removed"][i]
                    )
                    main_comparison_table["residual_accuracy"].append(
                        this_data_dict["per_process_residual_accuracy"][i]
                    )
                    main_comparison_table["count_array"].append(count)
                    count = count + 1

            print("Ordering process complete")

            custom_scoring_function = parameter_grid_dict_for_svm[
                "custom_scoring_function"
            ]
            if custom_scoring_function == "fbeta":
                custom_scorer_name = custom_scoring_function
            elif custom_scoring_function == "kp_custom":
                custom_scorer_name = custom_scoring_function
            else:
                custom_scorer_name = "standard_sklearn"

            index_of_best_component_set_found = np.argmin(
                np.array(main_comparison_table["residual_accuracy"])
            )
            most_influential_component_set = main_comparison_table[
                "component_set_removed"
            ][index_of_best_component_set_found]
            residual_accuracy_after_removing_most_influencial_components = main_comparison_table[
                "residual_accuracy"
            ][
                index_of_best_component_set_found
            ]
            print(
                "Most influential component set based on "
                + custom_scorer_name
                + " score function: ",
                most_influential_component_set,
            )
            print(
                "Residual accuracy after removing those: ",
                residual_accuracy_after_removing_most_influencial_components,
            )

            residual_accuracies_array = np.array(
                main_comparison_table["residual_accuracy"]
            )
            count_array = np.array(main_comparison_table["count_array"])
            component_set_removed_array = np.array(
                main_comparison_table["component_set_removed"]
            )

            sorted_indices = np.argsort(residual_accuracies_array)

            sorted_residual_accuracy_array = residual_accuracies_array[sorted_indices]
            sorted_component_sets_removed_array = component_set_removed_array[
                sorted_indices
            ]

            ordered_combinations_of_nmf_based_on_importance_to_svm_dict = {
                "sorted_residual_accuracies_array": sorted_residual_accuracy_array,
                "sorted_component_sets_removed_array": sorted_component_sets_removed_array,
                "unsorted_main_comparison_table": main_comparison_table,
            }

            if save_data == 1:
                split_name = saved_dim_reduced_filename.split("/")
                file_path = split_name[:-1]
                file_name = split_name[-1]
                pathname_to_save = (
                    "/"
                    + os.path.join(*file_path)
                    + "/"
                    + filename_prefix
                    + "order_of_combinations_of_"
                    + str(num_components_per_group)
                    + "_for_svm_accuracy_on_"
                    + file_name
                )
                np.save(
                    pathname_to_save,
                    ordered_combinations_of_nmf_based_on_importance_to_svm_dict,
                    "dtype=object",
                )

        else:
            ordered_combinations_of_nmf_based_on_importance_to_svm_dict = np.load(
                saved_ordered_combinations_filename, allow_pickle=True
            )[()]
            print("Loaded saved dictionary. Will NOT recalculate the order")
            sorted_residual_accuracy_array = ordered_combinations_of_nmf_based_on_importance_to_svm_dict[
                "sorted_residual_accuracies_array"
            ]
            sorted_component_sets_removed_array = ordered_combinations_of_nmf_based_on_importance_to_svm_dict[
                "sorted_component_sets_removed_array"
            ]
            num_components_per_group = len(sorted_component_sets_removed_array[0])

        for i in range(len(sorted_residual_accuracy_array)):
            print(
                sorted_residual_accuracy_array[i],
                sorted_component_sets_removed_array[i],
            )

        most_important_combination_residual_accuracy = sorted_residual_accuracy_array[0]
        least_important_combination_residual_accuracy = sorted_residual_accuracy_array[
            -1
        ]
        combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy = np.empty(
            [0, num_components_per_group]
        )
        combinations_within_cutoff_percentage_of_least_important_combintion_residual_accuracy = np.empty(
            [0, num_components_per_group]
        )

        for i in range(len(sorted_residual_accuracy_array)):
            if (
                sorted_residual_accuracy_array[i]
                <= most_important_combination_residual_accuracy + cutoff_percentage
            ):
                combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy = np.vstack(
                    (
                        combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy,
                        sorted_component_sets_removed_array[i],
                    )
                )
            elif (
                sorted_residual_accuracy_array[i]
                >= least_important_combination_residual_accuracy - cutoff_percentage
            ):
                combinations_within_cutoff_percentage_of_least_important_combintion_residual_accuracy = np.vstack(
                    (
                        combinations_within_cutoff_percentage_of_least_important_combintion_residual_accuracy,
                        sorted_component_sets_removed_array[i],
                    )
                )

        print("\n____________\n")
        print("most important:")
        for (
            i
        ) in combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy:
            print(i)

        print("\n____________\n")

        print("least important:")
        for (
            j
        ) in combinations_within_cutoff_percentage_of_least_important_combintion_residual_accuracy:
            print(j)

        top_occurences = Counter(
            combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy.flatten()
        )
        bottom_occurences = Counter(
            combinations_within_cutoff_percentage_of_least_important_combintion_residual_accuracy.flatten()
        )

        best_collection = {}
        other_hits = {}

        for i in top_occurences.keys():
            if i not in bottom_occurences:
                best_collection[i] = top_occurences[i]
            else:
                if top_occurences[i] - bottom_occurences[i] > 0:
                    other_hits[i] = top_occurences[i] - bottom_occurences[i]

        best_collection_keys_sorted_by_value = sorted(
            best_collection, key=best_collection.get, reverse=True
        )
        other_hits_keys_sorted_by_value = sorted(
            other_hits, key=other_hits.get, reverse=True
        )

        print(
            "Most influential component set based on the combination-wise elimination: ",
            sorted_component_sets_removed_array[0],
        )
        print(
            "Population best components that occured only within ",
            cutoff_percentage,
            " percent of highest contributing combination's residual accuracy (Sorted in descending order of highest occurence): ",
            [int(i) for i in best_collection_keys_sorted_by_value],
            " with frequency ",
            [best_collection[i] for i in best_collection_keys_sorted_by_value],
            " respectively out of ",
            len(
                combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy
            ),
            " evaluated combinations",
        )
        print(
            "Population best components that occured within ",
            cutoff_percentage,
            " percent of both highest contributing and least contributing combination's residual accuracy (Sorted in descending order of highest difference in occurence): ",
            [int(i) for i in other_hits_keys_sorted_by_value],
            " with frequency ",
            [other_hits[i] for i in other_hits_keys_sorted_by_value],
            " respectively  out of ",
            len(
                combinations_within_cutoff_percentage_of_most_important_combintion_residual_accuracy
            ),
            " evaluated combinations",
        )

        print("\n")
        print(
            "percentage cutoff used to determine population best: ", cutoff_percentage
        )

        print("\n")
