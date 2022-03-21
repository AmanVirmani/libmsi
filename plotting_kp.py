from sklearnex import patch_sklearn

patch_sklearn()

import sys
# sys.path.insert(1,'/home/kasun/aim_hi_project_kasun/kp_libmsi/libmsi')
# import dataloader as d

import libmsi
import numpy as np
import random as rd
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.decomposition import FastICA, PCA, NMF
from sklearn.preprocessing import minmax_scale
from mpl_toolkits import mplot3d

from itertools import combinations

from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean

import pandas as pd
import pickle

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import os

# import cuml.manifold.t_sne as cuml_TSNE

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from scipy.signal import find_peaks

from PIL import Image

import cv2

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

rd.seed(0)


class plotting_kp:

    def __init__(self, svm_kp_object, saved_segregated_data_filename=None, saved_3d_datagrid_filename=None, plots_dpi=600, save_figures=0, plots_fileformat='svg', figure_saving_folder=''):

        self.save_figures = save_figures
        self.global_plots_fileformat = plots_fileformat
        self.global_plots_dpi = plots_dpi
        self.svm_kp_object = svm_kp_object
        self.dim_reduced_object = svm_kp_object.data_preformatter_object.dim_reduced_object
        self.dataloader_kp_object = svm_kp_object.data_preformatter_object.dim_reduced_object.dataloader_kp_object
        self.global_plots_figure_saving_folder = figure_saving_folder
        ##############

        self.global_plots_used_dim_reduction_technique = self.dim_reduced_object.used_dim_reduction_technique
        self.dim_reduced_dict_recovered = self.dim_reduced_object.dim_reduced_dict
        ##############

        self.global_plots_saved_segregated_data_filename = saved_segregated_data_filename
        self.global_plots_saved_3d_datagrid_filename = saved_3d_datagrid_filename
        ##############
        if (saved_3d_datagrid_filename is None) or (saved_3d_datagrid_filename == ''):
            self.global_plots_datagrid_store_dict_recovered = svm_kp_object.data_preformatter_object.datagrid_store_dict
        else:
            self.global_plots_datagrid_store_dict_recovered = np.load(self.global_plots_saved_3d_datagrid_filename, allow_pickle=True)[()]

        self.datagrid_store_recovered = self.global_plots_datagrid_store_dict_recovered['datagrid_store']
        self.global_plots_num_datasets = len(self.datagrid_store_recovered)
        self.global_plots_num_dim_reduced_components = self.datagrid_store_recovered[0].shape[0]
        ##############

        if (saved_segregated_data_filename is None) or (saved_segregated_data_filename == ''):
            self.global_plots_segregated_data_dict_recovered = svm_kp_object.data_preformatter_object.segregated_data_dict
        else:
            self.global_plots_segregated_data_dict_recovered = np.load(self.global_plots_saved_segregated_data_filename, allow_pickle=True)[()]

        self.global_plots_window_size = self.global_plots_segregated_data_dict_recovered['window_size_used_in_image_patch']

        train_x_data = self.global_plots_segregated_data_dict_recovered['x_train']
        num_patches = train_x_data.shape[0]
        reshaped_train_x_data = train_x_data.reshape(num_patches, self.global_plots_num_dim_reduced_components, self.global_plots_window_size, -1)
        self.mean_image_patches = np.mean(reshaped_train_x_data, axis=0)
        ##############

        v = cm.get_cmap('Blues', 512)
        v_map = v(np.linspace(0, 1, 512))  ### Take 256 colors from the 'Greens' colormap, and distribute it between 0 and 1.
        r = cm.get_cmap('Reds_r', 512)
        r_map = r(np.linspace(0, 1, 512))
        new = np.append(r_map, v_map, axis=0)
        new[int(new.shape[0] * 0.5 - 1):int(new.shape[0] * 0.5 + 1), :] = [1, 1, 1, 1]
        self.global_plots_new_map = ListedColormap(new)
        ##############

        self.global_plots_my_linear_svc = svm_kp_object.one_time_svc_results
        self.create_mean_combined_arrays()
        self.min_mz_after_truncation = self.svm_kp_object.data_preformatter_object.dim_reduced_object.dataloader_kp_object.min_mz_after_truncation
        self.max_mz_after_truncation = self.svm_kp_object.data_preformatter_object.dim_reduced_object.dataloader_kp_object.max_mz_after_truncation

    def plot_normalized_hist_svm_weight_vectors(self):

        ######################
        ### Simple mean (per nmf) of linear SVM weights
        print('\n')
        for i in range(self.global_plots_num_dim_reduced_components):
            print('Mean for component ', i + 1, ' = ', np.abs(np.mean(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i])))
        print('\n')

        ######################
        ### Plot the Raw SVM weights' histograms for each Dim reduced dataset component

        fig, ax = plt.subplots(1, self.global_plots_num_dim_reduced_components, sharey=True)
        for i in range(self.global_plots_num_dim_reduced_components):
            n, bins, patches = ax[i].hist(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i], bins=50, range=(-3, 3), density=True)
            ax[i].spines[['right', 'top']].set_visible(0)
            ax[i].set_title('SVM Weights' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].yaxis.set_tick_params(length=20, width=5, labelsize=25)
            ax[i].xaxis.set_tick_params(length=20, width=5, labelsize=25)

            mu = np.mean(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i])  # mean of distribution
            sigma = np.std(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i])  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[i].plot(bins, y, 'r--', linewidth=5)
            ax[i].set_xlabel('mu: ' + str(np.round(mu, 3)) + '\n' + 'std: ' + str(np.round(sigma, 3)), fontsize=20)
            ax[i].set_ylim([0, 0.8])

        plot_title = 'Normalized Histograms of SVM Weight vectors for each ' + self.global_plots_used_dim_reduction_technique + ' component'
        plt.suptitle(plot_title, x=6, y=1.6, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=12, top=1)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

    def plot_svm_weight_vectors_in_flattened_form(self, color_min=-3, color_max=3):

        ### Plot the Raw SVM weight vectors for each Dim Reduced Dataset component as an unrolled set of vectors.

        fig, ax = plt.subplots(1, 1)
        my_mappable = ax.imshow(np.abs(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)), cmap=self.global_plots_new_map, vmin=color_min, vmax=color_max)
        ax.yaxis.set_tick_params(length=20, width=5, labelsize=25)
        ax.xaxis.set_tick_params(length=20, width=5, labelsize=25)

        plot_title = 'SVM Weight vectors in flattened form for each ' + self.global_plots_used_dim_reduction_technique + ' component'
        plt.suptitle(plot_title, x=6, y=1.2, fontsize=50)
        plt.subplots_adjust(left=0, bottom=0, right=12, top=1)
        plt.grid(False)

        ### Plot the colorbar used
        my_ax = fig.add_axes([3, -0.4, 6, 0.2])
        cbar = fig.colorbar(mappable=my_mappable, cax=my_ax, orientation='horizontal')
        cbar.ax.tick_params(length=10, width=5, labelsize=30)
        cbar.ax.set_xlabel('Colorbar', fontsize=30)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

    def plot_mean_image_patches_for_each_dim_reduced_component(self):
        ### Plot the Mean image patch spatial distribution for each dim reduced dataset component

        fig, ax = plt.subplots(1, self.global_plots_num_dim_reduced_components, sharey=True)
        for i in range(self.global_plots_num_dim_reduced_components):
            my_mappable = ax[i].imshow(self.mean_image_patches[i])  # , cmap=self.global_plots_new_map,vmin=color_min/1e1, vmax=color_max/1e1)
            ax[i].set_title('SVM Weights' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
            ax[i].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
            ax[i].set_ylabel(i + 1, fontsize=20)

        plot_title = 'Mean image patches for each ' + self.global_plots_used_dim_reduction_technique + ' component'
        plt.suptitle(plot_title, x=6, y=1.5, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=12, top=1)

        ### Plot the colorbar used
        my_ax = fig.add_axes([3, -0.4, 6, 0.2])
        cbar = fig.colorbar(mappable=my_mappable, cax=my_ax, orientation='horizontal')
        cbar.ax.tick_params(length=10, width=5, labelsize=30)
        cbar.ax.set_xlabel('Colorbar', fontsize=30)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

    def plot_svm_weight_vectors_reshaped_as_ion_image_patches(self, color_min=-3, color_max=3):
        ### Plot the SVM weight vectors for each component, reshaped as ion image patches.

        fig, ax = plt.subplots(1, self.global_plots_num_dim_reduced_components, sharey=True)
        for i in range(self.global_plots_num_dim_reduced_components):
            my_mappable = ax[i].imshow(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, self.global_plots_window_size, -1)[i], cmap=self.global_plots_new_map, vmin=color_min, vmax=color_max)
            #     my_mappable=ax[i].imshow(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components,self.global_plots_window_size,-1)[i])
            mean_compensated_weight = np.round(np.dot(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i], self.mean_image_patches[i].ravel()), 3)
            ax[i].set_title('SVM Weights' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
            ax[i].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
            ax[i].set_ylabel(i + 1, fontsize=20)
            ax[i].set_xlabel('Mean\n Weight:\n' + str(mean_compensated_weight), fontsize=40)

        plot_title = 'Uncompensated SVM Weight vectors in image patch form for each ' + self.global_plots_used_dim_reduction_technique + ' component (Compensated mean weights shown under each patch)'
        plt.suptitle(plot_title, x=6, y=2, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=12, top=1)

        ### Plot the colorbar used
        my_ax = fig.add_axes([3, -0.8, 6, 0.2])
        cbar = fig.colorbar(mappable=my_mappable, cax=my_ax, orientation='horizontal')
        cbar.ax.tick_params(length=10, width=5, labelsize=30)
        cbar.ax.set_xlabel('Colorbar', fontsize=30)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

    def plot_histograms_for_mean_image_paches_of_each_dim_reduced_comp(self):
        ### Plot the histograms for mean image patches corresponding to each Dim reduced dataset component

        fig, ax = plt.subplots(1, self.global_plots_num_dim_reduced_components)
        for i in range(self.global_plots_num_dim_reduced_components):
            n, bins, patches = ax[i].hist(self.mean_image_patches[i].ravel(), bins=50, density=True)
            ax[i].spines[['right', 'top']].set_visible(0)
            ax[i].set_title('Mean Patch' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].yaxis.set_tick_params(length=20, width=5, labelsize=25)
            ax[i].xaxis.set_tick_params(length=20, width=5, labelsize=25, rotation=45)

            mu = np.mean(self.mean_image_patches[i].ravel())  # mean of distribution
            sigma = np.std(self.mean_image_patches[i].ravel())  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[i].plot(bins, y, 'r--', linewidth=5)
            ax[i].set_xlabel('mu: ' + str(np.round(mu, 4)) + '\n' + 'std: ' + str(np.round(sigma, 5)), fontsize=20)
        #     ax[i].set_ylim([0,0.8])

        plot_title = 'Normalized Histograms of mean image patches for each ' + self.global_plots_used_dim_reduction_technique + ' component (Horizontally Arranged)'
        plt.suptitle(plot_title, x=6, y=1.6, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=15, top=1)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

    def plot_hists_for_mean_image_patches_of_each_dim_reduced_comp_vertically_down(self):
        ### Plot the histograms for mean image patches corresponding to each Dim reduced dataset component vertically down

        fig, ax = plt.subplots(self.global_plots_num_dim_reduced_components, 1)
        for i in range(self.global_plots_num_dim_reduced_components):
            n, bins, patches = ax[i].hist(self.mean_image_patches[i].ravel(), bins=500, range=(0, 0.035), density=True)
            ax[i].spines[['right', 'top']].set_visible(0)
            ax[i].set_ylabel('Mean \n' + 'Patch' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=50)
            ax[i].yaxis.set_tick_params(length=20, width=5, labelsize=40)
            ax[i].xaxis.set_tick_params(length=20, width=5, labelsize=40, rotation=0)

            mu = np.mean(self.mean_image_patches[i].ravel())  # mean of distribution
            sigma = np.std(self.mean_image_patches[i].ravel())  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[i].plot(bins, y, 'r--', linewidth=5)
            #     ax[i].set_xlabel('mu: '+str(np.round(mu,4)) +'\n'+'std: '+str(np.round(sigma,5)), fontsize=20)
            #     ax[i].set_ylim([0,0.8])
            ax[i].grid('OFF', axis='x')

        plot_title = 'Normalized Histograms of mean image patches for each ' + self.global_plots_used_dim_reduction_technique + ' component (Vertically Arranged)'
        plt.suptitle(plot_title, x=6, y=26, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=15, top=25)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')

        print('\n')

    def print_compensated_linear_svm_weights(self):
        ### Printing the compensated weights
        print('\n')

        for i in range(self.global_plots_num_dim_reduced_components):
            print('Compensated Mean SVM Weight for component ', i + 1, ' = ',
                  np.dot(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i],
                         self.mean_image_patches[i].ravel()))

        print('\n')

    def plot_pixelwise_product_of_mean_image_patches_and_svm_weight_vectors(self, color_min=-3, color_max=3):
        ### Pixelwise product of mean image patches and SVM weight vectors.

        fig, ax = plt.subplots(1, self.global_plots_num_dim_reduced_components, sharey=True)
        for i in range(self.global_plots_num_dim_reduced_components):
            im_svc = self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, self.global_plots_window_size, -1)[i]
            im_mean_patch = self.mean_image_patches[i]
            my_mappable = ax[i].imshow(np.multiply(im_svc, im_mean_patch), cmap=self.global_plots_new_map, vmin=color_min / 50, vmax=color_max / 50)
            #     my_mappable=ax[i].imshow(np.multiply(im_svc,im_mean_patch))
            ax[i].set_title('SVM Weights' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
            ax[i].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
            ax[i].set_ylabel(i + 1, fontsize=20)
        plot_title = 'Elementwise product of mean image patches and SVM Weight vectors in image patch form for each ' + self.global_plots_used_dim_reduction_technique + ' component'
        plt.suptitle(plot_title, x=6, y=1.5, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=12, top=1)

        ### Plot the colorbar used
        my_ax = fig.add_axes([3, -0.4, 6, 0.2])
        cbar = fig.colorbar(mappable=my_mappable, cax=my_ax, orientation='horizontal')
        cbar.ax.tick_params(length=10, width=5, labelsize=30)
        cbar.ax.set_xlabel('Colorbar', fontsize=30)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi, bbox_inches='tight')
        print('\n')

#############################################################################################
#############################################################################################
#############################################################################################

    def plot_mean_cph_naive_image_patch_spatial_distributions(self):
        ### Plot the Mean CPH and Naive image patch spatial distributions for each dim reduced dataset component

        train_x_data = self.global_plots_segregated_data_dict_recovered['x_train']
        train_y_data = self.global_plots_segregated_data_dict_recovered['y_train']
        num_patches = train_x_data.shape[0]
        reshaped_train_x_data = train_x_data.reshape(num_patches, self.global_plots_num_dim_reduced_components, self.global_plots_window_size, -1)

        cph_data = reshaped_train_x_data[(train_y_data == 0).ravel()]
        naive_data = reshaped_train_x_data[(train_y_data == 1).ravel()]

        mean_cph_patch = np.mean(cph_data, axis=0)
        mean_naive_patch = np.mean(naive_data, axis=0)

        combined_arr = [mean_cph_patch, mean_naive_patch]


        fig, ax = plt.subplots(2, self.global_plots_num_dim_reduced_components, sharey=True)
        for i in range(self.global_plots_num_dim_reduced_components):
            for j in range(2):
                this_vmin = np.min([np.min(combined_arr[0][i]), np.min(combined_arr[1][i])])
                this_vmax = np.max([np.max(combined_arr[0][i]), np.max(combined_arr[1][i])])
                my_mappable = ax[j, i].imshow(combined_arr[j][i], cmap='Blues', vmin=this_vmin, vmax=this_vmax)
                ax[j, i].set_title('Mean ' + str(self.name_arr[j]) + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=70)
                ax[j, i].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                ax[j, i].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                ax[j, i].set_ylabel(i + 1, fontsize=20)

            cbar = fig.colorbar(mappable=my_mappable, ax=ax[1, i], orientation='horizontal', ticks=[this_vmin, this_vmax])
            cbar.ax.tick_params(length=10, width=5, labelsize=60, rotation=70)

        plot_title = 'Mean image patches for CPH and Naive datasets for each ' + self.global_plots_used_dim_reduction_technique + ' component'
        plt.suptitle(plot_title, x=10, y=4.5, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=20, top=4)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')
        print('\n')

    def create_mean_combined_arrays(self):

        train_x_data = self.global_plots_segregated_data_dict_recovered['x_train']
        train_y_data = self.global_plots_segregated_data_dict_recovered['y_train']
        num_patches = train_x_data.shape[0]
        reshaped_train_x_data = train_x_data.reshape(num_patches, self.global_plots_num_dim_reduced_components, -1)

        cph_data = reshaped_train_x_data[(train_y_data == 0).ravel()]
        naive_data = reshaped_train_x_data[(train_y_data == 1).ravel()]

        svc_weight_vectors = self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)
        broadcasted_svc_weight_vectors_for_cph = np.tile(svc_weight_vectors, (cph_data.shape[0], 1, 1))
        broadcasted_svc_weight_vectors_for_naive = np.tile(svc_weight_vectors, (naive_data.shape[0], 1, 1))

        element_wise_product_cph = np.multiply(cph_data, broadcasted_svc_weight_vectors_for_cph)
        mean_predicted_class_for_each_cph_sample_for_each_component = np.sum(element_wise_product_cph, axis=2)
        mean_predicted_class_for_cph_for_each_component = np.mean(mean_predicted_class_for_each_cph_sample_for_each_component, axis=0)
        std_predicted_class_for_cph_for_each_component = np.std(mean_predicted_class_for_each_cph_sample_for_each_component, axis=0)

        element_wise_product_naive = np.multiply(naive_data, broadcasted_svc_weight_vectors_for_naive)
        mean_predicted_class_for_each_naive_sample_for_each_component = np.sum(element_wise_product_naive, axis=2)
        mean_predicted_class_for_naive_for_each_component = np.mean(mean_predicted_class_for_each_naive_sample_for_each_component, axis=0)
        std_predicted_class_for_naive_for_each_component = np.std(mean_predicted_class_for_each_naive_sample_for_each_component, axis=0)

        self.combined_arr_1 = [mean_predicted_class_for_each_cph_sample_for_each_component.T, mean_predicted_class_for_each_naive_sample_for_each_component.T]
        self.name_arr = ['cph', 'naive']
        self.mean_predicted_class_for_each_cph_sample_for_each_component = mean_predicted_class_for_each_cph_sample_for_each_component
        self.mean_predicted_class_for_each_naive_sample_for_each_component = mean_predicted_class_for_each_naive_sample_for_each_component
        self.mean_predicted_class_for_cph_for_each_component = mean_predicted_class_for_cph_for_each_component
        self.mean_predicted_class_for_naive_for_each_component = mean_predicted_class_for_naive_for_each_component

    def plot_normalized_histograms_of_svm_predictions(self):
        ### Normalized histograms of the svm 'class predictions for each nmf component' for each image patch for cph and naive separately

        fig, ax = plt.subplots(2, self.global_plots_num_dim_reduced_components)
        for i in range(self.global_plots_num_dim_reduced_components):
            n_max = 0
            for j in range(2):
                this_vmin = np.min([np.min(self.combined_arr_1[0][i]), np.min(self.combined_arr_1[1][i])])
                this_vmax = np.max([np.max(self.combined_arr_1[0][i]), np.max(self.combined_arr_1[1][i])])
                n, bins, patches = ax[j, i].hist(self.combined_arr_1[j][i], bins=100, range=(this_vmin, this_vmax), density=True)
                n_max = np.max([np.max(n), n_max])
                ax[j, i].set_title(str(self.name_arr[j]) + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=40)
                ax[j, i].spines[['right', 'top']].set_visible(0)
                ax[j, i].yaxis.set_tick_params(length=20, width=5, labelsize=40)
                ax[j, i].xaxis.set_tick_params(length=20, width=5, labelsize=40, rotation=0)

                mu = np.mean(self.combined_arr_1[j][i])  # mean of distribution
                sigma = np.std(self.combined_arr_1[j][i])  # standard deviation of distribution
                y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

                ax[j, i].plot(bins, y, 'r--', linewidth=5)
                ax[j, i].set_xlabel('mu: ' + str(np.round(mu, 4)) + '\n' + 'std: ' + str(np.round(sigma, 5)), fontsize=40)
                ax[j, i].grid('OFF', axis='x')

            ax[0, i].set_ylim([0, n_max])
            ax[1, i].set_ylim([0, n_max])

        plot_title = 'Normalized Histograms of svm predictions (' + r'$w^{(k)}x_i^{(k)}$' + ') for each data patch ' + r'$x_i$' + ' for each ' + self.global_plots_used_dim_reduction_technique + ' component ' + r'$k$'
        plt.suptitle(plot_title, x=12.5, y=11, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=25, top=10)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')
        print('\n')

    def plot_vertically_arranged_normalized_hists_of_svm_predictions(self):
        ### Vertically arranged Normalized histograms of the svm 'class predictions for each nmf component' for each image patch for cph and naive separately

        fig, ax = plt.subplots(self.global_plots_num_dim_reduced_components * 2, 1)
        this_max = 0
        this_min = 1000
        n_max = 0
        for i in range(self.global_plots_num_dim_reduced_components):
            this_max = np.max([this_max, np.max(self.combined_arr_1[0][i]), np.quantile(self.combined_arr_1[1][i], 0.8)])
            this_min = np.min([this_min, np.min(self.combined_arr_1[0][i]), np.quantile(self.combined_arr_1[1][i], 0.2)])

        for i in range(self.global_plots_num_dim_reduced_components):
            # cph
            n, bins, patches = ax[2 * i].hist(self.combined_arr_1[0][i], bins=500, density=True, range=(this_min, this_max), color='b')
            ax[2 * i].spines[['right']].set_visible(0)
            ax[2 * i].yaxis.set_tick_params(length=20, width=5, labelsize=40)
            ax[2 * i].xaxis.set_tick_params(length=20, width=5, labelsize=40, rotation=0)

            mu = np.mean(self.combined_arr_1[0][i])  # mean of distribution
            sigma = np.std(self.combined_arr_1[0][i])  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[2 * i].plot(bins, y, 'r--', linewidth=5)
            ax[2 * i].set_ylabel(
                self.name_arr[0] + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1) + '\n' + 'mu: ' + str(
                    np.round(mu, 4)) + '\n' + 'std: ' + str(np.round(sigma, 5)), fontsize=50)
            ax[2 * i].grid('OFF', axis='x')

            # Naive
            n, bins, patches = ax[2 * i + 1].hist(self.combined_arr_1[1][i], bins=500, density=True, range=(this_min, this_max),
                                                  color='r')
            ax[2 * i + 1].spines[['right', 'top']].set_visible(0)
            ax[2 * i + 1].yaxis.set_tick_params(length=20, width=5, labelsize=40)
            ax[2 * i + 1].xaxis.set_tick_params(length=20, width=5, labelsize=40, rotation=0)

            mu = np.mean(self.combined_arr_1[1][i])  # mean of distribution
            sigma = np.std(self.combined_arr_1[1][i])  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[2 * i + 1].plot(bins, y, 'r--', linewidth=5)
            ax[2 * i + 1].set_ylabel(
                self.name_arr[1] + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1) + '\n' + 'mu: ' + str(
                    np.round(mu, 4)) + '\n' + 'std: ' + str(np.round(sigma, 5)), fontsize=50)
            ax[2 * i + 1].grid('OFF', axis='x')

        plot_title = 'Normalized Histograms of svm predictions (' + r'$w^{(k)}x_i^{(k)}$' + ') for each data patch ' + r'$x_i$' + ' for each ' + self.global_plots_used_dim_reduction_technique + ' component ' + r'$k$'
        plt.suptitle(plot_title, x=5, y=61, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=10, top=60)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat,
                        dpi=self.global_plots_dpi - 100, bbox_inches='tight')

        print('\n')

    def plot_norm_hists_svm_predicts_for_each_naive_and_cph_patch_separately(self):
        ### Normalized histograms of the final svm class predictions for each image patch for cph and naive separately

        all_cph_predictions = np.sum(self.mean_predicted_class_for_each_cph_sample_for_each_component, axis=1)
        all_naive_predictions = np.sum(self.mean_predicted_class_for_each_naive_sample_for_each_component, axis=1)

        combined_arr = [all_cph_predictions.T, all_naive_predictions.T]


        fig, ax = plt.subplots(2, 1)

        this_vmin = np.min([np.min(combined_arr[0]), np.min(combined_arr[1])])
        this_vmax = np.max([np.max(combined_arr[0]), np.max(combined_arr[1])])

        for j in range(2):
            n, bins, patches = ax[j].hist(combined_arr[j], bins=100, range=(this_vmin, this_vmax), density=True)
            ax[j].set_ylabel('SVM Predictions (' + r'$XW$' + ') for \n' + str(
                self.name_arr[j]) + ' ' + self.global_plots_used_dim_reduction_technique + '\n (Y-axis is Truncated)', fontsize=70)
            ax[j].spines[['right', 'top']].set_visible(0)
            ax[j].yaxis.set_tick_params(length=20, width=5, labelsize=40)
            ax[j].xaxis.set_tick_params(length=20, width=5, labelsize=40, rotation=0)

            mu = np.mean(combined_arr[j])  # mean of distribution
            sigma = np.std(combined_arr[j])  # standard deviation of distribution
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))  # Fit a gaussian

            ax[j].plot(bins, y, 'r--', linewidth=5)
            ax[j].set_xlabel('mu: ' + str(np.round(mu, 4)) + '   ' + ' std: ' + str(np.round(sigma, 5)), fontsize=70)
            ax[j].grid('OFF', axis='x')

            ax[j].set_ylim([0, 1])

        plot_title = 'Normalized Histograms of svm predictions (' + r'$XW$' + ') for each data patch ' + r'$X_i$' + ' (' + self.global_plots_used_dim_reduction_technique + ' Data) for CPH and Naive data separately'
        plt.suptitle(plot_title, x=7.5, y=12.5, fontsize=80)
        plt.subplots_adjust(bottom=0, left=0, right=15, top=12)

        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')
        print('\n')

    def print_compensated_weights_cph_and_naive_separately(self):
        ### Printing the compensated weights for CPH and Naive separately
        print('\n')

        for i in range(self.global_plots_num_dim_reduced_components):
            #     normalized_difference=np.round(((mean_predicted_class_for_cph_for_each_component[i])-(mean_predicted_class_for_naive_for_each_component[i]))/np.sqrt(np.abs((mean_predicted_class_for_cph_for_each_component[i])*(mean_predicted_class_for_naive_for_each_component[i]))),4)
            difference = np.round(((self.mean_predicted_class_for_cph_for_each_component[i]) - (self.mean_predicted_class_for_naive_for_each_component[i])), 4)
            print('Compensated Mean SVM Weight for component ', i + 1, '\t:  For CPH: ', np.round(self.mean_predicted_class_for_cph_for_each_component[i], 4), '\t For Naive: ', np.round(self.mean_predicted_class_for_naive_for_each_component[i], 4), '\t Difference: ', difference)

        print('\n')

#############################################################################################
#############################################################################################
#############################################################################################

    def plot_dim_reduced_ion_images_common_colorbar_along_vertical(self):
        ### See what ion images of our NMF/PCA components look like - Plots with a common colorbar ONLY across datasets.

        ### Infrastructure
        if (self.global_plots_used_dim_reduction_technique == 'nmf'):
            col_vmin = -0.06
            col_vmax = 0.06
        elif (self.global_plots_used_dim_reduction_technique == 'pca'):
            col_vmin = -0.008
            col_vmax = 0.008

        datagrid_store_recovered = self.global_plots_datagrid_store_dict_recovered['datagrid_store']

        fig, ax = plt.subplots(self.global_plots_num_datasets + 1, self.global_plots_num_dim_reduced_components, gridspec_kw={'height_ratios': np.append(np.ones(self.global_plots_num_datasets), 0.1)})

        for j in range(self.global_plots_num_dim_reduced_components):
            this_component_min_intensity = 10000
            this_component_max_intensity = 0

            for i in range(self.global_plots_num_datasets):
                im_store = datagrid_store_recovered[i]
                im = im_store[j]
                if self.global_plots_used_dim_reduction_technique == 'nmf':
                    im[im == -1] = 0
                elif self.global_plots_used_dim_reduction_technique == 'pca':
                    im[im == 1000] = 0
                #         ax[j].imshow(im, cmap='viridis', vmin=-0.03, vmax=0.03).set_interpolation('none')

                this_component_min_intensity = np.min([this_component_min_intensity, np.quantile(im, 0.99)])
                this_component_max_intensity = np.max([this_component_max_intensity, np.quantile(im, 0.01)])

                my_ax = ax[i, j].imshow(im, cmap=self.global_plots_new_map)
                ax[i, j].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                ax[i, j].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                ax[i, j].set_title(self.global_plots_used_dim_reduction_technique + '  ' + str(j + 1), fontsize=40)
                ax[i, j].set_ylabel(self.name_arr[int(np.floor(i / 4))] + '  ' + str((i % 4) + 1), fontsize=40)

            best_cmap_lim = np.max(np.abs([this_component_min_intensity, this_component_max_intensity]))  # -0.2*np.max(np.abs([this_component_min_intensity,this_component_max_intensity]))
            for i in range(self.global_plots_num_datasets):
                my_axes_image = ax[i, j].get_images()[0]
                my_axes_image.set_clim([- 1 * best_cmap_lim, best_cmap_lim])

            cbar = plt.colorbar(mappable=my_axes_image, cax=ax[self.global_plots_num_datasets, j], orientation='horizontal', ticks=[-1 * best_cmap_lim, 0, best_cmap_lim])
            cbar.ax.tick_params(length=10, width=5, labelsize=40, rotation=70)

        plot_title = 'Ion images for each dataset for each  ' + self.global_plots_used_dim_reduction_technique + ' component. ' + '(Colormap is only uniform across individual columns)'
        plt.suptitle(plot_title, x=7.5, y=10.5, fontsize=80)
        fig.supylabel('Datasets', x=-0.5, y=5, fontsize=40)
        fig.supxlabel(self.global_plots_used_dim_reduction_technique + ' Components', x=7.5, y=-1, fontsize=40)
        plt.subplots_adjust(left=0, bottom=0, right=15, top=10)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

        print('\n')

    def plot_dim_reduced_ion_images_common_colorbar_throughout(self):
        ### See what ion images of our NMF components look like - Plots with a common colorbar across ALL datasets and ALL components.

        ####################################################
        ### Infrastructure
        if (self.global_plots_used_dim_reduction_technique == 'nmf'):
            col_vmin = -0.06
            col_vmax = 0.06
        elif (self.global_plots_used_dim_reduction_technique == 'pca'):
            col_vmin = -0.008
            col_vmax = 0.008

        fig, ax = plt.subplots(self.global_plots_num_datasets + 1, self.global_plots_num_dim_reduced_components)

        for i in range(self.global_plots_num_datasets):

            im_store = self.datagrid_store_recovered[i]
            for j in range(self.global_plots_num_dim_reduced_components):
                im = im_store[j]
                if self.global_plots_used_dim_reduction_technique == 'nmf':
                    im[im == -1] = 0
                elif self.global_plots_used_dim_reduction_technique == 'pca':
                    im[im == 1000] = 0
                #         ax[j].imshow(im, cmap='viridis', vmin=-0.03, vmax=0.03).set_interpolation('none')
                my_ax = ax[i, j].imshow(im, cmap=self.global_plots_new_map, vmin=col_vmin, vmax=col_vmax)
                #         ax[i,j].axis("off")
                ax[i, j].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                ax[i, j].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                ax[i, j].set_title(self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=40)
                ax[i, j].set_ylabel(self.name_arr[int(np.floor(i / 4))] + ' ' + str((i % 4) + 1), fontsize=40)

        # Add the plots of the NMF spectra for each component (H matrix)
        dim_reduced_dict_recovered = self.dim_reduced_dict_recovered
        dim_reduced_outputs_recovered = dim_reduced_dict_recovered['dim_reduced_outputs']

        self.h_matrix = dim_reduced_outputs_recovered[1][0]
        self.x_axis_ticks = np.linspace(self.min_mz_after_truncation, self.max_mz_after_truncation, len(self.h_matrix[0]))

        for j in range(20):
            ax[-1, j].stem(self.x_axis_ticks, self.h_matrix[j], markerfmt='None')

            ax[-1, j].set_title(self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=50)

            ax[-1, j].set_xlabel('mz', fontsize=15)
            ax[-1, j].xaxis.set_label_coords(1.05, -0.001)

        my_ax2 = fig.add_axes([0, -0.5, 2, 0.1])
        cbar = fig.colorbar(mappable=my_ax, cax=my_ax2, orientation='horizontal')
        cbar.ax.tick_params(length=10, width=5, labelsize=40, rotation=70)

        plot_title = 'Ion images for each dataset for each ' + self.global_plots_used_dim_reduction_technique + ' component. (Colormap is uniform throughout the plot)'
        plt.suptitle(plot_title, x=7.5, y=15.5, fontsize=80)
        fig.supylabel('Datasets', x=-0.5, y=7.5, fontsize=40)
        fig.supxlabel(self.global_plots_used_dim_reduction_technique + ' Components', x=7.5, y=-0.5, fontsize=40)
        plt.subplots_adjust(left=0, bottom=0, right=15, top=15)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

        print('\n')
        #####################################

        ### plots of the NMF spectra for each component (H matrix) arranged vertically
        fig, ax = plt.subplots(self.global_plots_num_dim_reduced_components, 1)

        for j in range(20):
            markers, stems, base = ax[j].stem(self.x_axis_ticks, self.h_matrix[j], markerfmt='None', )
            stems.set_linewidth(2)
            ax[j].set_ylabel('Intensity\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=50)
            ax[j].tick_params(length=10, width=5, labelsize=40)
            ax[j].spines[['right', 'top']].set_visible(0)
            ax[j].grid(1, which='both', axis='x')
            ax[j].set_xlabel('mz', fontsize=30)
            ax[j].xaxis.set_label_coords(1.03, -0.001)

            max_for_this_spectrum = np.max(self.h_matrix[j])
            this_spectrum = self.h_matrix[j]
            peaks, _ = find_peaks(this_spectrum, height=max_for_this_spectrum / 20, threshold=max_for_this_spectrum / 10)
            for k in peaks:
                ax[j].text(self.min_mz_after_truncation + k, this_spectrum[k] + 0.01, str(self.min_mz_after_truncation + k), size=20, rotation=90, ha='center')  # x coordinate, y coordinate, label

        plot_title = 'MSI Spectra (dimensionality reduced) for each ' + self.global_plots_used_dim_reduction_technique + ' component (Arranged Vertically down for clarity)'
        plt.suptitle(plot_title, x=2.5, y=30.5, fontsize=50)
        plt.subplots_adjust(left=0, bottom=0, right=5, top=30)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

    def plot_dim_reduced_spectra_for_each_component_common_vertical_lims(self):
        ### plots of the NMF spectra for each component (H matrix) arranged vertically, with common vertical axes limits.

        fig, ax = plt.subplots(self.global_plots_num_dim_reduced_components, 1)

        max_y_value = 0
        min_y_value = 1000
        for j in range(20):
            markers, stems, base = ax[j].stem(self.x_axis_ticks, self.h_matrix[j], markerfmt='None', )
            stems.set_linewidth(2)
            ax[j].set_ylabel('Intensity\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=50)
            ax[j].tick_params(length=10, width=5, labelsize=40)
            ax[j].spines[['right', 'top']].set_visible(0)
            ax[j].grid(1, which='both', axis='x')
            ax[j].set_xlabel('mz', fontsize=30)
            ax[j].xaxis.set_label_coords(1.03, -0.001)
            max_y_value = max([max_y_value, np.max(self.h_matrix[j])])
            min_y_value = min([min_y_value, np.min(self.h_matrix[j])])

            this_spectrum = self.h_matrix[j]
            peaks, _ = find_peaks(this_spectrum, height=0.05, threshold=0.08)
            for k in peaks:
                ax[j].text(self.min_mz_after_truncation + k, this_spectrum[k] + 0.05, str(self.min_mz_after_truncation + k), size=20, rotation=90, ha='center')  # x coordinate, y coordinate, label

        for j in range(20):
            ax[j].yaxis.axes.set_ylim(min_y_value, max_y_value)

        plot_title = 'MSI Spectra (dimensionality reduced) for each ' + self.global_plots_used_dim_reduction_technique + ' component (Consistent Vertical Scale)'
        plt.suptitle(plot_title, x=2.5, y=30.5, fontsize=50)
        plt.subplots_adjust(left=0, bottom=0, right=5, top=30)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

    def plot_select_dim_reduced_spectra_common_vertical_arrangement(self):
        ### plots of the NMF spectra for select components, arranged vertically, with common vertical axes limits.

        components_needed = [1, 8, 9, 10, 13, 14, 16]  # indexing starts at 0. Therefore need to subtract 1 from what I actually want

        fig, ax = plt.subplots(len(components_needed), 1)

        max_y_value = 0
        min_y_value = 1000
        for count, j in enumerate(components_needed):
            markers, stems, base = ax[count].stem(self.x_axis_ticks, self.h_matrix[j], markerfmt='None', )
            stems.set_linewidth(2)
            ax[count].set_ylabel('Intensity\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=50)
            ax[count].tick_params(length=10, width=5, labelsize=40)
            ax[count].spines[['right', 'top']].set_visible(0)
            ax[count].grid(1, which='both', axis='x')
            ax[count].set_xlabel('mz', fontsize=30)
            ax[count].xaxis.set_label_coords(1.03, -0.001)
            max_y_value = max([max_y_value, np.max(self.h_matrix[j])])
            min_y_value = min([min_y_value, np.min(self.h_matrix[j])])

            this_spectrum = self.h_matrix[j]
            peaks, _ = find_peaks(this_spectrum, height=0.05, threshold=0.08)
            for k in peaks:
                ax[count].text(self.min_mz_after_truncation + k, this_spectrum[k] + 0.01, str(self.min_mz_after_truncation + k), size=20, rotation=90, ha='center')  # x coordinate, y coordinate, label

        for count in range(len(components_needed)):
            ax[count].yaxis.axes.set_ylim(min_y_value, max_y_value)

        plot_title = 'MSI Spectra (dimensionality reduced) for Select ' + self.global_plots_used_dim_reduction_technique + ' components (Consistent Vertical Scale)'
        plt.suptitle(plot_title, x=2.5, y=8.5, fontsize=50)
        plt.subplots_adjust(left=0, bottom=0, right=5, top=8)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

#############################################################################################
#############################################################################################
#############################################################################################

    def plot_hist_of_spatial_distrib_of_dim_reduced_intensities_for_individual_cph_and_naive(self):
        ### Histogram of spatial distribution of NMF/PCA component intensities for individual cph and naive Data

        num_bins = 100
        remove_zero_bin = 1
        lower_quantile = 0
        upper_quantile = 1
        colors = ['red', 'blue']  ## Red is CPH. Blue is Naive


        dim_reduced_dict_recovered = self.dim_reduced_dict_recovered
        dim_reduced_outputs_recovered = dim_reduced_dict_recovered['dim_reduced_outputs']

        dim_reduced_dataset = dim_reduced_outputs_recovered[0][0]
        pixel_count_array = dim_reduced_dict_recovered['pixel_count_array']

        pix_count_previous = 0

        ## Break the combined NMF data for all datasets into individual datasets
        data_store = []
        for i in range(len(pixel_count_array)):
            pix_count = int(pixel_count_array[i])
            individual_dim_reduced_data = dim_reduced_dataset[pix_count_previous: pix_count_previous + pix_count, :]
            pix_count_previous = pix_count_previous + pix_count
            data_store.append(individual_dim_reduced_data)

        num_components = data_store[0].shape[1]
        num_datasets = len(data_store)

        ## Obtain the maximum and minimum pixel intensity for each component of each dataset
        minimum_intensity_store = []
        maximum_intensity_store = []
        for i in range(num_datasets):
            min_row = []
            max_row = []
            for j in range(num_components):
                #         min_row.append(np.min(data_store[i][:,j]))
                #         max_row.append(np.max(data_store[i][:,j]))
                min_row.append(np.quantile(data_store[i][:, j], lower_quantile))
                max_row.append(np.quantile(data_store[i][:, j], upper_quantile))

            minimum_intensity_store.append(min_row)
            maximum_intensity_store.append(max_row)

        ## Obtain the maximum and minimum across datasets for each component
        minimum_intensity_array = np.min(np.array(minimum_intensity_store), axis=0)
        maximum_intensity_array = np.max(np.array(maximum_intensity_store), axis=0)

        ## Obtain the histogram intensities and bin edges
        hist_store = []
        for i in range(num_datasets):

            hist_row = []
            for j in range(num_components):
                pixel_count_for_this_dataset = pixel_count_array[i]
                hist, bin_edges = np.histogram(data_store[i][:, j],
                                               bins=np.linspace(minimum_intensity_array[j], maximum_intensity_array[j],
                                                                num_bins))

                ##############
                ## Artificially set the histogram count of the bin centered at zero to zero.
                if remove_zero_bin == 1 and self.global_plots_used_dim_reduction_technique == 'nmf':
                    hist[0] = 0
                ###############
                hist_row.append([hist / pixel_count_for_this_dataset, bin_edges])
            #         hist_row.append([hist, bin_edges])

            hist_store.append(hist_row)

        ## Obtain the mean histogram for CPH data
        cph_hist_store = []
        for i in range(0, 4):
            cph_row = []
            for j in range(num_components):
                cph_row.append(hist_store[i][j][0])

            cph_hist_store.append(cph_row)

        mean_cph = np.mean(cph_hist_store, axis=0)

        ## Obtain the mean histogram for Naive data
        naive_hist_store = []
        for i in range(4, 8):
            naive_row = []
            for j in range(num_components):
                naive_row.append(hist_store[i][j][0])

            naive_hist_store.append(naive_row)

        mean_naive = np.mean(naive_hist_store, axis=0)

        ## Store the mean CPH and mean Naive histogram data in a single array
        hist_store_separated = [mean_cph, mean_naive]

        ## Plot the histograms
        fig, ax = plt.subplots(3, num_components, gridspec_kw={'height_ratios': [1, 1, 1]})

        for j in range(num_components):
            for i in range(2):
                bin_edges = hist_store[0][j][1]
                hist_vals = hist_store_separated[i][j]
                ax[i, j].bar(bin_edges[:-1], hist_vals, width=bin_edges[0] - bin_edges[1], color=colors[i], alpha=0.7)

                ax[i, j].xaxis.set_tick_params(length=10, width=5, labelsize=50, rotation=0)
                ax[i, j].yaxis.set_tick_params(length=10, width=5, labelsize=50)
                ax[i, j].set_title(self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=70)
                ax[i, j].spines[['right', 'top']].set_visible(0)

                ax[i, 0].set_ylabel(['CPH', 'Naive'][i], fontsize=70)

        ## Plot the mean histograms on top of each other

        for j in range(num_components):
            for i in range(2):
                bin_edges = hist_store[0][j][1]
                hist_vals = hist_store_separated[i][j]
                ax[2, j].bar(bin_edges[:-1], hist_vals, width=bin_edges[0] - bin_edges[1], color=colors[i], alpha=0.7)

                ax[2, j].xaxis.set_tick_params(length=10, width=5, labelsize=50, rotation=0)
                ax[2, j].yaxis.set_tick_params(length=10, width=5, labelsize=50)
                ax[2, j].set_title(self.global_plots_used_dim_reduction_technique + ' ' + str(j + 1), fontsize=70)
                ax[2, 0].set_ylabel('Overlayed\nCPH - Red, Naive - Blue', fontsize=70)
                ax[2, j].spines[['right', 'top']].set_visible(0)

        plot_title = 'Histogram of Spatial intensities for each ' + self.global_plots_used_dim_reduction_technique + ' component for Naive and CPH datasets separately'
        plt.suptitle(plot_title, x=12.5, y=10.5, fontsize=80)
        plt.subplots_adjust(left=0, bottom=0, right=25, top=10)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat,
                        dpi=self.global_plots_dpi, bbox_inches='tight')


#############################################################################################
#############################################################################################
#############################################################################################

    def plot_he_stains_basis_reconstruction(self):
        ### HE stains basis reconstruction


        dim_reduced_outputs_recovered = self.dim_reduced_dict_recovered['dim_reduced_outputs']

        he_file_names = self.dim_reduced_object.dataloader_kp_object.he_filename_array

        dim_reduced_dataset = dim_reduced_outputs_recovered[0][0]

        combined_data_object = self.dim_reduced_object.dataloader_kp_object.combined_data_object
        dataset_order = self.dim_reduced_object.dataloader_kp_object.dataset_order

        pix_count_previous = 0
        dataset_store = []
        he_store = []
        he_store_unflattened = []
        coeff_store = []
        nmf_store = []
        for i in range(len(combined_data_object)):
            num_rows = combined_data_object[i].rows
            num_cols = combined_data_object[i].cols
            coordinate_array = combined_data_object[i].coordinates
            pix_count = len(combined_data_object[i].coordinates)
            individual_dim_reduced_data = dim_reduced_dataset[pix_count_previous: pix_count_previous + pix_count, :]

            pix_count_previous = pix_count_previous + pix_count
            num_dim_reduced_components = individual_dim_reduced_data.shape[1]

            # this_he_image = Image.open(eval(dataset_order[i] + '_he_stain')).resize((num_cols, num_rows), Image.LANCZOS)
            this_he_image = Image.open(he_file_names[0][i]).resize((num_cols, num_rows), Image.LANCZOS)
            this_he_r_channel = np.array([np.array(this_he_image.getchannel(0)).flatten()]).T
            this_he_g_channel = np.array([np.array(this_he_image.getchannel(1)).flatten()]).T
            this_he_b_channel = np.array([np.array(this_he_image.getchannel(2)).flatten()]).T

            this_he_r_channel_unflattened = np.array(this_he_image.getchannel(0)).T
            this_he_g_channel_unflattened = np.array(this_he_image.getchannel(1)).T
            this_he_b_channel_unflattened = np.array(this_he_image.getchannel(2)).T

            he_store_unflattened.append(np.array([this_he_r_channel_unflattened, this_he_g_channel_unflattened, this_he_b_channel_unflattened]).T)
            he_store.append(np.array([this_he_r_channel, this_he_g_channel, this_he_b_channel]).T.squeeze())

            datagrid = np.zeros([num_dim_reduced_components, num_rows, num_cols])

            for temp1 in range(num_dim_reduced_components):
                for temp2 in range(pix_count):
                    this_col = coordinate_array[temp2][0]
                    this_row = coordinate_array[temp2][1]
                    datagrid[temp1, this_row - 1, this_col - 1] = individual_dim_reduced_data[temp2, temp1]

            this_nmf_set = datagrid.reshape(20, -1).T
            nmf_store.append(this_nmf_set)

            for j in range(3):
                r_channel_coeff = np.dot(np.linalg.inv(np.dot(this_nmf_set.T, this_nmf_set)),
                                         np.dot(this_nmf_set.T, this_he_r_channel))
                g_channel_coeff = np.dot(np.linalg.inv(np.dot(this_nmf_set.T, this_nmf_set)),
                                         np.dot(this_nmf_set.T, this_he_g_channel))
                b_channel_coeff = np.dot(np.linalg.inv(np.dot(this_nmf_set.T, this_nmf_set)),
                                         np.dot(this_nmf_set.T, this_he_b_channel))

            coeff_store.append(np.array([r_channel_coeff, g_channel_coeff, b_channel_coeff]).T)

        coeff_store = np.array(coeff_store)

        ### Plotting part

        datagrid_store_recovered = self.global_plots_datagrid_store_dict_recovered['datagrid_store']

        all_data = datagrid_store_recovered
        for dataset in all_data:
            dataset[dataset == -1] = 0

        ### The correlation array goes here

        corr_arr = coeff_store.squeeze()

        ### Do the reconstruction for red
        red_output_arr = []
        for dataset_count, dataset in enumerate(all_data):
            this_dat = np.dot(dataset.T, corr_arr[dataset_count][:, 0])
            red_output_arr.append(((this_dat - np.min(this_dat)) / (np.max(this_dat) - np.min(this_dat))) * 255)

        ### Do the reconstruction for green
        green_output_arr = []
        for dataset_count, dataset in enumerate(all_data):
            this_dat = np.dot(dataset.T, corr_arr[dataset_count][:, 1])
            green_output_arr.append(((this_dat - np.min(this_dat)) / (np.max(this_dat) - np.min(this_dat))) * 255)

        ### Do the reconstruction for blue
        blue_output_arr = []
        for dataset_count, dataset in enumerate(all_data):
            this_dat = np.dot(dataset.T, corr_arr[dataset_count][:, 2])
            blue_output_arr.append(((this_dat - np.min(this_dat)) / (np.max(this_dat) - np.min(this_dat))) * 255)

        ### Plot the reconstructed images for each dataset
        fig, ax = plt.subplots(4, len(red_output_arr))
        for i in range(len(red_output_arr)):
            rgb_image = np.uint8(np.array([red_output_arr[i], green_output_arr[i], blue_output_arr[i]]).T)
            recon_img = Image.fromarray(rgb_image)
            ax[0, i].xaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[0, i].yaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[0, i].set_title('Reconstructed\nH&E stained\n' + dataset_order[i], size=30)
            ax[0, i].imshow(recon_img)

            he_orig_img = Image.fromarray(he_store_unflattened[i])
            ax[1, i].xaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[1, i].yaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[1, i].set_title('Original \nH&E stained\n' + dataset_order[i], size=30)
            ax[1, i].imshow(he_orig_img)

            blended_he_and_recon_img = Image.blend(he_orig_img, recon_img, 0.5)
            ax[2, i].xaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[2, i].yaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[2, i].set_title('Blended\nReconstructed &\nH&E stained\n' + dataset_order[i], size=30)
            ax[2, i].imshow(blended_he_and_recon_img)

            nmf_component_1 = all_data[i][0]
            ax[3, i].xaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[3, i].yaxis.set_tick_params(length=0, width=0, labelsize=0)
            ax[3, i].set_title('NMF1\n' + dataset_order[i], size=30)
            ax[3, i].imshow(nmf_component_1)

        plot_title = 'H&E stained images reconstructed with ' + self.global_plots_used_dim_reduction_technique + '  images as basis'
        plt.suptitle(plot_title, x=4, y=8, fontsize=80)
        plt.subplots_adjust(left=0, bottom=0, right=8, top=7)
        plt.show()

        if self.save_figures == 1:
            fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat,
                        dpi=self.global_plots_dpi - 100, bbox_inches='tight')

        print('\n')

        ####################################################

