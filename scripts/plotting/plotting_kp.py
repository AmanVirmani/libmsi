from sklearnex import patch_sklearn

patch_sklearn()

import sys

import libmsi_kp
import numpy as np
import random as rd
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patheffects as path_effects

from mpl_toolkits import mplot3d

from itertools import combinations

from skimage import color

import pandas as pd
import pickle

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

# import cuml.manifold.t_sne as cuml_TSNE

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from scipy.signal import find_peaks

from PIL import Image

import cv2

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

import os

import pickle
from data_preformatter_kp import data_preformatter_kp
from nmf_kp import nmf_kp
from pca_kp import pca_kp

rd.seed(0)


class plotting_kp:

    def __init__(self, svm_kp_object=None, special_plotting_without_svm_kp_object=0, saved_segregated_data_filename=None, saved_3d_datagrid_filename=None, plots_dpi=600, save_figures=0, plots_fileformat='svg', figure_saving_folder=''):

        """
        @brief Plotting
                Usage example:
                                svm_kp_object = svm_kp_object
                                saved_segregated_data_filename = None
                                saved_3d_datagrid_filename = None
                                plots_dpi = 600
                                save_figures = 1
                                plots_fileformat = 'svg'
                                figure_saving_folder = double_compact_msi_data.global_path_name + '/saved_outputs/figures/'

                                special_plotting_without_svm_kp_object = 0  # This is a flag used to indicate that we are not supplying svm_kp objects. Instead, we plainly initialize the plotting class, but directly supply the required data into the method associated

                                plotting_kp_object = plotting_kp(svm_kp_object = svm_kp_object, special_plotting_without_svm_kp_object=special_plotting_without_svm_kp_object, saved_segregated_data_filename=saved_segregated_data_filename, saved_3d_datagrid_filename=saved_3d_datagrid_filename, plots_dpi=plots_dpi, save_figures=save_figures, plots_fileformat=plots_fileformat, figure_saving_folder=figure_saving_folder)


        @param svm_kp_object
        @param special_plotting_without_svm_kp_object This is an important parameter. If this is supplied, an svm_kp_object needs not be supplied. All data required for plotting will have to be given directly to a method in the plotting_kp class.
        @param saved_segregated_data_filename
        @param saved_3d_datagrid_filename
        @param plots_dpi
        @param save_figures
        @param plots_fileformat
        @param figure_saving_folder
        """

        self.save_figures = save_figures
        self.global_plots_fileformat = plots_fileformat
        self.global_plots_dpi = plots_dpi
        self.special_plotting_without_svm_kp_object = special_plotting_without_svm_kp_object

        v = cm.get_cmap('Blues', 512)
        v_map = v(np.linspace(0, 1, 512))  ### Take 256 colors from the 'Greens' colormap, and distribute it between 0 and 1.
        r = cm.get_cmap('Reds_r', 512)
        r_map = r(np.linspace(0, 1, 512))
        new = np.append(r_map, v_map, axis=0)
        new[int(new.shape[0] * 0.5 - 1):int(new.shape[0] * 0.5 + 1), :] = [1, 1, 1, 1]
        self.global_plots_new_map = ListedColormap(new)

        if (svm_kp_object is None) or (svm_kp_object == '') or (special_plotting_without_svm_kp_object == 1):
            # Follow a different protocol in this case.
            print("svm_kp_object will not be used. Please provide data directly into the relevant method")
        else:
            print("svm_kp_object WILL be used. Data provided directly into a method may be ignored")
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

            self.global_plots_my_linear_svc = svm_kp_object.one_time_svc_results
            self.create_mean_combined_arrays()
            self.min_mz_after_truncation = self.svm_kp_object.data_preformatter_object.dim_reduced_object.dataloader_kp_object.min_mz_after_truncation
            self.max_mz_after_truncation = self.svm_kp_object.data_preformatter_object.dim_reduced_object.dataloader_kp_object.max_mz_after_truncation

    def plot_formatting_example(self):

        #############################################
        ## Axis specific 1: Title formatting

        """
        right_max = 5
        top_max = 2
        font_scale_factor = np.min((right_max, top_max))

        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)
        ax[0, 0].set_title("my title", fontsize=10 * font_scale_factor, rotation=20, color='g', loc='left', fontstyle='italic')
        ax[0, 1].set_title("my title", fontsize=10 * font_scale_factor, rotation=105, color='g', loc='right', fontstyle='oblique')
        ax[0, 2].set_title("my title", fontsize=10 * font_scale_factor, rotation=-10, color='g', loc='center', fontweight=1000)
        ax[0, 3].set_title("my title", fontsize=10 * font_scale_factor, rotation=0, color='g', fontweight=1000, position=[-0.1, 100])  ## The y argument of the position parameter is not relavent when working on a ylabel
        ax[0, 4].set_title("my title", fontsize=10 * font_scale_factor, rotation=10, color='g', fontweight=1000, position=[1.2, 1000])  ## The y argument of the position parameter is not relavent when working on a ylabel

        ax[1, 0].set_title("my title", fontsize=10 * font_scale_factor, color='g', fontweight=1000, pad=100)  ## The y position of a title is changed using the pad parameter
        ax[1, 1].set_title("my title", fontsize=10 * font_scale_factor, color='g', fontweight=1000, pad=0)  ## The y position of a title is changed using the pad parameter
        ax[1, 2].set_title("my title", fontsize=10 * font_scale_factor, color='g', fontweight=1000, pad=-20)  ## However, setting negative values will NOT take the title inside the axes.
        ax[1, 3].set_title("my title", fontsize=10 * font_scale_factor, color='g', fontweight=1000, y=0.5)  ## By setting the 'y' parameter you can move the title to the desired y position with the bottom left of the axis taken as origin.

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        #############################################
        ## Axis specific 2: Xlabel formatting

        """
        right_max = 5
        top_max = 2
        font_scale_factor = np.min((right_max, top_max))

        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)
        ax[0, 0].set_xlabel("my x label", fontsize=10 * font_scale_factor, rotation=0, color='k', loc='left', fontstyle='italic')
        ax[0, 1].set_xlabel("my x label", fontsize=10 * font_scale_factor, rotation=10, color='k', loc='right', fontstyle='oblique')
        ax[0, 2].set_xlabel("my x label", fontsize=10 * font_scale_factor, rotation=5, color='k', loc='center', fontweight=1000)
        ax[0, 3].set_xlabel("my x label", fontsize=10 * font_scale_factor, rotation=-15, color='k', fontweight=100, position=[0, 100])  ## The y argument of the position parameter is not relavent when working on a ylabel
        ax[0, 4].set_xlabel("my x label", fontsize=10 * font_scale_factor, rotation=0, color='k', fontweight=1000, position=[1.2, 1000])  ## The y argument of the position parameter is not relavent when working on a ylabel

        ax[1, 0].set_xlabel("my x label", fontsize=10 * font_scale_factor, color='k', fontweight=1000, labelpad=-100)  ## The x position of a y label is changed using the labelpad parameter
        ax[1, 1].set_xlabel("my x label", fontsize=10 * font_scale_factor, color='k', fontweight=1000, labelpad=0)  ## The x position of a y label is changed using the labelpad parameter
        ax[1, 2].set_xlabel("my x label", fontsize=10 * font_scale_factor, color='k', fontweight=1000, labelpad=20)  ## The x position of a y label is changed using the labelpad parameter

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        #############################################
        ## Axis specific 3: Ylabel formatting

        """
        right_max = 5
        top_max = 2
        font_scale_factor = np.min((right_max, top_max))

        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)
        ax[0, 0].set_ylabel("my y label", fontsize=10 * font_scale_factor, rotation=70, color='r', loc='bottom', fontstyle='italic')
        ax[0, 1].set_ylabel("my y label", fontsize=10 * font_scale_factor, rotation=105, color='r', loc='top', fontstyle='oblique')
        ax[0, 2].set_ylabel("my y label", fontsize=10 * font_scale_factor, rotation=105, color='r', loc='center', fontweight=1000)
        ax[0, 3].set_ylabel("my y label", fontsize=10 * font_scale_factor, rotation=105, color='r', fontweight=100, position=[0, 1.2])  ## The x argument of the position parameter is not relavent when working on a ylabel
        ax[0, 4].set_ylabel("my y label", fontsize=10 * font_scale_factor, rotation=105, color='r', fontweight=1000, position=[100, 1.2])  ## The x argument of the position parameter is not relavent when working on a ylabel

        ax[1, 0].set_ylabel("my y label", fontsize=10 * font_scale_factor, color='r', fontweight=1000, labelpad=-100)  ## The x position of a y label is changed using the labelpad parameter
        ax[1, 1].set_ylabel("my y label", fontsize=10 * font_scale_factor, color='r', fontweight=1000, labelpad=0)  ## The x position of a y label is changed using the labelpad parameter
        ax[1, 2].set_ylabel("my y label", fontsize=10 * font_scale_factor, color='r', fontweight=1000, labelpad=20)  ## The x position of a y label is changed using the labelpad parameter

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        ######################################################
        #### Axis specific 4: ticks, ticklabels, and axis limits formatting

        """
        right_max = 5
        top_max = 4
        font_scale_factor = np.min((right_max, top_max))

        num_subplot_rows = 3
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        ax[0, 0].set(xlim=[-10, 10], ylim=[-15, 25])
        ax[0, 0].text(-7, 5, "Just a raw figure with \n only x and y lims set", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[0, 1].set(xticks=[1, 9, 50, 60, 120], yticks=[5, 9, 15, 22, 100])
        ax[0, 1].text(5, 50, "Custom x and y ticks set", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[0, 2].set(xticks=[1, 9, 50, 60, 120], yticks=[5, 9, 15, 22, 100], xticklabels=['a', 'x', '5', 'k', 'pp'], yticklabels=['ya', 'dsw', 'lpw', 'wk', 'dw'])
        ax[0, 2].text(5, 50, "labels have been added to ticks", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[0, 3].set(xticks=[1, 9, 50, 60, 120], yticks=[5, 9, 15, 22, 100], xticklabels=['a', 'x', '5', 'k', 'pp'], yticklabels=['ya', 'dsw', 'lpw', 'wk', 'dw'], xlim=[-10, 10], ylim=[-15, 25])
        ax[0, 3].text(-7, 5, "previous image but \n with x and y lims set", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[0, 4].set(xticks=[1, 9, 50, 60, 120], yticks=[5, 9, 15, 22, 100], xlim=[-10, 10], ylim=[-15, 25])
        ax[0, 4].text(-7, 5, "Shows what happens if\n you set ticks but \n not tick labels\n and then set x and y lims", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ##

        ax[1, 0].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[1, 0].tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3, rotation=50, labelcolor='r', color='b')
        ax[1, 0].text(7, 100, "Font size, rotation, and color \n of major tick labels on BOTH x and y axes\n have been changed. \n length, width, and color of major ticks \n on BOTH x and y axes\n have changed", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[1, 1].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[1, 1].tick_params(which='major', labelsize=0 * font_scale_factor, size=20)
        ax[1, 1].text(7, 100, "major Tick labels have vanished,\n and major tick sizes have changed.", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[1, 2].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[1, 2].tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3)
        ax[1, 2].minorticks_on()
        ax[1, 2].tick_params(which='minor', labelsize=5 * font_scale_factor, length=10, width=1.5)
        ax[1, 2].text(7, 100, "Font size and dimensions\n of major ticks and tick labels,\n have changed.\n Minor ticks have been made visible\n and their dimensions have\n been adjusted independent from\n major tick parameters\nfor BOTH x and y axes.", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[1, 3].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[1, 3].tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3, direction='in')
        ax[1, 3].minorticks_on()
        ax[1, 3].tick_params(which='minor', labelsize=5 * font_scale_factor, length=10, width=1.5)
        ax[1, 3].text(7, 100, "Major ticks of BOTH x and y axes\n have been sent to the inside\n of the plot while minor ticks\n remain outside", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[1, 4].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[1, 4].tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3, direction='inout', color='r', labelcolor='g')
        ax[1, 4].minorticks_on()
        ax[1, 4].tick_params(which='minor', labelsize=5 * font_scale_factor, length=10, width=1.5, direction='in', color='g')
        ax[1, 4].text(7, 100, "Major ticks of BOTH x and y axes\n are made visible BOTH inside and outside\n of the plot.\n Minor ticks of BOTH x and y axes\n have been sent to the inside\of the plot.\n colors have been independently added\n to major tick labels, major ticks, and\n minor ticks of BOTH x and y axes", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ##

        ax[2, 0].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[2, 0].tick_params(which='both', labelsize=8 * font_scale_factor, length=20, width=3, rotation=50, labelcolor='r', color='b', right=True, left=False, bottom=False, labelright=True, labelleft=False)  ## Set formatting parameters for the both major and minor ticks and tick labels on both x and y axes
        ax[2, 0].text(7, 100, "BOTH MAJOR and MINOR ticks\n have been processed\n simultaneously.\n ticks at the bottom are made \n invisible but tick labels are visible.\n BOTH ticks and tick labels at the left\n have also been made invisible.\n BOTH ticks and tick labels\n have been made visible\n on the right edge", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[2, 1].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[2, 1].tick_params(which='both', labelsize=0 * font_scale_factor, size=20, top=True, left=False)  ## Set formatting parameters for the both major and minor ticks and tick labels on both x and y axes
        ax[2, 1].text(7, 100, "Tick labels have been hidden.\n Ticks on the left edge\n have been hidden.\n Ticks have been added to the\n top edge", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[2, 2].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[2, 2].xaxis.set_tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3, color='g', labelcolor='c')  ## Set formatting parameters for the major ticks and tick labels ONLY on x axis
        ax[2, 2].xaxis.set_ticks([1, 19, 33, 83], minor=True)  ## Set the tick values for minor ticks ONLY on x axis.
        ax[2, 2].xaxis.set_ticklabels(['k', 's', 'w', 'p'], minor=True)  ## Set the labels for minor ticks  ONLY on x axis.
        ax[2, 2].xaxis.set_tick_params(which='minor', labelsize=5 * font_scale_factor, length=10, width=1.5, rotation=30, color='r', labelcolor='b')  ## Set formatting parameters for the minor ticks and tick labels ONLY on x axis
        ax[2, 2].text(7, 100, "Here, ONLY the parameters of\n XAXIS have been modified.\n As before, formatting has been done\n to major ticks of x axis.\n MINOR ticks and MINOR tick labels\n have also been independently added\n and formatted ONLY to the x axis.", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[2, 3].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[2, 3].xaxis.set_tick_params(which='major', labelsize=8 * font_scale_factor, length=20, width=3, direction='in', pad=50)  ## Set formatting parameters for the major ticks and tick labels ONLY on x axis
        ax[2, 3].minorticks_on()
        ax[2, 3].tick_params(which='minor', labelsize=5 * font_scale_factor, length=10, width=1.5)
        ax[2, 3].text(7, 100, "Again, ONLY x axis is modified. \n Here, the gap between \n MAJOR ticks and MAJOR tick labels\n have been increased ONLY for x axis", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        ax[2, 4].set(xticks=[1, 50, 100, 150, 200], yticks=[5, 55, 105, 155, 205], xticklabels=['a', 'b', 'c', 'd', 'e'], yticklabels=['m', 'n', 'o', 'p', 'q'])
        ax[2, 4].tick_params(which='major', labelsize=10 * font_scale_factor, length=20, width=3, direction='inout', color='r', labelcolor='g')  ## Set formatting parameters for the major ticks and tick labels on both x and y axes
        ax[2, 4].minorticks_on()
        ax[2, 4].yaxis.set_tick_params(which='minor', labelsize=8 * font_scale_factor, length=10, width=1.5, direction='in', color='g')  ## Set formatting parameters for the minor ticks and tick labels ONLY on Y axis
        ax[2, 4].text(7, 100, "Major ticks and tick labels\n have been changed on\n BOTH x and y axes like before.\n MINOR ticks on Y AXIS ONLY have\n been independently formatted", fontsize=4 * font_scale_factor)  ## The text position is in data coordinates by default. Can be changed though

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        #############################################
        #### Axis specific 5: Axis spines

        """

        right_max = 7
        top_max = 5
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(0, 2 * 3.14, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        ax[0, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 0].set_title("Just a plain plot with no spine formatting")

        ax[0, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 1].set_frame_on(False)  ## Turn off all spines in one go
        ax[0, 1].set_title("Turn off all spines in one go")

        ax[0, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 2].spines[['left', 'right']].set_visible(False)
        ax[0, 2].set_title("Turn off only left and right spines")

        ax[0, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 3].spines['left'].set(linewidth=8, color='g')
        ax[0, 3].set_title("Change line thickness and color of left spine")

        ax[0, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 4].spines['right'].set(linewidth=5, color='g', linestyle='-.')
        ax[0, 4].set_title("Change linestyle of right spine")

        ##

        ax[1, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 0].spines['left'].set(linewidth=5, color='y', bounds=[-1, 0])  ## Use data coordinates to set the bounds
        ax[1, 0].spines['bottom'].set(linewidth=5, color='m', bounds=[0.5, 5])  ## Use data coordinates to set the bounds
        ax[1, 0].set_title("Change length of the bottom and left spines")

        ax[1, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 1].spines['right'].set(linewidth=5, color='c', position=('outward', -50))
        ax[1, 1].set_title("Change the position of the right spine by moving it more to the left")

        ax[1, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 2].spines['left'].set(linewidth=5, color='k', position=('outward', -100))
        ax[1, 2].set_title("Change the position of the left spine by moving it more to the right")

        ax[1, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 3].spines['right'].set(linewidth=5, color='g', linestyle='-.')
        ax[1, 3].spines['left'].set(linewidth=4, position='center')
        ax[1, 3].spines['bottom'].set(linewidth=4, position='center')
        ax[1, 3].set_title("Set the left and bottom spines to center of plot")

        ax[1, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 4].spines['left'].set(linewidth=4, position=('data', 1.2))
        ax[1, 4].spines['bottom'].set(linewidth=4, position=('data', 0.5))
        ax[1, 4].set_title("Easily set the left and bottom spines to data coordinates (1.2, 0.5) of plot")

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        #############################################
        #### Axis specific 6: Miscellaneous
        # Axis positions in the figure
        # Axis aspect rations
        # Axis background colors
        # Secondary axes and transformations
        # Inverting axes

        """

        right_max = 7
        top_max = 5
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(0, 2 * 3.14, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 3
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        ax[0, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')

        ax[1, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')

        ax[2, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[2, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[2, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[2, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[2, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')

        plt.subplots_adjust(0, 0, right_max, top_max)

        ax[0, 0].set_title("Just a plain plot with no spine formatting")

        ax[0, 1].set(position=[(1 * right_max / num_subplot_cols), (1.2 * top_max), 1, 1])
        ax[0, 1].set_title("Set the position of the figure to x=(1 * right_max/num_subplot_cols), y= (1.2 * top_max)  \n and set the width and length to (1,1)")

        ax[0, 2].set(position=[(2 * right_max / num_subplot_cols), (1.2 * top_max), 1, 3])
        ax[0, 2].set_title("Set the position of the figure to x=(2 * right_max/num_subplot_cols), y= (1.2 * top_max)  \n and set the width and length to (1,3)")

        ax[0, 3].set(aspect='equal')
        ax[0, 3].set_title("Set the aspect ratio of the plot to be equal\n so that both x and y tick gaps are the same visually")

        ax[0, 4].set(aspect=2.5)
        ax[0, 4].set_title("Set the aspect ratio to a custom value (2.5)")

        ##

        ax[1, 0].set(xlim=[-3, 3], ylim=[-3, 3])
        ax[1, 0].set_title("Changed the axis limits. Observe how the x and y lims are the same \n but the gaps between ticks are different due to non-unity aspect ratio")

        ax[1, 1].set(xlim=[-3, 3], ylim=[-3, 3], aspect='equal')
        ax[1, 1].set_title("Observe how the x and y lims are the same \n and now the gaps between ticks are also the same after setting aspect ratio to unity")

        ax[1, 2].set(xlim=[-3, 3], ylim=[-3, 3], aspect='equal')
        ax[1, 2].invert_xaxis()
        ax[1, 2].set_title("x axis has been inverted to get a mirrored plot about y axis. \n However note that the x axis itself has also inverted")

        ax[1, 3].set(xlim=[-3, 3], ylim=[-3, 3], aspect='equal')
        ax[1, 3].invert_yaxis()
        ax[1, 3].set_title("y axis has been inverted to get a mirrored plot about x axis.\n However note that the y axis itself has also inverted")

        ax[1, 4].set(xlim=[-3, 3], ylim=[-3, 3], aspect='equal', facecolor='y')
        ax[1, 4].set_title("Set a color to the background of the axis")

        ##

        ax[2, 0].set(aspect='equal')
        ax[2, 0].secondary_xaxis(location='top', functions=(lambda x: (x + 2) / 3 + 5, lambda x: 3 * (x - 5) - 2))  ## In 'functions' parameter, both the desired transformation, and its inverse transformation must be given, either as anonymous lambda functions or real functions
        ax[2, 0].set_title("Set a secondary x axis with a transformation top_x = (x + 2)/3 + 5. \n Its inverse transformation must also be given. Both functions given as lambda functions ")

        ax[2, 1].set(aspect='equal')
        ax[2, 1].secondary_yaxis(location='right', functions=(lambda x: 2 * x, lambda y: y / 2))  ## In 'functions' parameter, both the desired transformation, and its inverse transformation must be given, either as anonymous lambda functions or real functions
        ax[2, 1].set_title("Set a secondary y axis with a transformation right_y = 2 * y. \n Its inverse transformation must also be given. Both functions given as lambda functions ")

        def my_transformation(x):
            return 3 * (x + 5)

        def my_inverse_transformation(y):
            return (y / 3) - 5

        ax[2, 2].set(aspect='equal')
        ax[2, 2].secondary_yaxis(location='right', functions=(my_transformation, my_inverse_transformation))  ## In 'functions' parameter, both the desired transformation, and its inverse transformation must be given, either as anonymous lambda functions or real functions
        ax[2, 2].set_title("Set a secondary y axis with a transformation right_y = 3 * (x + 5). \n Its inverse transformation must also be given. Both functions were defined as \n CONVENTIONAL functions")

        ax[2, 3].set_xmargin(2)
        ax[2, 3].set_title("Set x margin to 2")

        ax[2, 4].set(xmargin=7)
        ax[2, 4].set_title("Set x margin to 7")

        plt.show()
    
        """

        #############################################
        ## plt.plot() specific
        # Formatting along with data point highlighting.
        # Vertical and horizontal cursors.
        # Adding custom colors.
        # Changing transparency of plots.

        """

        right_max = 7
        top_max = 5
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(0, 2 * 3.14, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 3
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        ax[0, 0].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 0].set_title("plain plot without any markers or linestyles")

        ax[0, 1].plot(x_data_for_plot, y_data_for_plot, color='y', linestyle='-', linewidth=5)
        ax[0, 1].set_title("Thicker and colored line for sinusoid")

        ax[0, 2].plot(x_data_for_plot, y_data_for_plot, color='r', linestyle='--', linewidth=5)
        ax[0, 2].set_title("Dashed line for sinusoid")

        ax[0, 3].plot(x_data_for_plot, y_data_for_plot, color=(0.1, 0.56, 0.33), linestyle='-.', linewidth=5)
        ax[0, 3].set_title("Dash and dot line with a custom color setting using (0.1, 0.56, 0.33) RGB values")

        ax[0, 4].plot(x_data_for_plot, y_data_for_plot, color=(0.8, 0.3, 0.3), linestyle=':', linewidth=5)
        ax[0, 4].set_title("Dotted line along with a custom color setting using (0.8,0.3,0.3) RGB values")

        ##

        ax[1, 0].plot(x_data_for_plot, y_data_for_plot, color='g', linestyle='-', linewidth=9, marker='*', markersize=10, markerfacecolor='y', markeredgecolor='m', markeredgewidth=0.5)
        ax[1, 0].set_title("Thick green solid line with star markers on it\n with yellow fill and magenta outline")

        ax[1, 1].plot(x_data_for_plot, y_data_for_plot, color='y', linestyle='-', linewidth=5)
        ax[1, 1].plot(3.14, np.sin(3.14), color='g', marker='d', markersize=20, markeredgecolor='r')
        ax[1, 1].plot(3.14 / 2, np.sin(3.14 / 2), color='r', marker='o', markersize=20, markeredgecolor='c')
        ax[1, 1].set_title("Styled line for sinusoid.Added a green diamond marker \n with red outline at x = pi, and a red circle marker \n with cyan outline at x = pi/2")

        ax[1, 2].plot(x_data_for_plot, y_data_for_plot, color='y', linestyle='-', linewidth=5)
        ax[1, 2].plot(3.14 / 2, 2 * np.sin(3.14 / 2), color='g', marker='H', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 2].plot(3.14, np.sin(3.14), color='g', marker='h', markersize=20, markeredgecolor='y', markeredgewidth=5)
        ax[1, 2].set_title("Styled line for sinusoid. Added a green hexagon marker with thicker red outline at x = pi/2,\n but outside the line plot, and a slanted hexagon marker \n with orange outline at x = pi")

        ax[1, 3].plot(x_data_for_plot, y_data_for_plot, color='y', linestyle='-', linewidth=5)
        ax[1, 3].plot(0.3, 0.5, color='g', marker='_', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 3].plot(0.8, 0.5, color='g', marker='|', markersize=20, markeredgecolor='c', markeredgewidth=5)
        ax[1, 3].plot(1.3, 0.5, color='g', marker='X', markersize=20, markeredgecolor='m', markeredgewidth=5)
        ax[1, 3].plot(1.8, 0.5, color='w', marker='X', markersize=20, markeredgecolor='y', markeredgewidth=5)
        ax[1, 3].plot(2.3, 0.5, color='g', marker='x', markersize=20, markeredgecolor='k', markeredgewidth=5)
        ax[1, 3].plot(2.8, 0.5, color='g', marker='D', markersize=20, markeredgecolor='b', markeredgewidth=5)
        ax[1, 3].plot(3.3, 0.5, color='g', marker='d', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 3].plot(3.8, 0.5, color='g', marker='H', markersize=20, markeredgecolor='c', markeredgewidth=5)
        ax[1, 3].plot(4.3, 0.5, color='g', marker='h', markersize=20, markeredgecolor='m', markeredgewidth=5)
        ax[1, 3].plot(4.8, 0.5, color='g', marker='>', markersize=20, markeredgecolor='y', markeredgewidth=5)
        ax[1, 3].plot(5.3, 0.5, color='g', marker='<', markersize=20, markeredgecolor='k', markeredgewidth=5)
        ax[1, 3].plot(5.8, 0.5, color='g', marker='v', markersize=20, markeredgecolor='b', markeredgewidth=5)
        ax[1, 3].plot(6.3, 0.5, color='g', marker='^', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 3].set_title("Trying different markers in the following order:\n unfillable hline, unfillable vline, filled X, filled X with white filling, unfillable X, \n filled diamond, filled long diamond, filled hexagon, filled turned hexagon, \n filled right triangle, filled left triangle, filled down triangle, filled up triangle")

        ax[1, 4].plot(x_data_for_plot, y_data_for_plot, color='y', linestyle='-', linewidth=5)
        ax[1, 4].plot(0.3, 0.5, color='g', marker='.', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 4].plot(1.3, 0.5, color='g', marker='1', markersize=20, markeredgecolor='m', markeredgewidth=5)
        ax[1, 4].plot(1.8, 0.5, color='w', marker='2', markersize=20, markeredgecolor='y', markeredgewidth=5)
        ax[1, 4].plot(2.3, 0.5, color='g', marker='3', markersize=20, markeredgecolor='k', markeredgewidth=5)
        ax[1, 4].plot(2.8, 0.5, color='g', marker='4', markersize=20, markeredgecolor='b', markeredgewidth=5)
        ax[1, 4].plot(3.3, 0.5, color='g', marker='8', markersize=20, markeredgecolor='r', markeredgewidth=5)
        ax[1, 4].plot(3.8, 0.5, color='g', marker='s', markersize=20, markeredgecolor='c', markeredgewidth=5)
        ax[1, 4].plot(4.3, 0.5, color='g', marker='p', markersize=20, markeredgecolor='m', markeredgewidth=5)
        ax[1, 4].plot(4.8, 0.5, color='g', marker='P', markersize=20, markeredgecolor='y', markeredgewidth=5)
        ax[1, 4].plot(5.3, 0.5, color='g', marker='*', markersize=20, markeredgecolor='k', markeredgewidth=3)
        ax[1, 4].set_title("Some more markers in the following order:\n filled dot with outline, unfillable antibody up, unfillable antibody down,\n unfillable antibody left, unfillable antibody right, filled octogon, filled square, \n filled pentagon, filled plus, filled star")

        ##

        ax[2, 0].plot(x_data_for_plot, y_data_for_plot, color='c', linestyle='-', linewidth=5)
        ax[2, 0].plot(0.3, 0.25, color='g', marker='o', markersize=20, fillstyle='left')
        ax[2, 0].plot(0.8, 0.25, color='g', marker='d', markersize=20, fillstyle='right')
        ax[2, 0].plot(1.3, 0.25, color='g', marker='s', markersize=20, fillstyle='top')
        ax[2, 0].plot(1.8, 0.25, color='g', marker='o', markersize=20, markeredgecolor='r', markeredgewidth=5, fillstyle='bottom')
        ax[2, 0].axvline(3.14, color='r', linestyle='--', linewidth=2)
        ax[2, 0].axhline(0.5, color='m', linestyle='-.', linewidth=3)
        ax[2, 0].set_title("Vertical and horizontal dashed lines. \n Also shows half filled markers")

        ax[2, 1].plot(x_data_for_plot, y_data_for_plot, color='c', linestyle='-', linewidth=5)
        ax[2, 1].axvline(3.14, color='g', marker='d', markersize=20, markeredgecolor='r')
        ax[2, 1].axhline(0.5, color='m', linestyle='-.', linewidth=3, marker='o', markersize=20, markeredgecolor='r', markerfacecolor='y')
        ax[2, 1].set_title("Vertical line with the two edges as diamonds \n and horizontal line with two edges as circles")

        ax[2, 2].plot(x_data_for_plot, y_data_for_plot, color='c', linestyle='-', linewidth=5)
        ax[2, 2].axvline(3.14, ymin=0.1, ymax=0.8, color='g', marker='d', markersize=20, markeredgecolor='r')
        ax[2, 2].axhline(0.5, xmin=0.2, xmax=0.9, color='m', linestyle='-.', linewidth=3, marker='o', markersize=20, markeredgecolor='r', markerfacecolor='y')
        ax[2, 2].set_title("Vertical line with the two edges as diamonds \n and horizontal line with two edges as circles. \n Line edge locations have been adjusted")

        ax[2, 3].plot(x_data_for_plot, y_data_for_plot, color='c', linestyle='-', linewidth=5)
        ax[2, 3].axvline(3.14, ymin=0.1, ymax=0.8, color='g', marker='d', markersize=20, markeredgecolor='r', alpha=0.4)
        ax[2, 3].axhline(0.5, xmin=0.2, xmax=0.9, color='m', linestyle='-.', linewidth=3, marker='o', markersize=20, markeredgecolor='r', markerfacecolor='y', alpha=0.2)
        ax[2, 3].set_title("Transparency of the horizontal and vertical lines have been changed independently")

        ax[2, 4].plot(x_data_for_plot, y_data_for_plot, color='c', linestyle='-', linewidth=5, alpha=0.2)
        ax[2, 4].axvline(3.14, color='b', linewidth=5, marker='o', markersize=20, markeredgecolor='k', markerfacecolor='m', markeredgewidth=5)
        ax[2, 4].set_title("Transparency of the sinusoid has been changed")

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """

        #############################################
        ## Histogram specific
        # bins: (Int or array) If bins is an integer, it defines the number of equal-width bins in the range. If bins is a sequence, it defines the bin edges, including the left edge of the first bin and the right edge of the last bin; in this case, bins may be unequally spaced.
        # histyype: (String) This parameter is used to draw type of histogram. {‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’.
        # color: (string) The color of the histogram.
        # range : (Tuple) This parameter gives the lower and upper range of the bins.
        # density : (True or False) This parameter determines whether the histogram y axis should be normalized so it lies in the range 0 to 1.
        # cumulative: (True or False) If True, then a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values.
        # bottom: (Scalar or array) Location of the bottom of each bin, ie. bins are drawn from bottom to bottom + hist(x, bins) If a scalar, the bottom of each bin is shifted by the same amount. Can give a DC shift. If an array, each bin is shifted independently and the length of bottom must match the number of bins.
        # rwidth: (Float or None) The relative width of the bars as a fraction of the bin width. If None, automatically compute the width. Ignored if histtype is 'step' or 'stepfilled'.

        """

        right_max = 3
        top_max = 3
        font_scale_factor = np.min((right_max, top_max))

        ## Data for histogram
        num_data_points = 100
        data_for_histogram = np.multiply(np.random.randint(0, 100, num_data_points), np.random.randn(num_data_points))  # Generate 200 random numbers between zero and 100
        optional_bin_edge_values = [-50, 1, 5, 8, 20, 80, 99, 110]  # Make custom bin edges

        ## Plots for histogram
        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        # ax[1, 0].hist(data_for_histogram, bins=5, histtype='stepfilled', color='g', range=[0, 255], density=True, cumulative=False, bottom=11.3, rwidth = 0.2)

        ax[0, 0].hist(data_for_histogram, bins=5, histtype='bar', color='r', bottom=0, rwidth=1)
        ax[0, 1].hist(data_for_histogram, bins=5, histtype='bar', color='r', bottom=0, rwidth=0.8)
        ax[0, 2].hist(data_for_histogram, bins=5, histtype='bar', color='r', bottom=12, rwidth=0.8)
        ax[0, 3].hist(data_for_histogram, bins=5, histtype='bar', color='r', range=[10, 50], bottom=0, rwidth=0.8)
        ax[0, 4].hist(data_for_histogram, bins=5, histtype='bar', color='r', range=[10, 50], bottom=None, rwidth=None)

        ax[1, 0].hist(data_for_histogram, bins=5, histtype='step', color='g')
        ax[1, 1].hist(data_for_histogram, bins=5, histtype='bar', color='g', cumulative=True, rwidth=0.8)
        ax[1, 2].hist(data_for_histogram, bins=5, histtype='bar', color='g', density=True, rwidth=0.8)
        ax[1, 3].hist(data_for_histogram, bins=optional_bin_edge_values, histtype='bar', color='g', rwidth=0.8)
        ax[1, 4].hist(data_for_histogram, bins=5, histtype='bar', color='g', rwidth=0.8)
        # ax[1,4].set(xticks=[-50, 0, 50], xticklabels=['A','B', 3])

        plt.subplots_adjust(0, 0, right_max, top_max)
        plt.show()

        """


        #############################################
        ## Formatting Arrows and shapes and patches
        # Plotting arrows and customizing their parameters
        # Plotting equilibrium arrows
        # Plotting filled, non filled, curved arrows
        # Drawing circles, rectangles, squares, polygons and formatting them
        # Drawing segments, donuts, arcs, ellipses, triangles
        # Drawing custom shapes
        # Changing zorder

        """
        
        right_max = 5
        top_max = 4
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(0, 2 * 3.14, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 3
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        #
        ax[0, 0].set(xlim=[0, 6], ylim=[0, 6])

        my_arrow_patch_1 = matplotlib.patches.FancyArrowPatch(posA=(0.5, 1), posB=(1, 5), arrowstyle='Simple')  ## Here, posA and posB are coordinates
        ax[0, 0].add_patch(my_arrow_patch_1)
        my_arrow_patch_2 = matplotlib.patches.FancyArrowPatch(posA=(1.5, 1), posB=(2, 5), arrowstyle='Simple', mutation_scale=20)  ## Here, posA and posB are coordinates
        ax[0, 0].add_patch(my_arrow_patch_2)
        my_arrow_patch_3 = matplotlib.patches.FancyArrowPatch(posA=(2.5, 1), posB=(3, 5), arrowstyle='Simple', mutation_scale=40)  ## Here, posA and posB are coordinates
        ax[0, 0].add_patch(my_arrow_patch_3)
        my_arrow_patch_4 = matplotlib.patches.FancyArrowPatch(posA=(3.5, 1), posB=(4, 5), arrowstyle='Simple', mutation_scale=60)  ## Here, posA and posB are coordinates
        ax[0, 0].add_patch(my_arrow_patch_4)
        my_arrow_patch_5 = matplotlib.patches.FancyArrowPatch(posA=(4.5, 1), posB=(5, 5), arrowstyle='Simple', mutation_scale=80)  ## Here, posA and posB are coordinates
        ax[0, 0].add_patch(my_arrow_patch_5)

        ax[0, 0].set_title("Changing mutation scale from None to 80\n in steps of 20")

        #
        ax[0, 1].set(xlim=[0, 6], ylim=[0, 6])
        my_arrow_patch_1 = matplotlib.patches.FancyArrowPatch(posA=(0.5, 1), posB=(1.0, 5), mutation_scale=50, color='r', arrowstyle='fancy')
        ax[0, 1].add_patch(my_arrow_patch_1)
        my_arrow_patch_2 = matplotlib.patches.FancyArrowPatch(posA=(1.0, 1), posB=(1.5, 5), mutation_scale=50, color='r', arrowstyle='wedge')
        ax[0, 1].add_patch(my_arrow_patch_2)
        my_arrow_patch_3 = matplotlib.patches.FancyArrowPatch(posA=(1.5, 1), posB=(2.0, 5), mutation_scale=50, color='r', arrowstyle='->')
        ax[0, 1].add_patch(my_arrow_patch_3)
        my_arrow_patch_4 = matplotlib.patches.FancyArrowPatch(posA=(2.0, 1), posB=(2.5, 5), mutation_scale=50, color='r', arrowstyle='-|>')
        ax[0, 1].add_patch(my_arrow_patch_4)
        my_arrow_patch_5 = matplotlib.patches.FancyArrowPatch(posA=(2.5, 1), posB=(3.0, 5), mutation_scale=50, color='r', arrowstyle='<-')
        ax[0, 1].add_patch(my_arrow_patch_5)
        my_arrow_patch_6 = matplotlib.patches.FancyArrowPatch(posA=(3.0, 1), posB=(3.5, 5), mutation_scale=50, color='r', arrowstyle='<|-')
        ax[0, 1].add_patch(my_arrow_patch_6)
        my_arrow_patch_7 = matplotlib.patches.FancyArrowPatch(posA=(3.5, 1), posB=(4.0, 5), mutation_scale=50, color='r', arrowstyle='<->')
        ax[0, 1].add_patch(my_arrow_patch_7)
        my_arrow_patch_8 = matplotlib.patches.FancyArrowPatch(posA=(4.0, 1), posB=(4.5, 5), mutation_scale=50, color='r', arrowstyle='<|-|>')
        ax[0, 1].add_patch(my_arrow_patch_8)

        my_arrow_patch_9 = matplotlib.patches.FancyArrow(x=4.5, y=1, dx=0.1, dy=1, width=0.001, head_width=0.2, head_length=0.4, shape='full', overhang=0, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_9)
        my_arrow_patch_10 = matplotlib.patches.FancyArrow(x=5.0, y=1, dx=0.1, dy=1, width=0.001, head_width=0.2, head_length=0.4, shape='left', overhang=0, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_10)
        my_arrow_patch_11 = matplotlib.patches.FancyArrow(x=5.5, y=1, dx=0.1, dy=1, width=0.001, head_width=0.2, head_length=0.4, shape='right', overhang=0, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_11)
        my_arrow_patch_12 = matplotlib.patches.FancyArrow(x=5.0, y=3, dx=0.1, dy=1, width=0.001, head_width=0.2, head_length=0.4, shape='full', overhang=2, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_12)
        my_arrow_patch_13 = matplotlib.patches.FancyArrow(x=2.5, y=0.5, dx=1, dy=0, width=0.001, head_width=0.4, head_length=0.4, shape='right', overhang=0, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_13)
        my_arrow_patch_14 = matplotlib.patches.FancyArrow(x=3.9, y=0.4, dx=-1, dy=0, width=0.001, head_width=0.4, head_length=0.4, shape='right', overhang=0, color='c')  ## Here, dx and dy are the lengths and widths along the x and y axes respectively
        ax[0, 1].add_patch(my_arrow_patch_14)

        ax[0, 1].set_title("Changing arrow head styles for arrows. Shown in red are\n arrows in matplotlib.patches.FancyArrowPatch() class.\n Cyan arrows use a different \narrow plotting class: matplotlib.patches.FancyArrow()")
        #
        ax[0, 2].set(xlim=[0, 6], ylim=[0, 6])

        my_arrow_patch_1 = matplotlib.patches.FancyArrowPatch(posA=(0.5, 1), posB=(1.0, 5), mutation_scale=30, facecolor='r')
        my_arrow_patch_1.set_arrowstyle('fancy', head_length=2, head_width=1, tail_width=1)
        ax[0, 2].add_patch(my_arrow_patch_1)

        my_arrow_patch_2 = matplotlib.patches.FancyArrowPatch(posA=(1.5, 1), posB=(2.0, 5), mutation_scale=30, facecolor='r')
        my_arrow_patch_2.set_arrowstyle('fancy', head_length=1, head_width=1.5, tail_width=1)
        ax[0, 2].add_patch(my_arrow_patch_2)

        my_arrow_patch_3 = matplotlib.patches.FancyArrowPatch(posA=(2.5, 1), posB=(3.0, 5), mutation_scale=30, facecolor='r')
        my_arrow_patch_3.set_arrowstyle('fancy', head_length=1, head_width=1.5, tail_width=2)
        ax[0, 2].add_patch(my_arrow_patch_3)

        my_arrow_patch_4 = matplotlib.patches.FancyArrowPatch(posA=(3.5, 1), posB=(4.0, 5), mutation_scale=30, facecolor='g')
        my_arrow_patch_4.set_arrowstyle('<|-|>', head_length=2, head_width=1)
        ax[0, 2].add_patch(my_arrow_patch_4)

        my_arrow_patch_5 = matplotlib.patches.FancyArrowPatch(posA=(4.5, 1), posB=(5.0, 5), mutation_scale=30, facecolor='g', linewidth=5, edgecolor='g')
        my_arrow_patch_5.set_arrowstyle('<|-|>', head_length=0.2, head_width=1.5)
        ax[0, 2].add_patch(my_arrow_patch_5)

        ax[0, 2].set_title("Changing arrow head styles along with head parameters for \n 'fancy' arrow style (red)and curveFilledAB: '<|-|>' arrowstyle (green).")

        #

        ax[0, 3].set(xlim=[0, 6], ylim=[0, 6])

        my_arrow_patch_1 = matplotlib.patches.FancyArrowPatch(posA=(0.5, 1), posB=(1.0, 5), mutation_scale=30, edgecolor='r', arrowstyle='->', linewidth=2)
        my_arrow_patch_1.set_connectionstyle('arc3', rad=0.3)
        ax[0, 3].add_patch(my_arrow_patch_1)

        my_arrow_patch_2 = matplotlib.patches.FancyArrowPatch(posA=(1.5, 1), posB=(2.0, 5), mutation_scale=30, edgecolor='g', arrowstyle='->', linewidth=2)
        my_arrow_patch_2.set_connectionstyle('angle3', angleA=90, angleB=20)
        ax[0, 3].add_patch(my_arrow_patch_2)

        my_arrow_patch_3 = matplotlib.patches.FancyArrowPatch(posA=(2.5, 1), posB=(3.0, 5), mutation_scale=30, edgecolor='b', arrowstyle='->', linewidth=2)
        my_arrow_patch_3.set_connectionstyle('angle', angleA=90, angleB=0, rad=2)
        ax[0, 3].add_patch(my_arrow_patch_3)

        my_arrow_patch_4 = matplotlib.patches.FancyArrowPatch(posA=(3.5, 1), posB=(4.0, 5), mutation_scale=30, edgecolor='m', arrowstyle='->', linewidth=2)
        my_arrow_patch_4.set_connectionstyle('arc', angleA=90, angleB=0, armA=0, armB=0, rad=2)
        ax[0, 3].add_patch(my_arrow_patch_4)

        my_arrow_patch_5 = matplotlib.patches.FancyArrowPatch(posA=(4.5, 1), posB=(5.0, 5), mutation_scale=30, edgecolor='c', arrowstyle='->', linewidth=2)
        my_arrow_patch_5.set_connectionstyle('bar', armA=0.0, armB=0.0, fraction=0.3, angle=None)
        ax[0, 3].add_patch(my_arrow_patch_5)

        ax[0, 3].set_title("Curved arrows")

        #

        ax[0, 4].set(xlim=[0, 6], ylim=[0, 6])

        my_arrow_patch_1 = matplotlib.patches.FancyArrowPatch(posA=(0.5, 1), posB=(1.0, 5), mutation_scale=30, edgecolor='r', arrowstyle='->', linewidth=2)
        my_arrow_patch_1.set_connectionstyle('arc3', rad=0.1)
        ax[0, 4].add_patch(my_arrow_patch_1)
        my_arrow_patch_1b = matplotlib.patches.FancyArrowPatch(posA=(0.7, 1), posB=(1.2, 5), mutation_scale=30, edgecolor='r', arrowstyle='->', linewidth=2, linestyle=':')
        my_arrow_patch_1b.set_connectionstyle('arc3', rad=0.5)
        ax[0, 4].add_patch(my_arrow_patch_1b)

        my_arrow_patch_2 = matplotlib.patches.FancyArrowPatch(posA=(1.5, 1), posB=(2.0, 5), mutation_scale=30, edgecolor='g', arrowstyle='->', linewidth=2)
        my_arrow_patch_2.set_connectionstyle('angle3', angleA=180, angleB=100)
        ax[0, 4].add_patch(my_arrow_patch_2)
        my_arrow_patch_2b = matplotlib.patches.FancyArrowPatch(posA=(2.0, 1), posB=(2.5, 5), mutation_scale=30, edgecolor='g', arrowstyle='->', linewidth=2, linestyle='--')
        my_arrow_patch_2b.set_connectionstyle('angle3', angleA=0, angleB=90)
        ax[0, 4].add_patch(my_arrow_patch_2b)

        my_arrow_patch_3 = matplotlib.patches.FancyArrowPatch(posA=(2.8, 1), posB=(3.3, 5), mutation_scale=30, edgecolor='b', arrowstyle='->', linewidth=2)
        my_arrow_patch_3.set_connectionstyle('angle', angleA=90, angleB=0, rad=0)
        ax[0, 4].add_patch(my_arrow_patch_3)
        my_arrow_patch_3b = matplotlib.patches.FancyArrowPatch(posA=(3.0, 1), posB=(3.5, 5), mutation_scale=30, edgecolor='b', arrowstyle='->', linewidth=2, linestyle='-.')
        my_arrow_patch_3b.set_connectionstyle('angle', angleA=0, angleB=90, rad=0)
        ax[0, 4].add_patch(my_arrow_patch_3b)

        my_arrow_patch_4 = matplotlib.patches.FancyArrowPatch(posA=(3.8, 1), posB=(4.3, 5), mutation_scale=30, edgecolor='m', arrowstyle='->', linewidth=2)
        my_arrow_patch_4.set_connectionstyle('arc', angleA=0, angleB=0, armA=1, armB=1, rad=1)
        ax[0, 4].add_patch(my_arrow_patch_4)
        my_arrow_patch_4b = matplotlib.patches.FancyArrowPatch(posA=(4.1, 1), posB=(4.8, 5), mutation_scale=30, edgecolor='m', arrowstyle='->', linewidth=2, linestyle='--')
        my_arrow_patch_4b.set_connectionstyle('arc', angleA=80, angleB=0, armA=100, armB=50, rad=0)
        ax[0, 4].add_patch(my_arrow_patch_4b)

        my_arrow_patch_5 = matplotlib.patches.FancyArrowPatch(posA=(5.1, 1), posB=(5.4, 5), mutation_scale=30, edgecolor='c', arrowstyle='->', linewidth=2)
        my_arrow_patch_5.set_connectionstyle('bar', armA=0.2, armB=0.3, fraction=0.5, angle=2)
        ax[0, 4].add_patch(my_arrow_patch_5)
        my_arrow_patch_5b = matplotlib.patches.FancyArrowPatch(posA=(5.4, 1), posB=(5.8, 5), mutation_scale=30, edgecolor='c', arrowstyle='->', linewidth=2, linestyle=':')
        my_arrow_patch_5b.set_connectionstyle('bar', armA=2, armB=0.3, fraction=0.5, angle=20)
        ax[0, 4].add_patch(my_arrow_patch_5b)

        ax[0, 4].set_title("Same Curved arrows as previous plot. \n Parameters of those curved arrows have changed here")

        ##

        ax[1, 0].set(aspect='equal', xlim=[0, 6], ylim=[0, 6])
        my_wedge_patch_1 = matplotlib.patches.Wedge(center=(1, 5), r=0.6, theta1=20, theta2=100, facecolor='g', edgecolor='c', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_1)
        my_wedge_patch_2 = matplotlib.patches.Wedge(center=(3, 5), r=0.6, theta1=10, theta2=280, facecolor='r', edgecolor='y', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_2)
        my_wedge_patch_3 = matplotlib.patches.Wedge(center=(5, 5), r=0.6, theta1=20, theta2=200, facecolor='y', edgecolor='r', linewidth=4, hatch='/')
        ax[1, 0].add_patch(my_wedge_patch_3)
        my_wedge_patch_4 = matplotlib.patches.Wedge(center=(1, 3), r=0.6, theta1=80, theta2=180, width=0.4, facecolor='c', edgecolor='g', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_4)
        my_wedge_patch_5 = matplotlib.patches.Wedge(center=(3, 3), r=0.6, theta1=10, theta2=280, width=0.4, facecolor='m', edgecolor='b', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_5)
        my_wedge_patch_6 = matplotlib.patches.Wedge(center=(5, 3), r=0.6, theta1=20, theta2=200, width=0.4, facecolor='r', edgecolor='g', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_6)
        my_wedge_patch_7 = matplotlib.patches.Wedge(center=(1, 1), r=0.6, theta1=20, theta2=100, width=0.2, facecolor='c', edgecolor='k', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_7)
        my_wedge_patch_8 = matplotlib.patches.Wedge(center=(3, 1), r=0.6, theta1=10, theta2=280, width=0.1, facecolor='w', edgecolor='g', linewidth=4)
        ax[1, 0].add_patch(my_wedge_patch_8)
        my_wedge_patch_9 = matplotlib.patches.Wedge(center=(5, 1), r=0.6, theta1=20, theta2=200, width=0.5, facecolor='y', edgecolor='b', linewidth=4, hatch='+')
        ax[1, 0].add_patch(my_wedge_patch_9)
        ax[1, 0].set_title("The matplotlib.patches.Wedge() method is used here. \n These can be used to make filled circular segments and donut pieces")

        ax[1, 1].set(aspect='equal', xlim=[0, 6], ylim=[0, 6])
        my_regularpolygon_patch_1 = matplotlib.patches.RegularPolygon(xy=(1, 5), numVertices=3, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=0)
        ax[1, 1].add_patch(my_regularpolygon_patch_1)
        my_regularpolygon_patch_2 = matplotlib.patches.RegularPolygon(xy=(3, 5), numVertices=3, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=50)
        ax[1, 1].add_patch(my_regularpolygon_patch_2)
        my_regularpolygon_patch_3 = matplotlib.patches.RegularPolygon(xy=(5, 5), numVertices=3, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=170)
        ax[1, 1].add_patch(my_regularpolygon_patch_3)
        my_regularpolygon_patch_4 = matplotlib.patches.RegularPolygon(xy=(1, 3), numVertices=4, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=0)
        ax[1, 1].add_patch(my_regularpolygon_patch_4)
        my_regularpolygon_patch_5 = matplotlib.patches.RegularPolygon(xy=(3, 3), numVertices=4, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=40)
        ax[1, 1].add_patch(my_regularpolygon_patch_5)
        my_regularpolygon_patch_6 = matplotlib.patches.RegularPolygon(xy=(5, 3), numVertices=4, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=70)
        ax[1, 1].add_patch(my_regularpolygon_patch_6)
        my_regularpolygon_patch_7 = matplotlib.patches.RegularPolygon(xy=(1, 1.5), numVertices=5, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=0)
        ax[1, 1].add_patch(my_regularpolygon_patch_7)
        my_regularpolygon_patch_8 = matplotlib.patches.RegularPolygon(xy=(3, 1.5), numVertices=5, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=100)
        ax[1, 1].add_patch(my_regularpolygon_patch_8)
        my_regularpolygon_patch_9 = matplotlib.patches.RegularPolygon(xy=(5, 1.5), numVertices=5, radius=0.6, facecolor='y', edgecolor='k', linewidth=3, orientation=270)
        ax[1, 1].add_patch(my_regularpolygon_patch_9)
        ax[1, 1].text(0.1, 0.1, "Note: The matplotlib.patches.RegularPolygon() \n methods used here CAN be rotated", fontsize=15)
        ax[1, 1].set_title("These polygons use the matplotlib.patches.RegularPolygon() method. \n These polygons can be rotated")

        ax[1, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        my_bbox = matplotlib.patches.Rectangle(xy=(0, 1), width=1, height=2, facecolor='r', edgecolor='g', linewidth=5)  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 2].add_patch(my_bbox)
        ax[1, 2].set_title("Create a rectanglular bbox with left bottom corner at (0,1),\n with width 1 and height 2, and edgecolor green, facecolor red, and linewidth 5 ")

        ax[1, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        my_bbox = matplotlib.patches.Rectangle(xy=(0, 1), width=1, height=2, facecolor='r', edgecolor='g', linewidth=5, angle=45)  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 3].add_patch(my_bbox)
        ax[1, 3].set_title("Rotate the rectangle created in the previous step by 45 degree")

        ax[1, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        my_rect_patch_1 = matplotlib.patches.Rectangle(xy=(0, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='g', linewidth=1, hatch='/')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_1)
        my_rect_patch_2 = matplotlib.patches.Rectangle(xy=(2, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='b', linewidth=1, hatch='\\')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_2)
        my_rect_patch_3 = matplotlib.patches.Rectangle(xy=(4, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='m', linewidth=1, hatch='|')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_3)
        my_rect_patch_4 = matplotlib.patches.Rectangle(xy=(0, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='g', linewidth=1, hatch='-')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_4)
        my_rect_patch_5 = matplotlib.patches.Rectangle(xy=(2, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='b', linewidth=1, hatch='+')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_5)
        my_rect_patch_6 = matplotlib.patches.Rectangle(xy=(4, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='m', linewidth=1, hatch='X')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[1, 4].add_patch(my_rect_patch_6)
        ax[1, 4].set_title("Create rectangular boxes with different hatch styles")

        ax[2, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        my_rect_patch_1 = matplotlib.patches.Rectangle(xy=(0, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='g', linewidth=1, hatch='o')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_1)
        my_rect_patch_2 = matplotlib.patches.Rectangle(xy=(2, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='b', linewidth=1, hatch='O')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_2)
        my_rect_patch_3 = matplotlib.patches.Rectangle(xy=(4, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='m', linewidth=1, hatch='.')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_3)
        my_rect_patch_4 = matplotlib.patches.Rectangle(xy=(0, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='g', linewidth=1, hatch='*')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_4)
        my_rect_patch_5 = matplotlib.patches.Rectangle(xy=(2, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='k', linewidth=3, hatch='+', linestyle='-.')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_5)
        my_rect_patch_6 = matplotlib.patches.Rectangle(xy=(4, -1), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='m', linewidth=3, hatch='X', angle=45)  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 0].add_patch(my_rect_patch_6)
        ax[2, 0].set_title("Create rectangular boxes with different hatchings, linestyles, and angles")

        ax[2, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r', zorder=3)
        my_rect_patch_1 = matplotlib.patches.Rectangle(xy=(2.0, 0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='g', linewidth=1, hatch='o', zorder=4)  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 1].add_patch(my_rect_patch_1)
        my_rect_patch_2 = matplotlib.patches.Rectangle(xy=(2.5, 0.0), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='b', linewidth=1, hatch='O')  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 1].add_patch(my_rect_patch_2)
        my_rect_patch_3 = matplotlib.patches.Rectangle(xy=(3.0, -0.5), width=1, height=1, facecolor=(0.9, 0.9, 0.9), edgecolor='m', linewidth=1, hatch='.', zorder=2)  ## The first parameter is the x, y coordinate, and it NEEDS to be INSIDE a tuple. The next two arguments are width and height.
        ax[2, 1].add_patch(my_rect_patch_3)
        ax[2, 1].text(0, 0.5, "Just some text to show zorder", color='c', fontsize=20, fontstyle='italic', fontweight=1000)
        ax[2, 1].text(0, 0.75, "Some more text to show zorder", color='m', fontsize=20, fontstyle='italic', fontweight=1000, zorder=5)
        ax[2, 1].set_title("Change the order of different plot elements \n to determine which comes on top by changing zorder")

        ax[2, 2].set(aspect='equal', xlim=[0, 6], ylim=[0, 6])
        my_arc_patch_1 = matplotlib.patches.Arc(xy=(2.0, 0.5), width=1, height=1, theta1=0, theta2=360, color='r', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_1)
        my_arc_patch_2 = matplotlib.patches.Arc(xy=(3.0, 1.5), width=1, height=2, theta1=0, theta2=360, color='g', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_2)
        my_arc_patch_3 = matplotlib.patches.Arc(xy=(3.5, 3.5), width=2, height=1, theta1=0, theta2=360, color='b', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_3)
        my_arc_patch_4 = matplotlib.patches.Arc(xy=(4.0, 5.0), width=1, height=1, theta1=0, theta2=120, color='k', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_4)
        my_arc_patch_5 = matplotlib.patches.Arc(xy=(2.0, 5.0), width=1, height=1, theta1=120, theta2=160, color='c', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_5)
        my_arc_patch_6 = matplotlib.patches.Arc(xy=(1.0, 2.0), width=1, height=1, theta1=0, theta2=360, color='m', linewidth=3, linestyle='--')
        ax[2, 2].add_patch(my_arc_patch_6)
        my_arc_patch_7 = matplotlib.patches.Arc(xy=(1.0, 2.0), width=1, height=1, theta1=0, theta2=360, edgecolor='r', linewidth=3, linestyle='--', hatch='-')
        ax[2, 2].add_patch(my_arc_patch_7)
        my_arc_patch_8 = matplotlib.patches.Arc(xy=(4.0, 2.0), width=1, height=2, theta1=300, theta2=360, color='y', linewidth=3)
        ax[2, 2].add_patch(my_arc_patch_8)
        ax[2, 2].text(0.1, 4, "Note: The matplotlib.patches.Arc() \n methods used here can NOT be filled.\n For filled segments, use Wedge method", fontsize=15)
        ax[2, 2].set_title("Draw different types of matplotlib.patches.Arc() objects. \n Note: matplotlib.patches.Arc() objects cannot be filled")

        ax[2, 3].set(aspect='equal', xlim=[0, 6], ylim=[0, 6])
        my_circle_patch_1 = matplotlib.patches.Circle(xy=(2, 1), radius=0.5, facecolor='r', edgecolor='g', linewidth=3, hatch='*')
        ax[2, 3].add_patch(my_circle_patch_1)
        my_circlepolygon_patch_1 = matplotlib.patches.CirclePolygon(xy=(4, 4), radius=0.8, resolution=3, facecolor='r', edgecolor='m', linewidth=3, hatch='*')
        ax[2, 3].add_patch(my_circlepolygon_patch_1)
        my_circlepolygon_patch_2 = matplotlib.patches.CirclePolygon(xy=(2, 4), radius=0.8, resolution=4, facecolor='g', edgecolor='c', linewidth=3, hatch='-')
        ax[2, 3].add_patch(my_circlepolygon_patch_2)
        my_circlepolygon_patch_3 = matplotlib.patches.CirclePolygon(xy=(4, 2), radius=0.8, resolution=5, facecolor='r', edgecolor='b', linewidth=3, hatch='\\')
        ax[2, 3].add_patch(my_circlepolygon_patch_3)
        my_circlepolygon_patch_4 = matplotlib.patches.CirclePolygon(xy=(1, 2), radius=0.8, resolution=6, facecolor='b', edgecolor='k', linewidth=3, hatch='o')
        ax[2, 3].add_patch(my_circlepolygon_patch_4)
        my_circlepolygon_patch_5 = matplotlib.patches.CirclePolygon(xy=(1, 5), radius=0.5, resolution=7, facecolor='c', edgecolor='g', linewidth=3, hatch='/')
        ax[2, 3].add_patch(my_circlepolygon_patch_5)
        my_circlepolygon_patch_6 = matplotlib.patches.CirclePolygon(xy=(5, 1), radius=0.5, resolution=8, facecolor='m', edgecolor='r', linewidth=3, hatch='+')
        ax[2, 3].add_patch(my_circlepolygon_patch_6)
        ax[2, 3].text(0.1, 0.1, "Note: The matplotlib.patches.CirclePolygon() \n methods used here can NOT be rotated", fontsize=15)
        ax[2, 3].set_title("Using matplotlib.patches.Circle() class to draw the perfect circle.\n Using matplotlib.patches.CirclePolygon() class to draw polygons.\n Changing the 'resolution' parameter changes the number of edges on the polygon")

        ax[2, 4].set(aspect='equal', xlim=[0, 6], ylim=[0, 6])
        ## Make some data for a custom polygon
        x = np.linspace(2, 4, 10)
        y = 5 + np.sin(2 * x - 4)
        temp = np.array([x, y]).T
        coordinate_set = np.vstack((np.array([[0.5, 0.5], [0.5, 4], [2, 5]]), temp, np.array([[4, 5], [5, 4.5], [4, 4], [3, 3], [2, 3], [2, 2]])))
        my_polygon_patch_1 = matplotlib.patches.Polygon(coordinate_set, facecolor='y', edgecolor='r', hatch='|', linestyle='-', linewidth=4)
        ax[2, 4].add_patch(my_polygon_patch_1)
        ax[2, 4].set_title("Made a custom polygon by setting custom coordinates")

        plt.subplots_adjust(0, 0, right_max, top_max)

        plt.show()

        """



        #############################################
        ## Formatting Text and annotations
        # Changing different parameters of text including font size, style, bold/italic, color, highlighting etc
        # Enclosing text within text boxes
        # Text alignment
        # Using path effects to add shadows or outlines to text
        # Doing annotations using arrows or callouts

        """
        right_max = 5
        top_max = 4
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(0, 2 * 3.14, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 3
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols)

        ax[0, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 0].text(0, 0, "kasun", color='g')
        ax[0, 0].set_title("Add the text \"kasun\" at the (0,0) location and color it green")

        ax[0, 1].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 1].text(3.14 / 2, 0, "kasun", color='b', fontsize=20, fontstyle='italic')
        ax[0, 1].set_title("Add the text \"kasun\" at the (pi/2,0) location and color it blue.\n Increase fontsize and make font type italic")

        ax[0, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 2].text(0, 0, "kasun", color='b', fontsize=20, horizontalalignment='center')
        ax[0, 2].set_title("Example of how alignment works with respect to the location of text.\n CENTER aligned at (0,0) location")

        ax[0, 3].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 3].text(0, 0, "kasun", color='b', fontsize=20, horizontalalignment='left')
        ax[0, 3].set_title("Example of how alignment works with respect to the location of text.\n LEFT aligned at (0,0) location")

        ax[0, 4].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[0, 4].text(0, 0, "kasun", color='b', fontsize=20, horizontalalignment='right')
        ax[0, 4].set_title("Example of how alignment works with respect to the location of text.\n RIGHT aligned at (0,0) location")

        ##

        ax[1, 0].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 0].text(0, 0, "Italic and slightly boldfaced", color='b', fontsize=15, fontstyle='italic', fontweight=500)
        ax[1, 0].text(2, 0.1, "Italic and heavily boldfaced\n and rotated", color='b', fontsize=10, fontstyle='italic', fontweight=1000, rotation=45)
        ax[1, 0].text(3, -0.75, "Highlighted in yellow", color='k', fontsize=20, backgroundcolor='y')
        ax[1, 0].set_title("Some more text formatting")

        ax[1, 1].plot(x_data_for_plot, y_data_for_plot, fillstyle='left', color='r', path_effects=[path_effects.Normal(), path_effects.SimpleLineShadow()])
        ax[1, 1].text(0, 0, "Simple shadow", color='b', fontsize=20, path_effects=[path_effects.withSimplePatchShadow()])
        ax[1, 1].text(2, 0.4, "White text\n with green outline", color='w', fontsize=20, path_effects=[path_effects.withStroke(linewidth=3, foreground='g')])
        ax[1, 1].text(0, -0.6, "Add a green outline to white text,\n and add a heavy shadow\n that is red in color", color='w', fontsize=20, path_effects=[path_effects.withSimplePatchShadow(linewidth=5, foreground='r'), path_effects.withStroke(linewidth=3, foreground='g')])
        ax[1, 1].text(0, -1, "In this text, we have added a shadow using \n path_effects.withSimplePatchShadow(), .SimpleLineShadow(), \n and .withStroke() methods.\n These method do NOT accept conventional keywords", color='k', fontsize=10)
        ax[1, 1].set_title("Add a shadows and outlines to text or other objects")

        ax[1, 2].plot(x_data_for_plot, y_data_for_plot, marker='o', fillstyle='left', color='r')
        ax[1, 2].text(0, 0, "kasun", color='r', fontsize=60, path_effects=[path_effects.PathPatchEffect(offset=(4, -4), linewidth=1.5, hatch='+', facecolor='g', edgecolor='r'), path_effects.PathPatchEffect(edgecolor='k', linewidth=1.1, facecolor='c', linestyle=':')])  ## Note: When defining path effects, we should always include the normal style, and on top of that add the additional styles we want. if we use a method that startts with a 'with', the normal style is automatically included. Howwever, if not, we need to eexplicitly include the normal style as well.
        ax[1, 2].text(0, 1, 'This has a custom made shadow with two shadow layers ', path_effects=[path_effects.PathPatchEffect(facecolor='r', offset=(-2, -2), edgecolor='k'), path_effects.PathPatchEffect(facecolor='g', offset=(-3, -3), edgecolor='m'), path_effects.Normal()])
        ax[1, 2].text(0, -1, "In this text, we have added a shadow by creating an \n offset version of the original text. This uses a method called\n mpl.patheffects.PathPatchEffect(). This PathPatchEffect() method\n accepts the normal keywords like edgecolor, \n facecolor, linestyle, etc.", color='k', fontsize=10)
        ax[1, 2].set_title("Showing that more than one path effect can be added \n to generate compound font styles.")

        ax[1, 3].set(aspect='equal', xlim=(0, 8), ylim=(0, 6))
        my_fancybbox_patch_1 = matplotlib.patches.FancyBboxPatch(xy=(1, 4), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='k', linewidth=2, linestyle='--')
        ax[1, 3].add_patch(my_fancybbox_patch_1)
        my_fancybbox_patch_2 = matplotlib.patches.FancyBboxPatch(xy=(2.5, 4), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='w', linewidth=2, linestyle='--', mutation_scale=0.1, mutation_aspect=1)
        ax[1, 3].add_patch(my_fancybbox_patch_2)
        my_fancybbox_patch_3 = matplotlib.patches.FancyBboxPatch(xy=(4, 4), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='g', linewidth=2, linestyle='--', mutation_scale=2, mutation_aspect=1)
        ax[1, 3].add_patch(my_fancybbox_patch_3)
        my_fancybbox_patch_4 = matplotlib.patches.FancyBboxPatch(xy=(6, 4), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='y', linewidth=2, linestyle='--', mutation_scale=3, mutation_aspect=1)
        ax[1, 3].add_patch(my_fancybbox_patch_4)
        my_fancybbox_patch_5 = matplotlib.patches.FancyBboxPatch(xy=(1, 1), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='k', linewidth=2, linestyle='--')
        ax[1, 3].add_patch(my_fancybbox_patch_5)
        my_fancybbox_patch_6 = matplotlib.patches.FancyBboxPatch(xy=(2.5, 1), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='w', linewidth=2, linestyle='--', mutation_scale=1, mutation_aspect=0.1)
        ax[1, 3].add_patch(my_fancybbox_patch_6)
        my_fancybbox_patch_7 = matplotlib.patches.FancyBboxPatch(xy=(4, 1), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='g', linewidth=2, linestyle='--', mutation_scale=1, mutation_aspect=2)
        ax[1, 3].add_patch(my_fancybbox_patch_7)
        my_fancybbox_patch_8 = matplotlib.patches.FancyBboxPatch(xy=(6, 1), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='y', linewidth=2, linestyle='--', mutation_scale=1, mutation_aspect=3)
        ax[1, 3].add_patch(my_fancybbox_patch_8)
        ax[1, 3].set_title("Changing mutation scale (top row) and \n mutation aspect (bottom row) parameters of a rounded FancyBboxPatch()")

        ax[1, 4].set(aspect='equal', xlim=(0, 8), ylim=(0, 6))
        my_fancybbox_patch_1 = matplotlib.patches.FancyBboxPatch(xy=(1, 4), width=0.3, height=0.6, boxstyle='circle', edgecolor='r', facecolor='k', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_1)
        my_fancybbox_patch_2 = matplotlib.patches.FancyBboxPatch(xy=(3, 4), width=0.3, height=0.6, boxstyle='larrow', edgecolor='r', facecolor='w', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_2)
        my_fancybbox_patch_3 = matplotlib.patches.FancyBboxPatch(xy=(4.5, 4), width=0.3, height=0.6, boxstyle='square', edgecolor='r', facecolor='g', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_3)
        my_fancybbox_patch_4 = matplotlib.patches.FancyBboxPatch(xy=(6, 4), width=0.3, height=0.6, boxstyle='rarrow', edgecolor='r', facecolor='c', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_4)
        my_fancybbox_patch_5 = matplotlib.patches.FancyBboxPatch(xy=(1, 2), width=0.3, height=0.6, boxstyle='round', edgecolor='r', facecolor='y', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_5)
        my_fancybbox_patch_6 = matplotlib.patches.FancyBboxPatch(xy=(3, 2), width=0.3, height=0.6, boxstyle='round4', edgecolor='r', facecolor='m', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_6)
        my_fancybbox_patch_7 = matplotlib.patches.FancyBboxPatch(xy=(4.5, 2), width=0.3, height=0.6, boxstyle='sawtooth', edgecolor='r', facecolor='b', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_7)
        my_fancybbox_patch_8 = matplotlib.patches.FancyBboxPatch(xy=(6, 2), width=0.3, height=0.6, boxstyle='roundtooth', edgecolor='r', facecolor='gray', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_8)
        my_fancybbox_patch_9 = matplotlib.patches.FancyBboxPatch(xy=(4, 0.5), width=1, height=0.6, boxstyle='darrow', edgecolor='r', facecolor='r', linewidth=2, linestyle='--')
        ax[1, 4].add_patch(my_fancybbox_patch_9)
        my_fancybbox_patch_10 = matplotlib.patches.FancyBboxPatch(xy=(0.5, 0.5), width=0.3, height=0.6, boxstyle='circle', edgecolor='r', facecolor='k', linewidth=2, linestyle='--', path_effects=[path_effects.PathPatchEffect(offset=(-2, -2)), path_effects.Normal()])
        ax[1, 4].add_patch(my_fancybbox_patch_10)
        ax[1, 4].set_title("Different FancyBboxPatch() boxstyles. \n all these can be used as bboxes for text. \n Also shows that path effects can be used for anything")

        ##

        ax[2, 0].set(aspect='equal', xlim=(0, 8), ylim=(-6, 6))
        ax[2, 0].text(2, 4, "round", color='w', fontsize=20, rotation=45, bbox=dict(boxstyle='round', mutation_aspect=2, facecolor='k', edgecolor='y', hatch='-'))
        ax[2, 0].text(3, 1, "darrow", color='r', fontsize=20, rotation=20, bbox=dict(boxstyle='darrow', mutation_aspect=2, facecolor='g', edgecolor='k', hatch='*'))
        ax[2, 0].text(4, -1, "larrow", color='g', fontsize=20, rotation=0, bbox=dict(boxstyle='larrow', mutation_aspect=2, facecolor='r', edgecolor='k', hatch='/'))
        ax[2, 0].text(5, -5, "circle", color='w', fontsize=20, rotation=-30, bbox=dict(boxstyle='circle', mutation_aspect=1, facecolor='b', edgecolor='k', hatch='+'))
        ax[2, 0].text(1.5, -5, "ellipse", color='w', fontsize=20, rotation=-45, bbox=dict(boxstyle='circle', mutation_aspect=1.5, facecolor='m', edgecolor='k', hatch='|'))
        ax[2, 0].set_title("Text boxes")

        ax[2, 1].set(aspect='equal', xlim=(-6, 8), ylim=(-8, 6))

        ###
        num_data_points = 100
        x_data_for_plot = np.linspace(-12, 12, 200)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        ax[2, 2].set(aspect='equal', xlim=(-12, 12), ylim=(-8, 8))
        ax[2, 2].plot(x_data_for_plot, y_data_for_plot, fillstyle='left', color='gray')
        point_at_x_loc = 3.14 / 2
        point_at_y_loc = np.sin(point_at_x_loc)
        text_string = "Annotate location:\n" + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3)))
        ax[2, 2].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), color='c', fontsize=10, rotation=45)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 11
        text_start_y_loc = point_at_y_loc - 7
        text_string = "With Arrow tail bottom left of\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 2].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='r', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='r'))

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 0
        text_start_y_loc = point_at_y_loc - 7
        text_string = "With Arrow tail directly below\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 2].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='g', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='g'))

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 2.5
        text_start_y_loc = point_at_y_loc - 3.4
        text_string = "With Arrow tail bottom right of\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 2].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='m', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='m'))

        ax[2, 2].text(-11, 4, "I think the arrow tail does NOT go to the position given by \n'xytext' parameter. To check where the arrow tail connects to,\n I will repeat the same annotations, \n however with text boxes and markers at\n xy and xytext locations in the next subplot", color='k', fontsize=10)
        ax[2, 2].set_title("Annotations")

        #
        ax[2, 3].set(aspect='equal', xlim=(-12, 12), ylim=(-8, 8))
        ax[2, 3].plot(x_data_for_plot, y_data_for_plot, fillstyle='left', color='gray')

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 11
        text_start_y_loc = point_at_y_loc - 7
        text_string = "With Arrow tail bottom left of\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='r', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='r'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'rx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'ro', markersize=10, zorder=10)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 0
        text_start_y_loc = point_at_y_loc - 7
        text_string = "With Arrow tail directly below\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='g', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='g'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'gx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'go', markersize=10, zorder=10)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 2.5
        text_start_y_loc = point_at_y_loc - 3.4
        text_string = "With Arrow tail bottom right of\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='m', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='m'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'mx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'mo', markersize=10, zorder=10)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 11
        text_start_y_loc = point_at_y_loc + 5
        text_string = "With Arrow tail top left of\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='b', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='b'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'bx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'bo', markersize=10, zorder=10)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 0
        text_start_y_loc = point_at_y_loc + 5
        text_string = "With Arrow tail directly above\nAnnotate location: " + str((np.round(point_at_x_loc, 3), np.round(point_at_y_loc, 3))) + "\nwith text start location:\n " + str((np.round(text_start_x_loc, 3), np.round(text_start_y_loc, 3)))
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='k', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='k'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'kx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'ko', markersize=10, zorder=10)

        point_at_x_loc = 7
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 4
        text_start_y_loc = point_at_y_loc + 2.5
        text_string = "Testing center connection"
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='y', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='y'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'yx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'yo', markersize=10, zorder=10)

        point_at_x_loc = 0
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + -11.5
        text_start_y_loc = point_at_y_loc + -2
        text_string = "Here, connecting the arrow \n tail to the center right is \n shorter than connecting \n to the top or bottom right"
        ax[2, 3].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='c', fontsize=10, rotation=0, arrowprops=dict(width=2, headwidth=5, color='c'), bbox=dict(boxstyle='square', facecolor='w'))
        ax[2, 3].plot(point_at_x_loc, point_at_y_loc, 'cx', markersize=10, zorder=10)
        ax[2, 3].plot(text_start_x_loc, text_start_y_loc, 'co', markersize=10, zorder=10)

        ax[2, 3].set_title("Annotations with bounding boxes. These show that the arrow tail is \nALWAYS connected at the SHORTEST distance from pointing \nlocation to an EDGE or CENTER of the BOUNDING BOX\n around the text, whichever is smaller")

        #

        ax[2, 4].set(aspect='equal', xlim=(-12, 12), ylim=(-8, 8))
        ax[2, 4].plot(x_data_for_plot, y_data_for_plot, fillstyle='left', color='gray')

        point_at_x_loc = -3.14 / 2
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 4
        text_start_y_loc = point_at_y_loc + 4
        text_string = "center aligned text \n with NO arrow shrinking"
        ax[2, 4].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), horizontalalignment='center', color='m', fontsize=10, rotation=0, bbox=dict(facecolor='w'), arrowprops=dict(width=2, headwidth=5, color='m'))

        point_at_x_loc = -3.14 / 2
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc - 4.5
        text_start_y_loc = point_at_y_loc - 5
        text_string = "center aligned text \n WITH some arrow shrinking"
        ax[2, 4].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), horizontalalignment='center', color='c', fontsize=10, rotation=0, bbox=dict(facecolor='w'), arrowprops=dict(width=2, headwidth=5, color='c', shrink=0.1))

        point_at_x_loc = 3.14 / 2
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 4.5
        text_start_y_loc = point_at_y_loc + 2
        text_string = "90 degree arrow\n with shrinking"
        ax[2, 4].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='w', fontsize=10, bbox=dict(boxstyle='round', facecolor='gray'), arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, connectionstyle="angle,angleA=0,angleB=90,rad=10", color='gray'))

        point_at_x_loc = 3.14 / 2
        point_at_y_loc = np.sin(point_at_x_loc)
        text_start_x_loc = point_at_x_loc + 1
        text_start_y_loc = point_at_y_loc - 8
        text_string = "Bubble using borderless \n Wedge arrow and \n borderless rounded box"
        ax[2, 4].annotate(text_string, xy=(point_at_x_loc, point_at_y_loc), xytext=(text_start_x_loc, text_start_y_loc), color='k', fontsize=10, bbox=dict(boxstyle='round', facecolor='pink', edgecolor='pink'), arrowprops=dict(arrowstyle="wedge, tail_width=2", facecolor='pink', edgecolor='pink', patchA=None, patchB=None, relpos=(0.2, 0.8), connectionstyle="arc3, rad=-0.1"))

        ax[2, 4].set_title("Annotations with arrow shrinking, fancy arrows, wedge arrows, etc.")

        plt.subplots_adjust(0, 0, right_max, top_max)

        plt.show()

        """

        #############################################

        #############################################
        ## Custom subplot layouts using gridspec

        """

        right_max = 5
        top_max = 3
        font_scale_factor = np.min((right_max, top_max))
        font_size = 15

        num_main_subplot_rows = 3
        num_main_subplot_cols = 6

        # First create empty figure(s)
        fig_1 = plt.figure()

        main_grid_subplots = matplotlib.gridspec.GridSpec(nrows=num_main_subplot_rows, ncols=num_main_subplot_cols, figure=fig_1)  # The mpl.gridspec.GridSpec(nrows, ncols, fig) method will create a container that contains parameters (SubplotSpec objects) that CAN BE USED to produce a set of subplots in the figure called 'fig'.

        ##        
        ax_1_1 = fig_1.add_subplot(main_grid_subplots[0, 0])  # To show that the grid spec actually created parameters that can directly be used to create subplots. In the main_grid_subplots, we should have 3 rows and 6 columns because that is what we set. That means, there are parameters to create 18 subplots. Here, I will take the information in the top left of them to create a subplot
        ax_1_1.set_title("Single plot on main grid", fontsize=font_size)
        ax_1_2 = fig_1.add_subplot(main_grid_subplots[0, 1])  # Add a subplot at the (0,1) position
        ax_1_2.set_title("Single plot on main grid", fontsize=font_size)
        ax_1_3 = fig_1.add_subplot(main_grid_subplots[2, 1:3])  # This will create a single subplot that spans the 2nd row and the columns (2 and 3 combined).
        ax_1_3.set_title("Two COLUMNS of main grid combined", color='r', fontsize=font_size)
        ax_1_3.spines[['top', 'bottom', 'left', 'right']].set_color('r')
        ax_1_3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(4)
        ax_1_4 = fig_1.add_subplot(main_grid_subplots[0:2, 2])  # This will create a single subplot that spans the 3rd column and rows (1 and 2) combined
        ax_1_4.set_title("Two ROWS of main grid combined", color='g', fontsize=font_size)
        ax_1_4.spines[['top', 'bottom', 'left', 'right']].set_color('g')
        ax_1_4.spines[['top', 'bottom', 'left', 'right']].set_linewidth(4)
        ax_1_5 = fig_1.add_subplot(main_grid_subplots[0, 2])  # This will create a subplot at location (0,2), thereby drawing over an existing subplot.
        ax_1_5.spines[['top', 'bottom', 'left', 'right']].set_color('c')
        ax_1_5.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_1_5.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_1_5.text(0.5, 0.5, "Single plot on main grid\n drawn over an existing combined plot", horizontalalignment='center', fontsize=font_size, color='c')

        sub_grid_spec_on_ax_1_3 = main_grid_subplots[2, 1:3].subgridspec(2, 3)  # This will divide the subplot spanning row 3 and columns (2 and 3) into a smaller subgrid of size (2 x 3).
        ax_1_3_sub_1 = fig_1.add_subplot(sub_grid_spec_on_ax_1_3[0, 0:2])  # This will add a subplot combining the first two columns of the 1st row of the above created subgrid 
        ax_1_3_sub_1.text(0.5, 0.5, "Two SUB COLUMNS combined on the\n SUBGRID INSIDE a COMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_1_3_sub_2 = fig_1.add_subplot(sub_grid_spec_on_ax_1_3[0, 2])  # This will add an individual subplot to the subgrid
        ax_1_3_sub_2.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_1_3_sub_3 = fig_1.add_subplot(sub_grid_spec_on_ax_1_3[1, 0])  # This will add an individual subplot to the subgrid
        ax_1_3_sub_3.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_1_3_sub_4 = fig_1.add_subplot(sub_grid_spec_on_ax_1_3[1, 1])  # This will add an individual subplot to the subgrid
        ax_1_3_sub_4.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_1_3_sub_5 = fig_1.add_subplot(sub_grid_spec_on_ax_1_3[1, 2])  # This will add an individual subplot to the subgrid
        ax_1_3_sub_5.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)

        sub_grid_at_1_bottom_left = main_grid_subplots[1:3, 0].subgridspec(3, 3)
        ax_at_1_bottom_left_sub_1 = fig_1.add_subplot(sub_grid_at_1_bottom_left[0:2, 1])
        ax_at_1_bottom_left_sub_1.text(0.5, 0.05, "Two SUB ROWS combined on the \nSUBGRID INSIDE a COMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_1_bottom_left_sub_2 = fig_1.add_subplot(sub_grid_at_1_bottom_left[2, 0:2])
        ax_at_1_bottom_left_sub_2.text(0.5, 0.5, "Two SUB COLUMNS \ncombined on the SUBGRID \n INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_at_1_bottom_left_sub_3 = fig_1.add_subplot(sub_grid_at_1_bottom_left[0, 0])
        ax_at_1_bottom_left_sub_3.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_1_bottom_left_sub_4 = fig_1.add_subplot(sub_grid_at_1_bottom_left[1, 0])
        ax_at_1_bottom_left_sub_4.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_1_bottom_left_sub_5 = fig_1.add_subplot(sub_grid_at_1_bottom_left[0, 2])
        ax_at_1_bottom_left_sub_5.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_1_bottom_left_sub_6 = fig_1.add_subplot(sub_grid_at_1_bottom_left[1, 2])
        ax_at_1_bottom_left_sub_6.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_1_bottom_left_sub_7 = fig_1.add_subplot(sub_grid_at_1_bottom_left[2, 2])
        ax_at_1_bottom_left_sub_7.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)

        ## Repeat the same layout, but have different interplot spacings and paddings

        ax_2_1 = fig_1.add_subplot(main_grid_subplots[0, 3])
        ax_2_1.set_title("Single plot on main grid", fontsize=font_size)
        ax_2_2 = fig_1.add_subplot(main_grid_subplots[0, 4])
        ax_2_2.set_title("Single plot on main grid", fontsize=font_size)
        ax_2_3 = fig_1.add_subplot(main_grid_subplots[2, 4:6])
        ax_2_3.set_title("Two COLUMNS of main grid combined", color='r', fontsize=font_size)
        ax_2_3.spines[['top', 'bottom', 'left', 'right']].set_color('r')
        ax_2_3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(4)
        ax_2_4 = fig_1.add_subplot(main_grid_subplots[0:2, 5])
        ax_2_4.set_title("Two ROWS of main grid combined", color='g', fontsize=font_size)
        ax_2_4.spines[['top', 'bottom', 'left', 'right']].set_color('g')
        ax_2_4.spines[['top', 'bottom', 'left', 'right']].set_linewidth(4)
        ax_2_5 = fig_1.add_subplot(main_grid_subplots[0, 5])
        ax_2_5.spines[['top', 'bottom', 'left', 'right']].set_color('c')
        ax_2_5.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_5.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_5.text(0.5, 0.5, "Single plot on main grid\n drawn over an existing combined plot", horizontalalignment='center', fontsize=font_size, color='c')

        sub_grid_spec_on_ax_2_3 = main_grid_subplots[2, 4:6].subgridspec(2, 3, wspace=2, hspace=1.5)  # Increase the horizontal space and increase the vertical space between the subplots of this subgrid
        ax_2_3_sub_1 = fig_1.add_subplot(sub_grid_spec_on_ax_2_3[0, 0:2])
        ax_2_3_sub_1.text(0.5, 0.5, "Two SUB COLUMNS combined on the\n SUBGRID INSIDE a COMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_2_3_sub_1.spines[['top', 'bottom', 'left', 'right']].set_color('m')
        ax_2_3_sub_1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_3_sub_1.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_3_sub_2 = fig_1.add_subplot(sub_grid_spec_on_ax_2_3[0, 2])
        ax_2_3_sub_2.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size - 5)
        ax_2_3_sub_2.spines[['top', 'bottom', 'left', 'right']].set_color('m')
        ax_2_3_sub_2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_3_sub_2.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_3_sub_3 = fig_1.add_subplot(sub_grid_spec_on_ax_2_3[1, 0])
        ax_2_3_sub_3.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size - 5)
        ax_2_3_sub_3.spines[['top', 'bottom', 'left', 'right']].set_color('m')
        ax_2_3_sub_3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_3_sub_3.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_3_sub_4 = fig_1.add_subplot(sub_grid_spec_on_ax_2_3[1, 1])
        ax_2_3_sub_4.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size - 5)
        ax_2_3_sub_4.spines[['top', 'bottom', 'left', 'right']].set_color('m')
        ax_2_3_sub_4.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_3_sub_4.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_3_sub_5 = fig_1.add_subplot(sub_grid_spec_on_ax_2_3[1, 2])
        ax_2_3_sub_5.text(0.5, 0.5, "Single plot on the \nSUBGRID INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size - 5)
        ax_2_3_sub_5.spines[['top', 'bottom', 'left', 'right']].set_color('m')
        ax_2_3_sub_5.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_2_3_sub_5.spines[['top', 'bottom', 'left', 'right']].set_linestyle('-.')
        ax_2_3.text(0.5, 0.4, "The wspace and hspace of the SUBGRID plots INSIDE the COMBINED MAINGRID plot\n has been INCREASED", horizontalalignment='center', fontsize=font_size, color='m')

        sub_grid_at_2_bottom_left = main_grid_subplots[1:3, 3].subgridspec(3, 3, width_ratios=[1, 3, 1])  # Change the ratios of widths of each grid column in this subgrid to be in the ratio [1:3:1]
        ax_at_2_bottom_left_sub_1 = fig_1.add_subplot(sub_grid_at_2_bottom_left[0:2, 1])
        ax_at_2_bottom_left_sub_1.text(0.5, 0.05, "Two SUB ROWS combined on the \nSUBGRID INSIDE a COMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size, rotation=90)
        ax_at_2_bottom_left_sub_1.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_1.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_2 = fig_1.add_subplot(sub_grid_at_2_bottom_left[2, 0:2])
        ax_at_2_bottom_left_sub_2.text(0.5, 0.5, "Two SUB COLUMNS \ncombined on the SUBGRID \n INSIDE a \nCOMBINED MAINGRID plot", horizontalalignment='center', fontsize=font_size)
        ax_at_2_bottom_left_sub_2.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_2.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_3 = fig_1.add_subplot(sub_grid_at_2_bottom_left[0, 0])
        ax_at_2_bottom_left_sub_3.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size - 5, rotation=90)
        ax_at_2_bottom_left_sub_3.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_3.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_4 = fig_1.add_subplot(sub_grid_at_2_bottom_left[1, 0])
        ax_at_2_bottom_left_sub_4.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size - 5, rotation=90)
        ax_at_2_bottom_left_sub_4.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_4.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_5 = fig_1.add_subplot(sub_grid_at_2_bottom_left[0, 2])
        ax_at_2_bottom_left_sub_5.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size - 5, rotation=90)
        ax_at_2_bottom_left_sub_5.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_5.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_6 = fig_1.add_subplot(sub_grid_at_2_bottom_left[1, 2])
        ax_at_2_bottom_left_sub_6.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size - 5, rotation=90)
        ax_at_2_bottom_left_sub_6.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_6.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_7 = fig_1.add_subplot(sub_grid_at_2_bottom_left[2, 2])
        ax_at_2_bottom_left_sub_7.text(0.5, 0.1, "Single plot \n on the \nSUBGRID INSIDE a \nCOMBINED \nMAINGRID plot", horizontalalignment='center', fontsize=font_size - 5, rotation=90)
        ax_at_2_bottom_left_sub_7.spines[['top', 'bottom', 'left', 'right']].set_color('b')
        ax_at_2_bottom_left_sub_7.spines[['top', 'bottom', 'left', 'right']].set_linewidth(3)
        ax_at_2_bottom_left_sub_1.text(0.5, 1.01, "The COLUMN WIDTHS of the following set of subgrids\n have been arranged according to ratio [1, 3, 1] ", horizontalalignment='center', fontsize=font_size - 3, color='b')

        fig_1.suptitle("A main gridspec of size 3 x 6", y=0.2 + top_max, x=right_max / 2, fontsize=20)

        fig_1.subplots_adjust(0, 0, right_max, top_max)

        fig_1.show()

        """

        #############################################
        ## Creating subFIGURES

        """
        num_subfig_rows = 1
        num_subfig_cols = 2

        main_fig = plt.figure(facecolor='y', constrained_layout=True, figsize=(10, 5))
        main_fig.suptitle("Main figure", fontsize=30)

        sub_figures = main_fig.subfigures(num_subfig_rows, num_subfig_cols)

        ###
        left_figure = sub_figures[0]
        left_figure.set(facecolor='r')
        left_figure.suptitle("Left subfigure", fontsize=20)

        left_axes = left_figure.subplots(3, 3)

        ###

        right_figure = sub_figures[1]
        right_figure.set(facecolor='g', linewidth=5, edgecolor='c')
        right_figure.suptitle("Right subfigure", fontsize=20)

        right_gridspec = right_figure.add_gridspec(3, 3)
        right_ax_1 = right_figure.add_subplot(right_gridspec[:, 1])
        right_ax_2 = right_figure.add_subplot(right_gridspec[0:2, 0])
        right_ax_3 = right_figure.add_subplot(right_gridspec[1:, 2])
        right_ax_4 = right_figure.add_subplot(right_gridspec[0, 2])
        right_ax_5 = right_figure.add_subplot(right_gridspec[2, 0])

        ######

        num_subfig_rows = 1
        num_subfig_cols = 3

        main_fig = plt.figure(facecolor='y', constrained_layout=True, figsize=(15, 5))
        main_fig.suptitle("Main figure\n wspace on subfigures: 0.2", fontsize=30)

        sub_figures = main_fig.subfigures(num_subfig_rows, num_subfig_cols, wspace=0.2, width_ratios=[2, 1, 3])

        ###
        left_figure = sub_figures[0]
        left_figure.set(facecolor='r')
        left_figure.suptitle("Left subfigure\n ratio: 2", fontsize=20)

        left_axes = left_figure.subplots(3, 3)

        ###

        middle_figure = sub_figures[1]
        middle_figure.set(facecolor='g', linewidth=5, edgecolor='c')
        middle_figure.suptitle("Middle subfigure\n ratio: 1, subfig outline enabled", fontsize=15)

        middle_gridspec = middle_figure.add_gridspec(3, 3)
        middle_ax_1 = middle_figure.add_subplot(middle_gridspec[:, 1])
        middle_ax_2 = middle_figure.add_subplot(middle_gridspec[0:2, 0])
        middle_ax_3 = middle_figure.add_subplot(middle_gridspec[1:, 2])
        middle_ax_4 = middle_figure.add_subplot(middle_gridspec[0, 2])
        middle_ax_5 = middle_figure.add_subplot(middle_gridspec[2, 0])

        ###

        right_figure = sub_figures[2]
        right_figure.set(frameon=False)
        # Frameon parameter will enable or disable the background of a figure/subfigure
        right_figure.suptitle("Right subfigure\n ratio: 3, subfig frameon=False", fontsize=20)

        right_gridspec = right_figure.add_gridspec(3, 3)
        right_ax_1 = right_figure.add_subplot(right_gridspec[:, 1])
        right_ax_2 = right_figure.add_subplot(right_gridspec[0:2, 0])
        right_ax_3 = right_figure.add_subplot(right_gridspec[1:, 2])
        right_ax_4 = right_figure.add_subplot(right_gridspec[0, 2])
        right_ax_5 = right_figure.add_subplot(right_gridspec[2, 0])

        ######

        num_subfig_rows = 3
        num_subfig_cols = 3

        main_fig = plt.figure(facecolor='y', constrained_layout=True, figsize=(15, 10))
        main_fig.suptitle("Main figure\n Subfigures on gridspec. wspace set to 0, hspace set to 0.1", fontsize=30)
        sub_fig_gridspec = matplotlib.gridspec.GridSpec(num_subfig_rows, num_subfig_cols, figure=main_fig, wspace=0, hspace=0.1)

        sub_fig_1 = main_fig.add_subfigure(sub_fig_gridspec[0:2, 0], facecolor='r')
        sub_fig_1.suptitle("Top left TWO ROW combined subfigure\n width ratios 1:4")
        sub_fig_1_axes = sub_fig_1.subplots(2, 2, gridspec_kw=dict(width_ratios=[1, 4]))

        sub_fig_2 = main_fig.add_subfigure(sub_fig_gridspec[2, 0], facecolor='g')
        sub_fig_2.suptitle("Bottom left single subfigure\n height ratios 1:3")
        sub_fig_2_axes = sub_fig_2.subplots(2, 2, gridspec_kw=dict(height_ratios=[1, 3]))

        sub_fig_3 = main_fig.add_subfigure(sub_fig_gridspec[1:, 1:], facecolor='gray')
        sub_fig_3.suptitle("Bottom right TWO ROW, TWO COLUMN combined subfigure\n wspace=0.5")
        sub_fig_3_axes = sub_fig_3.subplots(2, 2, gridspec_kw=dict(wspace=0.5))

        sub_fig_4 = main_fig.add_subfigure(sub_fig_gridspec[0, 1], facecolor='m')
        sub_fig_4.suptitle("Single subfigure")
        sub_fig_4_axes = sub_fig_4.subplots(2, 3)

        sub_fig_5 = main_fig.add_subfigure(sub_fig_gridspec[0, 2], facecolor='c')
        sub_fig_5.suptitle("Single subfigure")
        sub_fig_5_axes = sub_fig_5.subplots(2, 3)

        """

        #############################################
        ## Inset formatting

        """

        right_max = 7
        top_max = 5
        font_scale_factor = np.min((right_max, top_max))

        ## Data for the plot
        num_data_points = 100
        x_data_for_plot = np.linspace(-6, 6, 100)  ## Generate some data between 0 and 2*pi
        y_data_for_plot = np.sin(x_data_for_plot)  ## create a sinusoid

        num_subplot_rows = 2
        num_subplot_cols = 5
        fig, ax = plt.subplots(num_subplot_rows, num_subplot_cols, constrained_layout=True, figsize=(20, 10))

        ax[0, 0].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 0].set(aspect='equal', xlim=[-6, 6], ylim=[-6, 6])
        ax[0, 0].set_title("Inset set using DATA coordinates")
        my_inset_ax = ax[0, 0].inset_axes(bounds=[3, 2, 2, 3], transform=ax[0, 0].transData)

        ax[0, 1].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 1].set(aspect='equal', xlim=[-6, 6], ylim=[-6, 6])
        ax[0, 1].set_title("Inset set using AXES coordinates")
        my_inset_ax = ax[0, 1].inset_axes(bounds=[0.7, 0.7, 0.2, 0.2], transform=ax[0, 1].transAxes, facecolor='y')
        my_inset_ax.spines['right'].set(linewidth=5, color='g', linestyle='--')
        my_inset_ax.spines['left'].set(linewidth=5, color='r', linestyle='--')
        my_inset_ax.spines['top'].set(linewidth=5, color='b', linestyle='-')
        my_inset_ax.spines['bottom'].set(linewidth=5, color='m', linestyle='-')

        ax[0, 2].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 2].set(aspect='equal', xlim=[-6, 6], ylim=[-6, 6])
        ax[0, 2].set_title("Inset set using DATA coordinates.\n Tick labels and locations of \ninset axis have been adjusted")
        my_inset_ax = ax[0, 2].inset_axes(bounds=[0, 2, 5, 3], transform=ax[0, 2].transData)
        my_inset_ax.set(xticks=[1, 9, 50, 60, 120], yticks=[5, 9, 15, 22, 100], xticklabels=['a', 'x', '5', 'k', 'pp'], yticklabels=['ya', 'dsw', 'lpw', 'wk', 'dw'], xlim=[-10, 120], ylim=[-15, 25])
        my_inset_ax.text(0.1, -11, "Some text inside\n the inset", fontsize=8)
        temp_x_data = np.linspace(0, 40, 100)
        temp_y_data = np.sqrt((temp_x_data - 15) ** 2)
        my_inset_ax.plot(temp_x_data, temp_y_data)

        ax[0, 3].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 3].set(aspect='equal', xlim=[-6, 6], ylim=[-6, 8])
        ax[0, 3].set_title("Inset set using DATA coordinates.\n Tick labels and locations of \ninset axis have been adjusted")
        my_inset_ax = ax[0, 3].inset_axes(bounds=[0, 3, 5, 3], transform=ax[0, 3].transData)  # Draw an inset by zooming on the range given by xlim and ylim
        my_inset_ax.set(xlim=[-2.5, -1], ylim=[-1.3, -0.7], title="Zoomed inset", xlabel="x", ylabel="y")
        my_inset_ax.plot(x_data_for_plot, y_data_for_plot)
        indicator_lines = ax[0, 3].indicate_inset_zoom(my_inset_ax)  # The zooming indicator box is drawn on the xlim and ylim of the inset, however on the main plot
        ax[0, 3].text(-5, -4, "The zooming indicator box is \ndrawn on the main plot, \n at the locations given by\n xlim and ylim on the inset")

        ax[0, 4].plot(x_data_for_plot, y_data_for_plot)
        ax[0, 4].set(aspect='equal', xlim=[-6, 6], ylim=[-6, 8])
        ax[0, 4].set_title("Made the indicator box darker, \ncolored, and styled")
        my_inset_ax = ax[0, 4].inset_axes(bounds=[0, 3, 5, 3], transform=ax[0, 4].transData)  # Draw an inset by zooming on the range given by xlim and ylim
        my_inset_ax.set(xlim=[-2.5, -1], ylim=[-1.3, -0.7])
        my_inset_ax.plot(x_data_for_plot, y_data_for_plot)
        rectangle_patch, indicator_lines = ax[0, 4].indicate_inset_zoom(my_inset_ax)
        rectangle_patch.set(facecolor='c', linewidth=4, edgecolor='m', linestyle='--')
        indicator_lines[1].set(linewidth=6, edgecolor='g', linestyle=':')
        indicator_lines[2].set(linewidth=6, edgecolor='r', linestyle=':')

        """


        #############################################


        right_max = 10
        top_max = 5
        font_scale_factor = np.min((right_max, top_max))

        #         this_color_channel_of_roi_image = roi_image_of_chosen_segment[:, :, i]
        #         ax[1, i].imshow(this_color_channel_of_roi_image)
        #
        #         this_color_channel_of_outlier_removed_roi_image_flattened = outlier_removed_roi_image_of_chosen_segment[:, :, i].flatten()
        #         this_color_channel_of_outlier_removed_roi_image_flattened_to_plot = this_color_channel_of_outlier_removed_roi_image_flattened[this_color_channel_of_outlier_removed_roi_image_flattened > 0]
        #         ax[2, i].hist(this_color_channel_of_outlier_removed_roi_image_flattened_to_plot, bins=40, histtype='stepfilled', color=color_set[i], range=[0, 255])
        #
        #         this_color_channel_of_outlier_removed_roi_image = outlier_removed_roi_image_of_chosen_segment[:, :, i]
        #         ax[3, i].imshow(this_color_channel_of_outlier_removed_roi_image)
        #
        #         for j in range(num_subplot_rows):
        #             # ax[j, i].set_title(color_set[i], fontsize=10*font_scale_factor)
        #             ax[j, i].xaxis.set_tick_params(length=1 * top_max, width=right_max, labelsize=0 * font_scale_factor)
        #             ax[j, i].spines[['right', 'top']].set_visible(False)
        #             ax[j, i].spines['left'].set_linewidth(3)
        #             ax[j, i].spines['bottom'].set_linewidth(3)
        #             ax[j, i].grid(False, which='both', axis='x')
        #             ax[j, i].yaxis.set_tick_params(length=1 * top_max, width=right_max, labelsize=0 * font_scale_factor)
        #
        #             if j % 2 != 0:
        #                 ax[j, i].xaxis.set_visible(False)
        #                 ax[j, i].yaxis.set_visible(False)
        #                 ax[j, i].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        #
        # my_ax2 = [0, 0]
        # my_ax2[0] = fig.add_axes([0, -top_max - 0.5, right_max / 2 - 0.1, top_max])
        # my_ax2[0].imshow(original_he_stained_image)
        # my_ax2[0].set_title("Original H&E image", fontsize=10 * font_scale_factor)
        #
        # my_ax2[1] = fig.add_axes([right_max / 2 + 0.1, -top_max - 0.5, right_max / 2 - 0.1, top_max])
        # my_ax2[1].imshow(chosen_segment_image)
        # my_ax2[1].set_title("A chosen segment", fontsize=10 * font_scale_factor)
        #
        # for j in range(len(my_ax2)):
        #     my_ax2[j].xaxis.set_visible(False)
        #     my_ax2[j].yaxis.set_visible(False)
        #     my_ax2[j].spines[['right', 'top', 'left', 'bottom']].set_visible(0)
        #
        # plot_title = 'H and E image segmentation'
        # # fig.text(-0.1 * right_max, top_max / 2, 'Intensity', ha='center', va='center', rotation='vertical', fontsize=15 * font_scale_factor)
        # plt.suptitle(plot_title, x=right_max / 2, y=top_max + 0.5, fontsize=10 * font_scale_factor)
        # plt.subplots_adjust(left=0, bottom=0, right=right_max, top=top_max, wspace=0.15, hspace=0.3)
        # plt.show()
        #
        # if self.save_figures == 1:
        #     fig.savefig(self.global_plots_figure_saving_folder + plot_title.replace(' ', '_') + '.' + self.global_plots_fileformat, dpi=self.global_plots_dpi - 100, bbox_inches='tight')

    def plot_normalized_hist_svm_weight_vectors(self, right_max=30, top_max=6):

        font_scale_factor = np.min((right_max, top_max))

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
            n, bins, patches = ax[i].hist(self.global_plots_my_linear_svc.coef_.reshape(self.global_plots_num_dim_reduced_components, -1)[i], bins=50, range=(-3, 3), histtype='stepfilled', color='r', density=True)
            ax[i].spines[['right', 'top']].set_visible(0)
            ax[i].set_title('SVM Weights' + '\n' + self.global_plots_used_dim_reduction_technique + ' ' + str(i + 1), fontsize=30)
            ax[i].yaxis.set_tick_params(length=20, width=5, labelsize=25)
            ax[i].xaxis.set_tick_params(length=20, width=5, labelsize=25)
            # ax[i].xaxis.set_visible(False)
            # ax[i].yaxis.set_visible(False)
            # ax[i].grid(False, which='both', axis='x')
            # ax[i].spines['bottom'].set_linewidth(3)

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
                im_store = np.copy(datagrid_store_recovered[i])
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

            im_store = np.copy(self.datagrid_store_recovered[i])
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

        for j in range(self.global_plots_num_dim_reduced_components):
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

        for j in range(self.global_plots_num_dim_reduced_components):
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
        for j in range(self.global_plots_num_dim_reduced_components):
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

        for j in range(self.num_dim_reduced_components):
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


#############################################################################################
#############################################################################################
#############################################################################################

    def normalize_3d_datagrid_store_to_given_range(self, range_min_value, range_max_value, datagrid_store_3d, used_dim_reduction_technique):


        num_datasets = len(datagrid_store_3d)
        num_dim_reduced_components = datagrid_store_3d[0].shape[0]
        normalized_datagrid_store = []
        for i in range(num_datasets):
            this_dataset = np.copy(datagrid_store_3d[i])
            for_nmf = this_dataset == -1
            for_pca = this_dataset == 1000
            if used_dim_reduction_technique == 'nmf':
                this_dataset[for_nmf | for_pca] = 0
            elif used_dim_reduction_technique == 'pca':
                for component_k in range(this_dataset.shape[0]):
                    this_component = np.copy(this_dataset[component_k])
                    this_component[this_component == 1000] = np.min(this_dataset[component_k])
                    this_dataset[component_k] = this_component
            # The above 4 lines removed the -1s or 1000s I introduced to the background pixels of nmf and pca respectively

            each_dim_reduced_component_unrolled = np.reshape(this_dataset, (num_dim_reduced_components, -1))
            max_per_dim_reduced_component = np.tile(np.reshape(np.max(each_dim_reduced_component_unrolled, axis=1), (-1, 1, 1)), (1, this_dataset.shape[1], this_dataset.shape[2]))
            min_per_dim_reduced_component = np.tile(np.reshape(np.min(each_dim_reduced_component_unrolled, axis=1), (-1, 1, 1)), (1, this_dataset.shape[1], this_dataset.shape[2]))
            this_dataset_normalized = 0 + np.multiply((np.divide((this_dataset - (min_per_dim_reduced_component)), ((max_per_dim_reduced_component) - (min_per_dim_reduced_component)))), (range_max_value - range_min_value))
            this_dataset_normalized = np.uint8(this_dataset_normalized)

            normalized_datagrid_store.append(this_dataset_normalized)

        return normalized_datagrid_store

    def create_false_colored_normalized_dim_reduced_datagrid_store(self, normalized_datagrid_store):


        num_datasets = len(normalized_datagrid_store)
        num_dim_reduced_components = normalized_datagrid_store[0].shape[0]
        false_colored_normalized_datagrid_store = []

        for i in range(num_datasets):
            false_colored_normalized_datagrid_store_per_dataset = []
            for j in range(num_dim_reduced_components):
                dim_reduced_image_gray = normalized_datagrid_store[i][j]
                zeros = np.zeros(dim_reduced_image_gray.shape, np.uint8)
                dim_reduced_image_false_BGR = cv2.merge((zeros, dim_reduced_image_gray, zeros))  # False BGR image by assiging a grayscale image to the green channel of an otherwise all-black BGR image

                false_colored_normalized_datagrid_store_per_dataset.append(dim_reduced_image_false_BGR)

            false_colored_normalized_datagrid_store.append(false_colored_normalized_datagrid_store_per_dataset)

        return false_colored_normalized_datagrid_store

    def cv_subplot(self, imgs, pad=10, titles=None, win_name='all_combo_aligned'):
        """
        @brief Can be used to easily plot subplots in opencv.
        @param imgs: 2d np array of imgs (each img an np arrays of depth 1 or 3).
        @param pad: number of pixels to use for padding between images. must be even
        @param titles: (optional) np array of subplot titles
        @param win_name: name of cv2 window
        @return combined_frame
        """
        rows, cols = imgs.shape

        subplot_shapes = np.array([list(map(np.shape, x)) for x in imgs])
        sp_height, sp_width, depth = np.max(np.max(subplot_shapes, axis=0), axis=0)

        title_pad = 30
        if titles is not None:
            pad_top = pad + title_pad
        else:
            pad_top = pad

        frame = np.zeros((rows * (sp_height + pad_top), cols * (sp_width + pad), depth))

        for r in range(rows):
            for c in range(cols):
                img = imgs[r, c]
                h, w, _ = img.shape
                y0 = r * (sp_height + pad_top) + pad_top // 2
                x0 = c * (sp_width + pad) + pad // 2
                frame[y0:y0 + h, x0:x0 + w, :] = img

                if titles is not None:
                    frame = cv2.putText(frame, titles[r, c], (x0, y0 - title_pad // 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        # cv2.imshow(win_name, np.uint8(frame))
        # cv2.waitKey(0)

        return np.uint8(frame)

    def slider_callback_for_partial_dim_reduced_result_plotting(self, val):

        # 'partial_dim_reduced_results'


        current_partial_result_number = cv2.getTrackbarPos('batch_number', 'partial_dim_reduced_results')

        input_image = self.temp_image_store[current_partial_result_number]

        cv2.imshow('partial_dim_reduced_results', input_image)

    def plot_evolution_of_partial_dim_reduced_ion_images_with_num_iter(self, compact_msi_data_pathname, folder_name_for_partial_dim_reduced_results, used_dim_reduction_technique, use_opencv_plotting=1, path_to_saved_open_cv_grid_of_partial_results=''):

        """
        @brief See how ion images of NMF components in partial nmf results during batch wise NMF evolve as the number of batches (iterations) increase

        @param compact_msi_data_pathname: The pathname to a compact_msi_data object (NOT double_compact. Double compact will NOT work)
        @param folder_name_for_partial_dim_reduced_results: This is the name of the folder where all the partial nmf results for a given run is stored. Enter ONLY the folder name, and NOT the path name.
        @param used_dim_reduction_technique: Need to include this for now. In the future, it will be replaced to be done automatically
        @param use_opencv_plotting: Always enabled
        @param path_to_saved_open_cv_grid_of_partial_results: This is an optional parameter. If this is given, we skip the calculation of image grids, and go straightaway to the plotting

        """
        if not(path_to_saved_open_cv_grid_of_partial_results):

            compact_msi_data = pickle.load(open(compact_msi_data_pathname, 'rb'))  # Load the compact version of msi data
            name_arr = [compact_msi_data.dataset_order, compact_msi_data.dataset_order[0][:-1], compact_msi_data.dataset_order, compact_msi_data.dataset_order[4][:-1]]

            folder_path = compact_msi_data.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name_for_partial_dim_reduced_results +'/'

            partial_result_filename_array = os.listdir(folder_path)
            partial_nmf_dict_store = []
            partial_nmf_based_datagrid_store = []


            all_image_stores = [0]*len(partial_result_filename_array)
            for count, this_partial_result_filename in enumerate(partial_result_filename_array):
                if this_partial_result_filename == '.' or this_partial_result_filename == '..':
                    pass
                else:
                    print("Now processing partial dimensionality reduced dataset ", count, " out of ", len(partial_result_filename_array), " datasets")

                    this_dim_reduced_dict_filename = folder_path + this_partial_result_filename
                    dim_reduced_dict = np.load(this_dim_reduced_dict_filename, allow_pickle=True)[()]
                    self.used_dim_reduction_technique = used_dim_reduction_technique
                    this_partial_dataset_batch_number = dim_reduced_dict['current_batch_number']
                    if used_dim_reduction_technique == 'nmf':
                        dim_reduced_object = nmf_kp(compact_msi_data, saved_nmf_filename=this_dim_reduced_dict_filename)
                    elif used_dim_reduction_technique == 'pca':
                        dim_reduced_object = pca_kp(compact_msi_data, saved_pca_filename=this_dim_reduced_dict_filename)
                    num_dim_reduced_components = dim_reduced_object.num_dim_reduced_components
                    this_3d_datagrid_store = data_preformatter_kp(dim_reduced_object).create_3d_array_from_dim_reduced_data()['datagrid_store']
                    num_datasets = len(this_3d_datagrid_store)
                    this_normalized_datagrid_store_3d = self.normalize_3d_datagrid_store_to_given_range(0, 255, this_3d_datagrid_store, used_dim_reduction_technique)
                    false_colored_normalized_datagrid_store = self.create_false_colored_normalized_dim_reduced_datagrid_store(this_normalized_datagrid_store_3d)


                    if use_opencv_plotting == 1:
                        image_label_store = []
                        for i in range(num_datasets):
                            image_label_store_per_dataset = []
                            for j in range(num_dim_reduced_components):
                                label_string_this_dataset = compact_msi_data.dataset_order[i] + ',' + used_dim_reduction_technique + str(j)
                                image_label_store_per_dataset.append(label_string_this_dataset)

                            image_label_store.append(image_label_store_per_dataset)

                        labels_for_display = np.array(image_label_store, dtype=object)

                        false_colored_dim_reduced_datagrid_store_for_display = np.array(false_colored_normalized_datagrid_store, dtype=object)

                        combined_false_colored_dim_reduced_datagrid_store_for_display = self.cv_subplot(false_colored_dim_reduced_datagrid_store_for_display, titles=labels_for_display)

                        all_image_stores[this_partial_dataset_batch_number-1] = combined_false_colored_dim_reduced_datagrid_store_for_display


            save_path_and_filename = compact_msi_data.global_path_name + 'saved_outputs/figures/' + folder_name_for_partial_dim_reduced_results + 'nmf_images.npy'
            np.save(save_path_and_filename, all_image_stores, 'dtype=object')

        else:
            all_image_stores = np.load(path_to_saved_open_cv_grid_of_partial_results, allow_pickle=True)


        self.temp_image_store = all_image_stores
        cv2.namedWindow('partial_dim_reduced_results', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('batch_number', 'partial_dim_reduced_results', 0, len(all_image_stores), self.slider_callback_for_partial_dim_reduced_result_plotting)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def plot_evolution_of_partial_dim_reduced_spectra_with_num_iter(self, double_compact_msi_data_pathname, folder_name_for_partial_dim_reduced_results, used_dim_reduction_technique, path_to_saved_spectra_of_partial_results='', select_component_range=[0, 1, 2, 5]):

        """
        @brief See how the SPECTRA  of NMF components in partial nmf results during batch wise NMF evolve as the number of batches (iterations) increase

        @param double_compact_msi_data_pathname: The pathname to a double compact_msi_data object
        @param folder_name_for_partial_dim_reduced_results: This is the name of the folder where all the partial nmf results for a given run is stored. Enter ONLY the folder name, and NOT the path name.
        @param used_dim_reduction_technique: Need to include this for now. In the future, it will be replaced to be done automatically
        @param path_to_saved_spectra_of_partial_results: This is an optional parameter. If this is given, we skip the calculation of spectra grids, and go straightaway to the plotting
        @param select_component_range: This argument is used to select only a subset of the components for plotting. This is important because displaying all components simulataneously makes the diagram too cluttered

        """
        if not(path_to_saved_spectra_of_partial_results):

            compact_msi_data = pickle.load(open(double_compact_msi_data_pathname, 'rb'))  # Load the compact version of msi data
            name_arr = [compact_msi_data.dataset_order, compact_msi_data.dataset_order[0][:-1], compact_msi_data.dataset_order, compact_msi_data.dataset_order[4][:-1]]

            folder_path = compact_msi_data.global_path_name + 'saved_outputs/nmf_outputs/' + folder_name_for_partial_dim_reduced_results +'/'

            partial_result_filename_array = os.listdir(folder_path)

            all_spectra_stores = [0]*len(partial_result_filename_array)
            for count, this_partial_result_filename in enumerate(partial_result_filename_array):
                if this_partial_result_filename == '.' or this_partial_result_filename == '..':
                    pass
                else:
                    print("Now processing partial dimensionality reduced dataset ", count, " out of ", len(partial_result_filename_array), " datasets")

                    this_dim_reduced_dict_filename = folder_path + this_partial_result_filename
                    dim_reduced_dict = np.load(this_dim_reduced_dict_filename, allow_pickle=True)[()]
                    self.used_dim_reduction_technique = used_dim_reduction_technique
                    this_partial_dataset_batch_number = dim_reduced_dict['current_batch_number']
                    if used_dim_reduction_technique == 'nmf':
                        dim_reduced_object = nmf_kp(compact_msi_data, saved_nmf_filename=this_dim_reduced_dict_filename)
                    elif used_dim_reduction_technique == 'pca':
                        dim_reduced_object = pca_kp(compact_msi_data, saved_pca_filename=this_dim_reduced_dict_filename)
                    num_dim_reduced_components = dim_reduced_object.num_dim_reduced_components
                    num_datasets = len(dim_reduced_object.dim_reduced_dict['dim_reduced_outputs'])
                    spectra_store = dim_reduced_object.dim_reduced_dict['dim_reduced_outputs'][1][0]



                    fig, ax = plt.subplots(np.min([num_dim_reduced_components, len(select_component_range)]), 1, dpi=self.global_plots_dpi)
                    min_mz_all_datasets = compact_msi_data.min_mz_after_truncation
                    max_mz_all_datasets = compact_msi_data.max_mz_after_truncation
                    mz_array = np.arange(min_mz_all_datasets, max_mz_all_datasets, compact_msi_data.bin_size)



                    for component_number, component_spectra in enumerate(spectra_store[select_component_range]):
                        markers, stems, base = ax[component_number].stem(mz_array, spectra_store[component_number], markerfmt='None', )
                        stems.set_linewidth(2)
                        markers.set_markersize(0.1)
                        ax[component_number].spines[['bottom', 'right', 'left', 'top']].set_visible(0)
                        ax[component_number].tick_params(top=0, bottom=0, left=0, right=0, labelleft=0, labelbottom=0)
                        # ax[component_number].set_title(self.global_plots_used_dim_reduction_technique + '  ' + str(j + 1), fontsize=40)
                        # ax[component_number].set_ylabel(self.name_arr[int(np.floor(i / 4))] + '  ' + str((i % 4) + 1), fontsize=40)

                    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
                    plt.rcParams["figure.figsize"] = [7.50, 3.50]
                    fig.canvas.draw()
                    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    all_spectra_stores[this_partial_dataset_batch_number-1] = img


            save_path_and_filename = compact_msi_data.global_path_name + 'saved_outputs/figures/' + folder_name_for_partial_dim_reduced_results + 'nmf_spectra_for_components' + str(select_component_range).replace(" ","_") + '.npy'
            np.save(save_path_and_filename, all_spectra_stores, 'dtype=object')

        else:
            all_spectra_stores = np.load(path_to_saved_spectra_of_partial_results, allow_pickle=True)


        self.temp_image_store = all_spectra_stores
        cv2.namedWindow('partial_dim_reduced_results', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('batch_number', 'partial_dim_reduced_results', 0, len(all_spectra_stores), self.slider_callback_for_partial_dim_reduced_result_plotting)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


