
def he_filenames(my_os='linux'):
    """
    @brief Returns the file path names of the H&E stained images. Also returns the order in which file names are arranged.
        Example   usage: my_os = 'windows'
        he_filename_array = he_filenames(my_os=my_os)

    @param os Default is linux. Can set to windows as well.

    """
    if my_os == 'linux':

        path_name= "/home/kasun/aim_hi_project_kasun/data/"
        cph1_he_stain = path_name+"DMSI0005_F894CPH_LipidNeg_IMZML/DMSI0005HE.tif"
        cph2_he_stain = path_name+"DMSI0004_F893CPH_LipidNeg_IMZML/DMSI0004HE.tif"

        cph3_he_stain = path_name+"DMSI0008_F895CPH_LipidNeg_IMZML/DMSI0008HE.tif"
        cph4_he_stain = path_name+"DMSI0011_F896CPH_LipidNegIMZML/DMSI0011HE.tif"

        naive1_he_stain = path_name+"DMSI0002_F885naive_LipidNeg_IMZML/DMSI0002HE.tif"
        naive2_he_stain = path_name+"DMSI0006_FF886naive_LipidNeg_IMZML/DMSI0006HE.tif"

        naive3_he_stain = path_name+"DMSI0009_F887Naive_LipidNeg_IMZML/DMSI0009HE.tif"
        naive4_he_stain = path_name+"DMSI0012_F888Naive_LipidNeg_IMZML/DMSI0012HE.tif"

        he_stain_file_names = [cph1_he_stain, cph2_he_stain, cph3_he_stain, cph4_he_stain, naive1_he_stain, naive2_he_stain, naive3_he_stain,
                          naive4_he_stain]

    elif my_os == 'windows':

        path_name = "D:/msi_project_data/binned_binsize_1/"
        cph1_he_stain = path_name+"DMSI0005_F894CPH_LipidNeg_IMZML/DMSI0005HE.tif"
        cph2_he_stain = path_name+"DMSI0004_F893CPH_LipidNeg_IMZML/DMSI0004HE.tif"

        cph3_he_stain = path_name+"DMSI0008_F895CPH_LipidNeg_IMZML/DMSI0008HE.tif"
        cph4_he_stain = path_name+"DMSI0011_F896CPH_LipidNegIMZML/DMSI0011HE.tif"

        naive1_he_stain = path_name+"DMSI0002_F885naive_LipidNeg_IMZML/DMSI0002HE.tif"
        naive2_he_stain = path_name+"DMSI0006_FF886naive_LipidNeg_IMZML/DMSI0006HE.tif"

        naive3_he_stain = path_name+"DMSI0009_F887Naive_LipidNeg_IMZML/DMSI0009HE.tif"
        naive4_he_stain = path_name+"DMSI0012_F888Naive_LipidNeg_IMZML/DMSI0012HE.tif"

        he_stain_file_names = [cph1_he_stain, cph2_he_stain, cph3_he_stain, cph4_he_stain, naive1_he_stain, naive2_he_stain, naive3_he_stain,
                      naive4_he_stain]

    dataset_order = ['cph1', 'cph2', 'cph3', 'cph4', 'naive1', 'naive2', 'naive3', 'naive4']

    return he_stain_file_names,dataset_order

