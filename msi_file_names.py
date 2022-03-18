
def msi_filenames(my_os='linux'):

    """
    @brief Returns the file path names of the msi datasets.
        Example Usage:     my_os = 'windows'
        msi_filename_array, dataset_order = msi_filenames(my_os=my_os)

    @param my_os Default is linux. Can set to windows as well.

    """
    if my_os == 'linux':
        global_path_name = "/home/kasun/aim_hi_project_kasun"
        msi_path_name = "/home/kasun/aim_hi_project_kasun/data/"
        cph1_file = msi_path_name+"DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy"
        cph2_file = msi_path_name+"DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy"

        cph3_file = msi_path_name+"DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy"
        cph4_file = msi_path_name+"DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy"

        naive1_file = msi_path_name+"DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy"
        naive2_file = msi_path_name+"DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy"

        naive3_file = msi_path_name+"DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy"
        naive4_file = msi_path_name+"DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy"

        msi_file_names = [cph1_file, cph2_file, cph3_file, cph4_file, naive1_file, naive2_file, naive3_file, naive4_file]

    elif my_os == 'windows':

        global_path_name = "D:/msi_project_data"
        msi_path_name = "D:/msi_project_data/binned_binsize_1/"
        cph1_file = msi_path_name+"DMSI0005_F894CPH_LipidNeg_IMZML/dmsi0005.npy"
        cph2_file = msi_path_name+"DMSI0004_F893CPH_LipidNeg_IMZML/dmsi0004.npy"

        cph3_file = msi_path_name+"DMSI0008_F895CPH_LipidNeg_IMZML/dmsi0008.npy"
        cph4_file = msi_path_name+"DMSI0011_F896CPH_LipidNegIMZML/dmsi0011.npy"

        naive1_file = msi_path_name+"DMSI0002_F885naive_LipidNeg_IMZML/dmsi0002.npy"
        naive2_file = msi_path_name+"DMSI0006_FF886naive_LipidNeg_IMZML/dmsi0006.npy"

        naive3_file = msi_path_name+"DMSI0009_F887Naive_LipidNeg_IMZML/dmsi0009.npy"
        naive4_file = msi_path_name+"DMSI0012_F888Naive_LipidNeg_IMZML/dmsi0012.npy"

        msi_file_names = [cph1_file, cph2_file, cph3_file, cph4_file, naive1_file, naive2_file, naive3_file, naive4_file]

    dataset_order = ['cph1', 'cph2', 'cph3', 'cph4', 'naive1', 'naive2', 'naive3', 'naive4']

    return msi_file_names, dataset_order, global_path_name

