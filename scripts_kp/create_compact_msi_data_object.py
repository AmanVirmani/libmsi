from msi_file_names import msi_filenames
from he_file_names import he_filenames
from dataloader_kp import dataloader_kp


my_os = 'windows'
bin_size = 1
msi_filename_array, dataset_order, global_path_name = msi_filenames(my_os=my_os)
he_filename_array = he_filenames(my_os=my_os)
tic_truncate_data = 1
tic_truncate_quantity = 500
tic_add_dummy_data = 0

msi_data = dataloader_kp(global_path_name, dataset_order=dataset_order, msi_filename_array=msi_filename_array,
                         he_filename_array=he_filename_array, bin_size=bin_size, tic_truncate_data=tic_truncate_data,
                         tic_add_dummy_data=tic_add_dummy_data, tic_truncate_quantity=tic_truncate_quantity)

destination_filename_for_compact_msi_data = "D:/msi_project_data/binned_binsize_1/compact_data_object/compact_msi_data_test.npy"
compact_msi_data = msi_data.create_reduced_memory_footprint_file(destination_filename_for_compact_msi_data)