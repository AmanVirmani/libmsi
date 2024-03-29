import os
import random
import pickle
from dataloader import DataLoader
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearnex import patch_sklearn
patch_sklearn()

def generate_patches(nmf_data, data_list, patch_size=(15, 15), random_patches=True, random_state=42, n_patches=1000):
    patches = []
    curr = 0
    for file in data_list:
        nmf_data_3d = file.reconstruct_3d_array(nmf_data[curr:curr + len(file.imzml_2d_array)])
        curr += len(file.imzml_2d_array)

        if random_patches:
            random.seed(random_state)
            count_patch = 0
            while count_patch < n_patches:
                row = random.randint(0, nmf_data_3d.shape[0])
                col = random.randint(0, nmf_data_3d.shape[1])
                patch = nmf_data_3d[row:row + patch_size[0], col: col + patch_size[1]]
                if patch.shape[:2] != patch_size or not np.count_nonzero(patch, axis=2).all():
                    continue
                flat_patch = np.ravel(patch)
                patches.append(flat_patch)
                count_patch += 1
        else:
            for row in range(0, nmf_data_3d.shape[0], patch_size[0]):
                for col in range(0, nmf_data_3d.shape[1], patch_size[1]):
                    patch = nmf_data_3d[row:row + patch_size[0], col: col + patch_size[1]]
                    if patch.shape[:2] != patch_size or not np.count_nonzero(patch, axis=2).all():
                        continue
                    flat_patch = np.ravel(patch)
                    patches.append(flat_patch)

    return np.array(patches, dtype=np.float64)

def generate_patch_data(nmf_data, imzml_data, label_idx, patch_size=(15, 15), random_patches=True, random_state=42, n_patches=1000):
    train_patches = generate_patches(nmf_data, imzml_data.train_data, patch_size=(15, 15), random_patches=True, random_state=42, n_patches=1000)
    test_patches = generate_patches(nmf_data, imzml_data.val_data, patch_size=(15, 15), random_patches=True, random_state=42, n_patches=1000)

    if label_idx:
        train_labels = np.ones(len(train_patches))
        test_labels = np.ones(len(test_patches))
    else:
        train_labels = np.zeros(len(train_patches))
        test_labels = np.zeros(len(test_patches))

    return train_patches, train_labels, test_patches, test_labels

def fit_best_svm(patches, labels, test_size=None, param_grid=None):
    if param_grid is None:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(patches, labels, test_size=0.2, random_state=42)
    else:
        X_train, y_train = patches, labels
    grid = GridSearchCV(SVC(), param_grid, n_jobs=-1, refit=True, scoring='accuracy', verbose=2)
    grid.fit(X_train, y_train)

    print('best_estimator', grid.best_estimator_)

    if test_size is not None:
        grid_predictions = grid.predict(X_test)
        # print(confusion_matrix(y_test,grid_predictions))
        print(classification_report(y_test, grid_predictions))
    return grid


if __name__ == '__main__':
    data_path = "/mnt/sda/avirmani/data"
    cph_paths = [
                    "DMSI0047_F893CPH_LipidPos_IMZML/dmsi0047_0_05_v2.pickle",
                    "DMSI0048_F894CPH_LipidPos_IMZML/dmsi0048_0_05_v2.pickle",
                    "DMSI0053_F895CPH_LipidPos_IMZML/dmsi0053_0_05_v2.pickle",
                    "DMSI0054_F896CPH_LipidPos_IMZML/dmsi0054_0_05_v2.pickle",
                    ]
    naive_paths = [
                    "DMSI0045_F885naive_LipidPos_IMZML/DMSI0045_0_05_v2.pickle",
                    "DMSI0046_F886naive_LipidPos_IMZML/dmsi0046_0_05_v2.pickle",
                    "DMSI0049_F887naive_LipidPos_IMZML/dmsi0049_0_05_v2.pickle",
                    "DMSI0068_F888naive_LipidPos_IMZML/dmsi0068_0_05_v2.pickle",
                   ]

    msi_cph = DataLoader([os.path.join(data_path, cph) for cph in cph_paths], 0.75)
    msi_naive = DataLoader([os.path.join(data_path, naive) for naive in naive_paths], 0.75)
    
    nmf_file = "/mnt/sda/avirmani/libmsi/scripts/pos_nmf_20_all_0_05_v2.pickle"
    nmf_data_list = pickle.load(open(nmf_file, 'rb'))['nmf_data_list']
    nmf_naive_idx = [0,1,4,7]
    nmf_cph_idx = [2,3,5,6]
    cph_nmf_data_list = [nmf_data_list[idx] for idx in nmf_cph_idx]
    naive_nmf_data_list = [nmf_data_list[idx] for idx in nmf_naive_idx]
    ## perform NMF Regression
    n_comp_list = np.arange(20,21)
    for n_comp in n_comp_list:
        cph_nmf_data = np.vstack(tuple(cph_nmf_data_list))
        naive_nmf_data = np.vstack(tuple(naive_nmf_data_list))

        ## make roi of dimension (15, 15)
        patch_dim = (15, 15)
        cph_train_patches, cph_train_labels, cph_test_patches, cph_test_labels = generate_patch_data(cph_nmf_data, msi_cph, 1)
        naive_train_patches, naive_train_labels, naive_test_patches, naive_test_labels = generate_patch_data(naive_nmf_data, msi_naive, 0)

        # perform svm
        fn = f'patch_{patch_dim[0]}_nmf_pos_{n_comp}_train_75_0.05.npy'
        train_patches = np.vstack([cph_train_patches, naive_train_patches])
        train_labels = np.hstack([cph_train_labels, naive_train_labels])
        test_patches = np.vstack([cph_test_patches, naive_test_patches])
        test_labels = np.hstack([cph_test_labels, naive_test_labels])
        # np.save(fn, {'patches': [train_patches, test_patches], 'labels': [train_labels, test_labels]})
        # data = np.load('patches_nmf_5_comp.npy', allow_pickle=True)[()]
        # patches = data['patches']
        # labels = data['labels']
        param_grid = {'C': np.arange(8.41, 8.43, 0.001), 'gamma': np.arange(0.79, 0.81, 0.001),
                      'kernel': ['rbf']}  # , 'poly', 'sigmoid']}
        param_grid = {'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        grid = fit_best_svm(train_patches, train_labels, param_grid=None)
        # np.save(fn, {'patches': [train_patches, test_patches], 'labels': [train_labels, test_labels], "svm_grid": grid})

        ## get accuracy
        acc = grid.score(test_patches, test_labels)
        print('accuracy: ', acc)
    pass
