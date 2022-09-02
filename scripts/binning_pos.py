import os
import sys

from multiprocessing import Process


def bin_data(fn, bs):
    print('binning ', fn,' with binSize: ', bs)
    fs = libmsi.Imzml(fn, bs)
    save_fn = fn.replace('.imzML','_' +
            str(bs).replace('.','_')+ '_v2.pickle')
    fs.saveImzmlData(save_fn)



if __name__ == "__main__":
    sys.path.append("/mnt/sda/avirmani/libmsi")
    import libmsi

    data_path = "/mnt/sda/avirmani/data"

    all_paths = [
        "DMSI0045_F885naive_LipidPos_IMZML/DMSI0045.imzML",
        "DMSI0046_F886naive_LipidPos_IMZML/dmsi0046.imzML",
        "DMSI0047_F893CPH_LipidPos_IMZML/dmsi0047.imzML",
        "DMSI0048_F894CPH_LipidPos_IMZML/dmsi0048.imzML",
        "DMSI0049_F887naive_LipidPos_IMZML/dmsi0049.imzML",
        "DMSI0053_F895CPH_LipidPos_IMZML/dmsi0053.imzML",
        "DMSI0054_F896CPH_LipidPos_IMZML/dmsi0054.imzML",
        "DMSI0068_F888naive_LipidPos_IMZML/dmsi0068.imzML",
    ]
    binSize = 0.05

    t_list = []
    for path in all_paths:
        # bin_data(os.path.join(data_path, path), binSize)
        t_list.append(Process(target=bin_data,args=(os.path.join(data_path,
        path), binSize,)))

    for t in t_list:
        t.start()
    for t in t_list:
        t.join()

