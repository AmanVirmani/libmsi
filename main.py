import libmsi

if __name__ == "__main__":
    file = "../data/DMSI0003_F889crd_LipidNeg_IMZML/DMSI0003_F889crd_LipidNeg_IMZML/dmsi0003scilsexporttest.imzML"
    msi = libmsi.Imzml(file,200)
    msi.get_peaks()
    pass