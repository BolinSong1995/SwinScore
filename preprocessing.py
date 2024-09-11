import os 
import numpy as np
import pandas as pd
from parameters import mkdirs
import pickle
from medpy.io import load

def write_list(L, file):
    # store list in binary file so 'wb' mode
    with open(file, 'wb') as fp:
        pickle.dump(L, fp)
        print('Done writing list into a binary file')

def build_ct_boxes(path):
    df = pd.read_csv(os.path.join(path,'data_table.csv'))
    for label_name in ['label.nii.gz', 'node.nii.gz']:
        X_min, X_max, Y_min, Y_max, Z_min, Z_max = [], [], [], [], [], []
        for patient in df["radiology_folder_name"]:
            print(os.path.join(path, patient, label_name))
            if os.path.isfile(os.path.join(
                    path, patient, label_name)):
                print("it is file")
                label, _ = load(os.path.join(
                        path, patient, label_name))
                x_min, x_max = np.min(np.nonzero(label)[1]), np.max(
                    np.nonzero(label)[1])
                y_min, y_max = np.min(np.nonzero(label)[0]), np.max(
                    np.nonzero(label)[0])
                z_min, z_max = np.min(np.nonzero(label)[2]), np.max(
                    np.nonzero(label)[2])
            else:
                x_min, x_max = 0, 0
                y_min, y_max = 0, 0
                z_min, z_max = 0, 0
            X_min.append(x_min)
            Y_min.append(y_min)
            Z_min.append(z_min)
            X_max.append(x_max)
            Y_max.append(y_max)
            Z_max.append(z_max)
        if label_name == "label.nii.gz":
            df["X_min_tumor"] = X_min
            df["Y_min_tumor"] = Y_min
            df["X_max_tumor"] = X_max
            df["Y_max_tumor"] = Y_max
            df["Z_min_tumor"] = Z_min
            df["Z_max_tumor"] = Z_max

        else:
            df["X_min_lymph"] = X_min
            df["Y_min_lymph"] = Y_min
            df["X_max_lymph"] = X_max
            df["Y_max_lymph"] = Y_max
            df["Z_min_lymph"] = Z_min
            df["Z_max_lymph"] = Z_max

    df.to_csv(os.path.join(path, 'data_table_new.csv'))

    return

path = r"C:\Users\bsong47\Documents\p16_positive_opc_ct\radiology"

build_ct_boxes(path)