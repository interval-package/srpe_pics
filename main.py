import pickle
import os
import numpy as np
import seaborn  as sb

baseDir = "home/molumitu/code/srpe/srpe_results_and_viz"

if __name__ == "__main__":

    fileNames = [
        "xyz_batch_mlp_error.pkl",
        "xyz_batch_ode_error.pkl",
        "xyz_batch_ode3_error.pkl"
    ]

    for fileName in fileNames:
        with open(os.path.join(baseDir, fileName), "rb") as f:
            arr = pickle.load(f)
            print(arr)
            pass
    
    pass