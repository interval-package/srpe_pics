import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import os

data_dir = "srpe/srpe_results_and_viz/"

xyz_batch_mlp_error     = pickle.load(open(os.path.join(data_dir, "xyz_batch_mlp_error.pkl"), "rb"))
xyz_batch_ode_error     = pickle.load(open(os.path.join(data_dir, "xyz_batch_ode_error.pkl"), "rb"))
xyz_batch_ode3_error    = pickle.load(open(os.path.join(data_dir, "xyz_batch_ode3_error.pkl"), "rb"))



# (142,186,215)
# (254,190,136)
# (148,205,150)


## font scale
sns.set(font_scale=1.5)
## font style
sns.set_style("ticks")
## times new roman
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
color_list = ['#8EBAD7', '#FEBE88', '#94CD96']
bin_num = 60
ax.hist(xyz_batch_mlp_error.flatten(), 
        bins=bin_num, label='MLP', alpha=0.7, density=True, log=True,
        color=color_list[0],
        edgecolor=None,
        histtype="stepfilled"
        )
ax.hist(xyz_batch_ode_error.flatten(), 
        bins=bin_num, label='CTPO', alpha=0.7, density=True, log=True,
        color=color_list[1],
        edgecolor=None,
        histtype="stepfilled"
        )
ax.hist(xyz_batch_ode3_error.flatten(), 
        bins=bin_num, label='PI-CTPO ', alpha=0.7, density=True, log=True,
        color=color_list[2],
        edgecolor=None,
        histtype="stepfilled"
        )
ax.set_xlabel('Distance error [m]', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
ax.legend(fontsize=16)
## ticks font size
ax.tick_params(axis='both', which='major', labelsize=12)

ax.set_ylim((1, 10000))
ax.set_xlim((0, 0.0025))
plt.tight_layout()
fig.savefig('compare.pdf', bbox_inches='tight')

