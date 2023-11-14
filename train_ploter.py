import os
from utils import get_event_data, get_data_frame, process_df, get_event_file_path_list, smooth
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm

mlp_log_dir     = os.path.join("./srpe/","srpe_results_and_viz","logs/","srpe-20230406-170724/test")
ode_log_dir     = os.path.join("./srpe/","srpe_results_and_viz","logs/","srpe-20230417-155740/test")
ode3_log_dir    = os.path.join("./srpe/","srpe_results_and_viz","logs/","srpe-20230511-221203/test")
runs = [mlp_log_dir, ode_log_dir, ode3_log_dir]

## read all tensorboard events
event_list = []
for run in runs:
    event_list += [os.path.join(run, i) for i in os.listdir(run) if i.startswith('events')]

label_list = ['Max_error', 'Mean_error']
tags = {
    'Max_error': 'error/dist_max',
    'Mean_error': 'error/dist_mean',
}
df_path = "./buffer/tensor_df.pkl"

if os.path.exists(df_path):
    df_list = pickle.load(open(df_path, "rb"))
    print("already exists")
else:
    df_list = []
    for event in tqdm(event_list):
        data = get_event_data(event, label_list, 20000, tags)
        steps = [event.step for event in data['Max_error']]
        values_max_error = [event.value for event in data['Max_error']]
        df = pd.DataFrame({'step': steps, 'Max_error': values_max_error, 'Mean_error': [event.value for event in data['Mean_error']]})
        df = process_df(df, 'Max_error', 0, 20000)
        df = process_df(df, 'Mean_error', 0, 20000)
        df = pd.DataFrame({'step': df['step'], 'Max_error': df['Max_error_smooth'], 'Mean_error': df['Mean_error_smooth']})
        df_list.append(df)
    pickle.dump(df_list, open(df_path, "wb"))


alg_list = ['Baseline', 'CTPO', 'PI-CTPO']
color_list = ['blue', 'orange', 'green']
fig_size = (8,4)


def plot_1():
    ## font scale
    sns.set(font_scale=1.5)
    ## font style
    sns.set_style("ticks")
    ## times new roman
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)
    for i, df in enumerate(df_list):
        df["smoothed"] = smooth(df["Max_error"],250)
        sns.lineplot(data=df, x="step", y="Max_error", 
                     ax=ax, 
                     label=alg_list[i], 
                     errorbar=("se", 2),
                     alpha=0.5,
                     )
        sns.lineplot(data=df, x="step", y="smoothed", 
                ax=ax, 
                errorbar=("se", 2),
                )
        lines = ax.get_lines()[-2:]
        for line in lines:
            line.set_color(color_list[i])
    ax.set_xlim((5, 20000))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('')
    ax.set_title("Max error (m)", loc="left")
    ## log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    
    ## save figure
    fig.savefig('max_error.pdf', bbox_inches='tight')


def plot_2():
    ## font scale
    sns.set(font_scale=1.5)
    ## font style
    sns.set_style("ticks")
    ## times new roman
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    for i, df in enumerate(df_list):
        df["smoothed"] = smooth(df["Mean_error"],250)
        sns.lineplot(data=df, x="step", y="Mean_error", 
                     ax=ax, 
                     label=alg_list[i], 
                     errorbar=("se", 2),
                     alpha=0.5,
                     )
        sns.lineplot(data=df, x="step", y="smoothed", 
                ax=ax, 
                errorbar=("se", 2),
                )
        lines = ax.get_lines()[-2:]
        for line in lines:
            line.set_color(color_list[i])
    ax.set_xlim((5, 20000))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('')
    ax.set_title("Mean error (m)", loc="left")
    # log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ## save figure
    fig.savefig('mean_error.pdf', bbox_inches='tight')

if __name__ == "__main__":
    plot_1()
    plot_2()
    pass