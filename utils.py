from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
from typing import List, Dict
import os
def get_event_data(log_file: str, label_list: list, max_num:int, tags: Dict) -> Dict:
    """Returns log files for label list"""

    load_config = {
        'scalars': max_num,
    }

    event_acc = EventAccumulator(log_file, load_config)
    event_acc.Reload()

    event_data_dict = {}
    for label in label_list:
        label_log = tags[label]
        assert label_log in event_acc.Tags()["scalars"], "Selected label: {} does not exist in the list of selectable labels:\n {}".format(label, event_acc.Tags()["scalars"])
        event_data_dict[label] = event_acc.Scalars(label_log)
        
    return event_data_dict

def get_data_frame(event_data: list, label: str, num: int=None):
    """Returns data frame from event data"""
    steps = [event.step for event in event_data]
    values = [event.value for event in event_data]
    if num is not None:
        steps = steps[:num]
        values = values[:num]
    df = pd.DataFrame({'step': steps, label: values})
    return df

def smooth(x, width=1):
    y = np.ones(width)
    z = np.ones(len(x))
    return np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    

def process_df(df: pd.DataFrame, label: str, start: int, end: int):
    # interpolation
    # new_step = np.linspace(int(start), int(end), 100)
    # new_value = np.interp(new_step, df['step'], df[label])
    # df = pd.DataFrame({'step': new_step, label: new_value})

    ## smooth
    smoothed_value = smooth(df[label])
    # smoothed_value = df[label]
    df.insert(loc=len(df.columns), column=f'{label}_smooth', value=smoothed_value)
    return df

def get_event_file_path_list(resource_path, test_case_seeds):
    event_file_path_list = []
    for test_case in test_case_seeds:
        event_file = [event for event in os.listdir(os.path.join(resource_path, test_case)) if 'events' in event][0]
        event_file_path = os.path.join(resource_path, test_case, event_file)
        event_file_path_list.append(event_file_path)
    return event_file_path_list
