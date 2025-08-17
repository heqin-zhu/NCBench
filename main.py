import json
import pickle

import numpy as np

from src.NCfold.utils.data_processing import prepare_dataset_RNAVIEW_pickle


if __name__ == '__main__':
    data_path = 'data/NC_data.json'
    dest = 'data/NC_data.pickle'
    prepare_dataset_RNAVIEW_pickle(dest, data_path)
