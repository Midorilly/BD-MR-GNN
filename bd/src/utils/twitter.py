import importlib
import numpy as np
import pandas as pd
import os
import pickle
from termcolor import colored
import logging
from utils.utils import *
import random
import itertools

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_users(unique_users, df, columns, dataset1, dataset2):

    logger.info(colored('Deleting users from {} network not in {}'.format(dataset1, dataset2), 'blue'))

    indexes = []
    for c in columns:
        for i, item in enumerate(c):
            if item not in unique_users:
                indexes.append(i)

    indexes_set = set(indexes)

    df = df.drop(index=indexes_set)

    logger.info(colored('{} rows deleted'.format(len(indexes_set)), 'green'))

    return df

def map_users(prefix, users, i, size):
    
    print('Mapping {} users'.format(len(users)))
    #ids = sorted(users)
    mapped_users = {}
    for id in users:
        mapped_users[id] = i
        i += 1
    serialize(os.path.join(prefix, str(size), 'users_map.dict'), mapped_users)

    return mapped_users

def add_mapped_users(df, columns, mapping):

    '''for ids, index in users:
        map = []
        for id in ids:
            map.append(mapping[id])
        df[index] = map'''

    for column, header in columns:
        map = []
        for id in column:
            map.append(mapping[id])
        df[header] = map
    
    return df

def add_rows(dataset, col0, col1, value):

    logger.info(colored('Adding rows...', 'blue'))

    '''if len(col0) > len(col1):
        col0 = list(col0)
        col1 = list(col1)*int((np.round(len(col0)/len(col1))+1))
        col1 = list(col1[:len(col0)])
    elif len(col0) < len(col1):
        col0 = list(col0)*int((np.round(len(col1)/len(col0))+1))
        col0 = list(col0[:len(col1)])
        col1 = list(col1)
    elif len(col0) == len(col1):
        col0 = list(col0)
        col1 = list(col1)'''

    col0_diff = list(col0.difference(col1))
    col1_diff = list(col1.difference(col0))
    inter = list(col0.intersection(col1))

    if len(col0_diff) > len(col1_diff):
        col0 = col0_diff + inter
        col1 = list(inter+col1_diff)*int((np.round(len(col0)/len(col1))+1))
        col1 = col1[:len(col0)]
    elif len(col0) < len(col1):
        col0 = list(col0)*int((np.round(len(col1)/len(col0))+1))
        col0 = col0[:len(col1)]
        col1 = list(col1)
    else:
        col0 = list(col0_diff + inter)
        col1 = list(inter + col1_diff)

    data = []

    count = 0
    for idx0, idx1 in zip(col0, col1):
        if idx0 == idx1:
            count += 1
        data.append((idx0, idx1))

    logger.info(colored('{} identical rows'.format(count), 'red'))
    data_0 = [idx0 for (idx0, idx1) in data]
    data_1 = [idx1 for (idx0, idx1) in data]    
    data_2 = [value] * len(data_0)

    data = {0 : data_0, 1 : data_1,  2: data_2}
    df = pd.DataFrame(data)

    frames = [dataset, df]
    d = pd.concat(frames, ignore_index=True)
    logger.info(colored('{} rows added\n'.format(len(data_0)), 'green'))

    return d

def dataset_stats(dataset, name):

    one = 0
    zero = 0
    for x in dataset:
        if x == 0.0:
            zero += 1
        elif x == 1.0:
            one += 1

    logger.info(colored('{}\n{} examples, of which\n{} 1 labels\n{} 0 labels\n'.format(name.upper(), str(one+zero), one, zero), 'yellow'))