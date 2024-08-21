import scipy.io
import importlib
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd
import os
import argparse
from termcolor import colored
import logging
from utils.twitter import *
from utils.utils import *

def prepare_groundtruth_dataset(prefix):

    #train = pd.read_csv(os.path.join(prefix, 'train.csv'), header=0, usecols=['id', 'label'])
    #test = pd.read_csv(os.path.join(prefix, 'test.csv'), header=0, usecols=['id', 'label'])

    train = pd.read_csv(os.path.join(prefix, 'grande', 'full_dataset.csv'), header=0, usecols=['id', 'label']).drop_duplicates(ignore_index=True)
    test = pd.read_csv(os.path.join(prefix, 'piccolo', 'full_dataset.csv'), header=0, usecols=['id', 'label']).drop_duplicates(ignore_index=True)
    id_duplicates = test[test.duplicated(subset=['id'], keep=False)]
    idx_to_drop = set(id_duplicates[id_duplicates['label'] == 0].index.tolist())
    test = test.drop(index=idx_to_drop)

    # check users ids in train and test set

    #users_intersection = set(test['id']).intersection(set(train['id']))
    users_intersection = set(test['id']).symmetric_difference(set(train['id']))
    print('Symmetric difference of users in train and test: {}'.format(len(users_intersection)))
    if len(users_intersection) > 0:
        train = delete_users(users_intersection, train, [train['id'].values.tolist()], 'Train', 'Test')
        test = delete_users(users_intersection, test, [test['id'].values.tolist()], 'Test', 'Train')
        users_intersection = set(test['id']).symmetric_difference(set(train['id']))
        print('Symmetric difference of users in train and test after deletion: {}'.format(len(users_intersection)))

    groundtruth_users = set(test['id']).union(set(train['id']))
    train_users = set(train['id'])
    test_users = set(test['id'])
    print('Starting with unique users: {}\n'.format(len(groundtruth_users)))
    print('Starting with unique users in train: {}\n'.format(len(train_users)))
    print('Starting with unique users in test: {}\n'.format(len(test_users)))

    return train, test, groundtruth_users

def prepare_relations_dataset(prefix, groundtruth_users):

    #social = pd.read_csv(os.path.join(prefix, 'graph', 'social_network.edg'), sep='\t', header=None)
    frames = [pd.read_csv(os.path.join(prefix, 'grande', 'graph', 'social_network.edg'), sep='\t', header=None).drop_duplicates(ignore_index=True), 
                pd.read_csv(os.path.join(prefix, 'piccolo', 'graph', 'social_network.edg'), sep='\t', header=None).drop_duplicates(ignore_index=True)]
    social = pd.concat(frames, ignore_index=True)
    social[2] = np.ones(shape=len(social[0].values.tolist()), dtype=np.float64)

    #spatial = pd.read_csv(os.path.join(prefix, 'graph', 'spatial_network.edg'), sep='\t', header=None)
    frames = [pd.read_csv(os.path.join(prefix, 'grande', 'graph', 'spatial_network.edg'), sep='\t', header=None).drop_duplicates(ignore_index=True), 
                pd.read_csv(os.path.join(prefix, 'piccolo', 'graph', 'spatial_network.edg'), sep='\t', header=None).drop_duplicates(ignore_index=True)]
    spatial = pd.concat(frames, ignore_index=True)

    # remove users not in train and test
    social_columns = [social[0].values.tolist(), social[1].values.tolist()]
    spatial_columns = [spatial[0].values.tolist(), spatial[1].values.tolist()]
    social = delete_users(groundtruth_users, social, social_columns, 'Social', 'Groundtruth')
    spatial = delete_users(groundtruth_users, spatial, spatial_columns, 'Spatial', 'Groundtruth')

    social = adjust_relation_users(social, groundtruth_users, 0.0)
    spatial = adjust_relation_users(spatial, groundtruth_users, 0.0)

    return social, spatial

def adjust_relation_users(df, groundtruth_users, value):

    groundtruth_df0_difference = groundtruth_users.difference(set(df[0]))
    groundtruth_df1_difference = groundtruth_users.difference(set(df[1]))
    # add users in groundtruth and NOT in Social
    df = add_rows(df, groundtruth_df0_difference, groundtruth_df1_difference, value)
    print('Symmetric difference between Social columns: {}'.format(len(set(df[0]).symmetric_difference(set(df[1])))))
    print('Symmetric difference between Social columns and groundtruth users: {}\n'.format(len(set(df[0]).union(set(df[1])).symmetric_difference(groundtruth_users))))

    return df

if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasetFolder', help='path to dataset folder', required=True)
    parser.add_argument('-w', '--w2vFolder', help='path to w2v folder', required=True)
    parser.add_argument('-s', '--size', help='embedding size', required=True)
    parser.add_argument('-m', '--mapping', help='ids mapping', default=False, action='store_true')
    args = vars(parser.parse_args())
    prefix = args['datasetFolder']
    w2v = args['w2vFolder']
    mapping = args['mapping']
    size = args['size']

    twitter = {}

    train, test, groundtruth_users = prepare_groundtruth_dataset(prefix)
    social, spatial = prepare_relations_dataset(prefix, groundtruth_users)

    if mapping:
        users_map = map_users(prefix, groundtruth_users, 0, size)
        train.columns = ['original_id', 'label']
        train = add_mapped_users(train, [(train['original_id'].values.tolist(), 'id')], users_map)
        test.columns = ['label', 'original_id']
        test = add_mapped_users(test, [(test['original_id'].values.tolist(), 'id')], users_map)
        social.columns = ['original_0', 'original_1', 2]
        social = add_mapped_users(social, [(social['original_0'].values.tolist(), 0), (social['original_1'].values.tolist(), 1)], users_map)
        spatial.columns = ['original_0', 'original_1', 2]
        spatial = add_mapped_users(spatial, [(spatial['original_0'].values.tolist(), 0), (spatial['original_1'].values.tolist(), 1)], users_map)

    # LABELS 

    twitter['train_indexes'] = train['id'].values.tolist()
    twitter['train_labels'] = train['label'].values.tolist()
    twitter['test_indexes'] = test['id'].values.tolist()
    twitter['test_labels'] = test['label'].values.tolist()

    labels = twitter['train_labels'] + twitter['test_labels']
    indexes = twitter['train_indexes'] + twitter['test_indexes']
    logger.info(colored('Total instances: {}\n'.format(len(indexes)), 'green'))

    twitter['index'] = indexes
    twitter['label'] = np.array([labels])

    # RELATIONS

    ## Social
    row = np.array(social[0])
    col = np.array(social[1])
    data = np.array(social[2])
    ufu = csc_matrix((data, (row, col)))
    twitter['net_ufu'] = ufu
    ## Spatial
    row = np.array(spatial[0])
    col = np.array(spatial[1])
    data = np.array(spatial[2])
    distances = csc_matrix((data, (row, col)))
    twitter['net_spatial'] = distances

    # FEATURES

    ids = deserialize(os.path.join(w2v, 'ids-train-'+str(size))) + deserialize(os.path.join(w2v, 'ids-test-'+str(size)))
    sums = deserialize(os.path.join(w2v, 'sums-train-'+str(size))) + deserialize(os.path.join(w2v, 'sums-test-'+str(size)))

    row = []
    col = []
    data = []
    for id, sum in zip(ids, sums):
        if id in groundtruth_users:
            for i, s in enumerate(sum):
                if mapping:
                    row.append(users_map[id])
                else:
                    row.append(id)
                col.append(i)
                data.append(s)
    embeddings = csc_matrix((np.array(data), (np.array(row), np.array(col))))
    twitter['features'] = embeddings

    outputFolder = os.path.join(prefix, str(size))
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    serialize(os.path.join(outputFolder, 'dataset-'+size+'.dict'), twitter)
    #logger.info(colored('Dataset serialized at {}'.format(outputFolder), 'green'))