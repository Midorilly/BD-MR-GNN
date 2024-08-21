from gensim.models.word2vec import Word2Vec
import logging
import argparse
import os
from utils.utils import *
import pandas as pd
from termcolor import colored
import numpy as np
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

#device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# default parameters
params = {
    'min_count' : 0,
    'window' : 10,
    'sg' : 1,
    'seed' : 123,
    'workers' : 5
}

'''def remove_duplicates(test):

    id_duplicates = test[test.duplicated(subset=['id'], keep=False)]
    print(len(id_duplicates))

    new_id = test['id'].max() + 1
    tbc_list = id_duplicates[id_duplicates['label'] == 0].iloc[:, :].index.tolist()
    for idx, data in test.iterrows():
        if idx in tbc_list:
            test.loc[idx, 'id'] = new_id
            new_id += 1

    return test'''

def trainw2v(datasetFile, modelPath):

    vectorSize = [128,256,512]
    fileName = os.fsdecode(datasetFile).split('.')[0]
    file = open(datasetFile)

    #for file in os.listdir(datasetPath):
        #fileName = os.fsdecode(file).split('.')[0]
        #f = open(os.path.join(datasetPath, file))
    corpus = pd.read_csv(file, header=0)['text_cleaned'].values.tolist()
    wordsList = []
    for sentence in corpus:
        wordsList.append(sentence.split(' '))
    logger.info(colored('[FILE] {}'.format(fileName), 'yellow'))
    for size in vectorSize:
        outputFolder = os.path.join(modelPath, str(size))
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder) # create a folder for each lemma
        logger.info(colored('[TRAINING] vector size {}'.format(size), 'yellow'))        
        model = Word2Vec(sentences=wordsList, min_count=params['min_count'], workers=params['workers'], 
                vector_size=size, window=params['window'], sg=params['sg'], epochs=1)
        model.save(os.path.join(outputFolder, 'model-'+str(size)+'.w2v'))
        logger.info(colored('[SAVING] model saved at {}'.format(os.path.join(outputFolder, 'model-'+str(size))), 'green'))

        sumVecs(datasetFile, modelPath, size)

def sumVecs(datasetFile, modelPath, size):
    file = open(datasetFile)
    outputFolder = os.path.join(modelPath, str(size))
    model = Word2Vec.load(os.path.join(outputFolder, 'model-'+str(size)+'.w2v')).wv
    corpus = pd.read_csv(file, header=0)
    text = corpus['text_cleaned'].values.tolist()
    dataset = pd.DataFrame()
    wordsList = []
    for sentence in text:
        wordsList.append(sentence.split(' '))
    sum = []
    missing = 0
    for sentence in wordsList:
        totalVector = np.zeros(int(size))
        for word in sentence:
            if word in model:
                totalVector += np.array(model[word])
            else:
                logger.info(colored('Word {} not in dictionary'.format(word), 'yellow'))
                missing += 1
        sum.append(totalVector.tolist())
    logger.info(colored('{} words not in dictionary'.format(missing), 'yellow'))

    serialize(os.path.join(outputFolder, 'ids-'+str(mode)+'-'+str(size)), corpus['id'].values.tolist())
    serialize(os.path.join(outputFolder, 'sums-'+str(mode)+'-'+str(size)), sum)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='train/test', required=False)
    parser.add_argument('-d', '--datasetPath', help='path to dataset', required=True)
    parser.add_argument('-w', '--modelPath', help='path to w2v model', required=True)
    parser.add_argument('-s', '--vectorSize', help='128, 256, 512', required=False)
    args = vars(parser.parse_args())
    mode=args['mode']
    datasetPath=args['datasetPath']
    modelPath=args['modelPath']
    size=args['vectorSize']

    if mode == 'train':
        trainw2v(datasetPath, modelPath)
    elif mode == 'test':
        if size != None:
            sumVecs(datasetPath, modelPath, size)
        else:
            logger.info(colored('Missing vector size parameter' ,'red'))

    