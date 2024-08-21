import logging
import pickle
from termcolor import colored

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def deserialize(file):
    f = open(file, 'rb')
    obj = pickle.load(f)
    return obj

def serialize(file, item):
    pkl = open(file, 'wb')
    pickle.dump(item, pkl)
    pkl.close
    logger.info(colored('Object serialized at {}'.format(file), 'green'))