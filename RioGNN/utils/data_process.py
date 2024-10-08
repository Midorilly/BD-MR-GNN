from utils import sparse_to_adjlist
from scipy.io import loadmat
import argparse
import os
import pickle

"""
	Read data and save the adjacency matrices to adjacency lists
	Paper: Reinforced Neighborhood Selection Guided Multi-Relational Graph Neural Networks
	Source: https://github.com/safe-graph/RioGNN
"""

def deserialize(file):
    f = open(file, 'rb')
    obj = pickle.load(f)
    return obj

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--name', help='dataset to prepare', required=True)
	parser.add_argument('-d', '--dataset', help='path to dataset', required=True)
	#parser.add_argument('-m', '--output', help='output path', required=True)
	args = vars(parser.parse_args())
	dataset = args['name']
	path = args['dataset']
	#output = args['output']

	prefix = '../bd/data/'

	if dataset == 'yelp':

		# Yelp
		yelp = loadmat('data/YelpChi.mat')
		net_rur = yelp['net_rur']
		net_rtr = yelp['net_rtr']
		net_rsr = yelp['net_rsr']
		yelp_homo = yelp['homo']

		sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
		sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
		sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
		sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')
	
	elif dataset == 'amazon':

		# Amazon
		amz = loadmat(prefix + 'Amazon.mat')
		net_upu = amz['net_upu']
		net_usu = amz['net_usu']
		net_uvu = amz['net_uvu']
		amz_homo = amz['homo']

		sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
		sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
		sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
		sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')

	elif dataset == 'mimic':

		# Mimic
		mic = loadmat(prefix + 'Mimic.mat')
		rel_vav = mic['rel_vav']
		rel_vdv = mic['rel_vdv']
		rel_vmv = mic['rel_vmv']
		rel_vpv = mic['rel_vpv']
		mic_homo = mic['homo']

		sparse_to_adjlist(rel_vav, prefix + 'mic_vav_adjlists.pickle')
		sparse_to_adjlist(rel_vdv, prefix + 'mic_vdv_adjlists.pickle')
		sparse_to_adjlist(rel_vmv, prefix + 'mic_vmv_adjlists.pickle')
		sparse_to_adjlist(rel_vpv, prefix + 'mic_vpv_adjlists.pickle')
		sparse_to_adjlist(mic_homo, prefix + 'mic_homo_adjlists.pickle')

	elif dataset == 'twitter':

		# Twitter
		twt = deserialize(path)
		net_ufu = twt['net_ufu']
		net_weigh = twt['net_spatial']
		output = os.path.dirname(path)
		sparse_to_adjlist(net_ufu, os.path.join(output, 'net_ufu_adjlists.pickle'))
		sparse_to_adjlist(net_weigh, os.path.join(output, 'net_spatial_adjlists.pickle'))

