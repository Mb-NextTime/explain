import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data', help='path to the file with dataset')
parser.add_argument('target', help='index of the target column', type=int)
parser.add_argument('--header', action='store_true')
parser.add_argument('--model')

args = parser.parse_args()

# TODO: process all data and save results
