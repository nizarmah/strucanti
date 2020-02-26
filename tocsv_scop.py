import re

import sys
from os import path

import pandas as pd

def flatten_scop_tuple(scop_tuple):
	"""
	Flattens SCOP Tuple

	Tuple consists of :
	- sid
	- class
	- folds
	- superfamilies
	- families
	- sequence

	Note : Sequence contains (\\n). Therefore need to flatten it out too.
	"""
	scop_list = list(scop_tuple)

	scop_list[5] = scop_list[5].replace('\n', '')
	scop_list[5] = scop_list[5].lower()

	return ','.join(scop_list) + '\n'

def tocsv_scop(scop_file):
	"""
	Converts Fasta File to CSV

	- Takes FASTA_FILE_PATH as parameter
	- Saves Output to CURRENT_PATH/FASTA_FILENAME.csv
	"""
	SEQUENCE_PATTERN = r'(?<=>)(\S+)(?:\s)(\w+)\.(\d+)\.(\d+)\.(\d+)' \
						+ r'(?:.*\n)([\w\s]*)'

	scop_cols = ['sid', 'class', 'folds', 'superfamilies', 
					'families', 'sequence']

	csv_file = path.splitext(path.basename(scop_file))[0] + '.csv'

	csv_header = ''
	csv_data = []

	with open(scop_file, 'r') as file:
		file_content = file.read()

		csv_header = ','.join(scop_cols) + '\n'

		csv_data = ''.join([ flatten_scop_tuple(scop_tuple) 
			for scop_tuple in re.findall(SEQUENCE_PATTERN, file_content) ])

	with open(csv_file, 'w+') as file:
		file.write(csv_header)

		file.write(csv_data)

	print(pd.read_csv(csv_file))

if __name__ == '__main__':
	try:
		tocsv_scop(sys.argv[1])
	except:
		print('Error : Missing SCOPe File Argument')
		print('Usage : python ', path.basename(__file__), ' file.fa')