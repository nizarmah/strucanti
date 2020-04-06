import re

import sys
from os import path

import numpy as np
import pandas as pd

from functools import partial
from multiprocessing import Pool, cpu_count

class HCADB(object):
	"""
	Creates HCADB object out of HCADB CSV Path
	Also creates a Dictionary Mapping Key -> Value
		- Key   : Encoded Representation of Protein 
		- Value : Protein Name

	> Note : the Value represents the protein column name in the dataset
			 it is formed through concatting Protein Code and Affinity
	"""
	def __init__(self, hcadb_path):
		self.df = pd.read_csv(hcadb_path)

		self.protein_names = (self.df['P-code'].astype(str)
								+ '_' + self.df['affinity']).values

		self.encoded_proteins = self.df['Binary'].astype(str).values

		self.proteins_dict = dict(zip(self.encoded_proteins, 
										self.protein_names))

	def get_protein_name(self, binary_seq):
		if binary_seq in self.proteins_dict:
			return self.proteins_dict[binary_seq]

		return None

	def protein_names_empty_dataset(self, num_rows):
		"""
		Creates a DataFrame with Columns as Protein Name filled with 0s
		"""
		return pd.DataFrame(np.zeros((num_rows, len(self.protein_names))), 
								columns=self.protein_names)

class SCOP(object):
	"""
	Creates SCOP object out of SCOP CSV Path
	Also creates a DataFrame of all Sequences in the SCOP CSV file
	"""
	def __init__(self, scop_path):
		self.df = pd.read_csv(scop_path)

		self.num_rows = self.df.shape[0]

	def join(self, other_df):
		self.df = self.df.join(other_df)

	def distribute(self, func, num_processes):
		self.num_processes = num_processes

		self.chunksize = (self.num_rows // self.num_processes)

		with Pool(self.num_processes) as pool:
			return pool.map(func, range(num_processes))

	def get_segment(self, process_id):
		starting_index = (self.chunksize * process_id)
		next_starting_index = (self.chunksize * (process_id + 1))

		return (self.df.iloc[starting_index:] \
				if ((process_id + 1) == self.num_processes) \
					else self.df.iloc[starting_index:next_starting_index])\
						.copy(deep=True)

class AminoEncoder(object):
	"""
	Encodes certain amino acids as 1,
	Others are reserved amino acids that stay unchanged,
	And the amino acids left are changed into 0
	"""
	def __init__(self, 
		encoded_aminos=['v', 'i', 'l', 'm', 'y', 'w', 'f'],
		reserved_aminos=['p']):
		self.VALUE_ONE_PATTERN = re.compile(r'[' \
										+ (''.join(encoded_aminos)) + r']')
		self.VALUE_ZERO_PATTERN = re.compile(r'[^\d' \
										+ (''.join(reserved_aminos)) + r']')

	def encode(self, sequence):
		"""
		Maps : Sequence -> Encoded Sequence
		"""
		sequence = re.sub(self.VALUE_ONE_PATTERN, '1', sequence)
		sequence = re.sub(self.VALUE_ZERO_PATTERN, '0', sequence)

		return sequence

class EncodedSequenceEnumerator(object):
	"""
	Takes in:
	- Series (ProteinCode_Affinity : Binary Sequence)

	The series is used in order to find ProteinCode_Affinity
	According to the Binary Sequences extracted from the Encoded Sequence
	"""
	def __init__(self, hcadb):
		self.hcadb = hcadb

	def enumerate(self, row):
		"""
		Applies Row Transformation for Encoded Sequence DataFrame
		"""	
		binary_seqs = re.split('(0*p0*)|(0{4,})', row['encoded_sequence'])

		for binary_seq in binary_seqs:
			protein_name = self.hcadb.get_protein_name(binary_seq)

			if protein_name != None:
				row[protein_name] += 1

		return row

def enumerate_sequences(hcadb, scop, process_id):
	"""
	Takes a segment of the DataFrame and counts proteins
	in sequences after filtering the sequences and encoding them
	"""
	df = scop.get_segment(process_id)

	amino_encoder = AminoEncoder()

	# encode sequences into a new series
	df['encoded_sequence'] = df.sequence.map(amino_encoder.encode)

	encoded_seq_enumerator = EncodedSequenceEnumerator(hcadb)
	df = df.apply(encoded_seq_enumerator.enumerate, axis='columns')

	return df.drop('encoded_sequence', 1)

def create_dataset(hcadb_path, scop_path, num_processes=1):
	"""
	Creates DataSet from 
	- Dictionary Path
	- SCOPe Sequence Set Path
	"""
	hcadb = HCADB(hcadb_path)
	
	scop = SCOP(scop_path)
	scop.join(hcadb.protein_names_empty_dataset(scop.num_rows))

	dataset_distributed = scop.distribute(
								partial(enumerate_sequences, hcadb, scop), 
								num_processes)

	dataset_df = pd.concat(dataset_distributed)

	scop_filename = path.splitext(path.basename(scop_path))[0]
	dataset_file = 'dataset_' + scop_filename + '.csv'
	dataset_df.to_csv(dataset_file, index=False)

def start_script():
	"""
	Reads Arguments from CMD
	And Starts the Script Accordingly
	"""
	num_processes = cpu_count() - 1
	hcadb_path = sys.argv[1]
	scop_path = sys.argv[2]

	if sys.argv[1] == '--processes':
		num_processes = sys.argv[2]

		hcadb_path = sys.argv[3]
		scop_path = sys.argv[4]

	create_dataset(hcadb_path, scop_path, num_processes=num_processes)

if __name__ == '__main__':
	try:
		start_script()
	except:
		print('Error : Missing Arguments')
		print('Usage : python ', path.basename(__file__), 
			' hcadb.csv scop.csv')