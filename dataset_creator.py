import re

import sys
from os import path

import numpy as np
import pandas as pd

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
	def __init__(self, binary_protein_series):
		self.binary_protein_series = binary_protein_series

	def find_record(self, binary_seq):
		"""
		Searches (ProteinCode_Affinity : Binary Sequence) Series
		For certain Binary Sequence, in order to retrieve ProteinCode_Affinity
		"""
		record = self.binary_protein_series[self.binary_protein_series
															== binary_seq]

		if len(record) == 1:
			return (record.index[0], record.iloc[0])
		else: return None

	def enumerate(self, row):
		"""
		Applies Row Transformation for Encoded Sequence DataFrame
		"""	
		binary_seqs = re.split('(0*p0*)|(0{4,})', row['encoded_sequence'])

		for binary_seq in binary_seqs:
			protein_record = self.find_record(binary_seq)

			if protein_record != None:
				row[protein_record[0]] += 1

		return row


def create_dataset(hcadb_path, scop_path):
	"""
	Creates DataSet from 
	- Dictionary Path
	- SCOPe Sequence Set Path
	"""
	scop_df = pd.read_csv(scop_path)
	hcadb_df = pd.read_csv(hcadb_path)

	amino_encoder = AminoEncoder()

	# drop all entries that contain 'x' in their sequence
	scop_df = scop_df[[ ('x' not in i) for i in scop_df.sequence ]]

	# encode sequences into a new series
	encoded_seqs_series = scop_df.sequence.map(amino_encoder.encode) \
											.rename('encoded_sequence')

	# get all protein codes and affinities
	protein_codes_series = (hcadb_df['P-code'].astype(str)
							+ '_' + hcadb_df['affinity'])

	# combine protein_codes_series with their binary protein sequences
	binary_protein_series = hcadb_df['Binary'].astype(str)
	binary_protein_series.index = protein_codes_series.values

	protein_codes_df = binary_protein_series.to_frame().T[0:0]
	encoded_seqset_df = pd.concat([encoded_seqs_series, 
									protein_codes_df], axis=1)
	encoded_seqset_df.fillna(0, inplace=True)

	encoded_seq_enumerator = EncodedSequenceEnumerator(binary_protein_series)
	encoded_dataset_df = encoded_seqset_df.apply(
							encoded_seq_enumerator.enumerate, axis='columns')

	dataset_df = pd.concat([scop_df, 
								encoded_dataset_df], axis=1)

	scop_filename = path.splitext(path.basename(scop_path))[0]
	dataset_file = 'dataset_' + scop_filename + '.csv'
	dataset_df.to_csv(dataset_file, index=False)

if __name__ == '__main__':
	try:
		create_dataset(sys.argv[1], sys.argv[2])
	except:
		print('Error : Missing Arguments')
		print('Usage : python ', path.basename(__file__), 
			' hcadb.csv scop.csv')