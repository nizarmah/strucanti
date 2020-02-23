import re

import sys
from os import path

import pandas as pd

def tocsv_html_hcadb(html_file):
	"""
	Converts HTML Table File to CSV

	- Takes HTML_FILE_PATH as parameter
	- Saves Output to CURRENT_PATH/HTML_FILENAME.csv
	"""
	HEADER_PATTERN = r'(?<=<th>)(.*?)(?=<\/th>)'

	ENTRY_PATTERN = r'(?:(?:<td)(?:.*?>){1,2}(.*?)(?:<\/.*?>){0,})' \
						+ r'(?:<\/td>)(?:\s?<\/tr>(\n)?)?'

	csv_file = path.splitext(path.basename(html_file))[0] + '.csv'

	csv_header = ''
	csv_data = []

	with open(html_file, 'r') as file:
		file_content = file.read()

		csv_header = ','.join(re.findall(HEADER_PATTERN, file_content)) + '\n'

		csv_data = ''.join([ (val + ((',' if brk == '' else brk))) 
			for val, brk in re.findall(ENTRY_PATTERN, file_content) ])

	with open(csv_file, 'w+') as file:
		file.write(csv_header)

		file.write(csv_data)

	print(pd.read_csv(csv_file))

if __name__ == '__main__':
	try:
		tocsv_html_hcadb(sys.argv[1])
	except:
		print('Error : Missing HTML File Argument')
		print('Usage : python ', path.basename(__file__), ' file.html')