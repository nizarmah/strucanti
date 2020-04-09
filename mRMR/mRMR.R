#!/usr/bin/env Rscript

if (!require(mRMRe)) install.packages('mRMRe')
library(mRMRe)

args = commandArgs(trailingOnly=TRUE)

if (length(args) == 0) {
	print('Error : Missing Arguments')
	print('Usage : Rscript RMR.R dataset.csv feature_count')
	stop()
}

dataset_path <- args[1]
number_features <- as.numeric(args[2])

# use check.names=FALSE in order to avoid column renaming :
# 	eg. '3_h' -> 'X3_h'
dataset <- read.csv(dataset_path, check.names=FALSE)

# data columns must be of type numeric for mRMR.data
# but column class is of type integer, so we transform
dataset$class <- as.numeric(dataset$class)

# dataset should be of the form mRMR.dat
mrmr_data <- mRMR.data(data=dataset)

# we suspect the target index to be 1
mrmr <- mRMR.classic(data=mrmr_data, target_indices=c(1),
			  	feature_count=number_features)

cat(mrmr@feature_names[mrmr@filters$`1`], sep='\n')
