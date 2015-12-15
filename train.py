from __future__ import division

import csv
import numpy as np

# Open CSV file and read header
csv_file = csv.reader(open('train.csv', 'rb')) 
header = csv_file.next()

# Read the remaining line by line
data = []
for row in csv_file:
    data.append(row)
data = np.array(data) 

# Compute the proportion of survivors
num_passengers = np.size(data[0::,1].astype(np.float))
num_survived = np.sum(data[0::,1].astype(np.float))
proportion_survived = num_survived / num_passengers

# Find indices of each gender
position_women = data[0::,4] == 'female'
position_men = data[0::,4] == 'male'

num_women = np.sum(position_women)
num_men = np.sum(position_men)
num_other = num_passengers - num_women - num_men

proportion_women = num_women / num_passengers
proportion_men = num_men / num_passengers

# Compute the proportion of survivors by gender
proportion_survived_women = np.sum(data[position_women,1].astype(np.float)) / num_women
proportion_survived_men = np.sum(data[position_men,1].astype(np.float)) / num_men
print("Proportion of survivors by gender. Women: {:.0%}. Men: {:.0%}.".format(proportion_survived_women, proportion_survived_men))
