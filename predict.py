from __future__ import division

import csv
import numpy as np

# Read training CSV file
with open('train.csv', 'rb') as train_file:
    train_obj = csv.reader(train_file)

    # Read header
    header_train = train_file.next()

    # Read the remaining line by line
    data = []
    for row in train_obj:
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

# First simple model: only female survives.

# Read test CSV file and write model to new CSV file.
with open('test.csv', 'rb') as test_file, open('model_gender.csv', 'wb') as predict_file:
    test_obj = csv.reader(test_file)
    header_test = test_file.next()
    
    predict_obj = csv.writer(predict_file)
    predict_obj.writerow(['PassengerId','Survived'])

    for row in test_obj:
        # Apply model to each person
        if row[3] == 'female':
            predict_obj.writerow([row[0], '1'])
        else:
            predict_obj.writerow([row[0], '0'])

# Second simple model: only female that didn't pay more than 20$ for a third class ticket survives.

# Read test CSV file and write model to new CSV file.
with open('test.csv', 'rb') as test_file, open('model_genderticket.csv', 'wb') as predict_file:
    test_obj = csv.reader(test_file)
    header_test = test_file.next()
    
    predict_obj = csv.writer(predict_file)
    predict_obj.writerow(['PassengerId','Survived'])

    for row in test_obj:
        # Apply model to each person
        if row[3] == 'male':
            predict_obj.writerow([row[0], '0'])
        elif int(row[1]) == 3 and float(row[8] > 20):
            predict_obj.writerow([row[0], '0'])
        else:
            predict_obj.writerow([row[0], '1'])
