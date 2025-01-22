import pandas as pd
import random
import os
import shutil

# load Ground Truth
df = pd.read_csv('ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
# source directory
source_dirpath = 'ISIC2018_Task3_Training_Input'

# split train / test
random.seed(123)
trainset = random.sample(list(range(10015)), 7000)

os.makedirs('ISIC2018_dataset', exist_ok=True)
for index, row in df.iterrows():
    # get file name
    filename = row['image'] + '.jpg'
    # get label
    label = row[row==1].index[0]
    # train or test
    datasettype = ['test', 'train'][index in trainset]

    target_dirpath = 'ISIC2018_dataset/' + datasettype + '/' + label
    os.makedirs(target_dirpath, exist_ok=True)
    print(os.path.join(source_dirpath, filename), "to", os.path.join(target_dirpath, filename))
    
    # file copy
    shutil.copy2(os.path.join(source_dirpath, filename), os.path.join(target_dirpath, filename))
