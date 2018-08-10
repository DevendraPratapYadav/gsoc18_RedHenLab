#*************************************************
# Stage 6 - Feature Extraction
#*************************************************


import os
import sys
import subprocess
import cv2
import csv

import stages

stage = stages.stage6

openface_FaceLandmarkVidMulti_path = stage.dependencies['openface_FaceLandmarkVidMulti_path']

def PV(arr):
    for x in arr:
        print(x);


def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


def get_csv_data(csvFile):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data


def do_feature_extraction(input_folder, output_folder):
    """
    Execute feature extraction program using OpenFace
    """
    program_location = stage.dependencies['feature_extraction_script_path']
    c1 = 'python \'' + program_location + '\' \'' + openface_FaceLandmarkVidMulti_path + '\' \'' + input_folder + '\' \'' + output_folder + '\'';

    # execute command

    print(c1)
    subprocess.call(c1, shell=True)



input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

print('Begin Stage ', stage.number, ' : ', stage.name)


assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


# process files
do_feature_extraction(input_folder, output_folder);

print('End Stage ', stage.number, ' : ', stage.name)
