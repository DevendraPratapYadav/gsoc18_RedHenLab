#*************************************************
# Stage 4 - Face Cropping
#*************************************************


import os
import sys
import subprocess
import cv2
import csv

import stages


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


def do_face_cropping(filename, filePath, output_folder):
    """
    Execute face cropping program for face detection csv file
    """
    program_location = stage.dependencies['face_crop_script_path']
    c1 = 'python \'' + program_location + '\' \'' + filePath + '\' ' + output_folder;

    # execute command

    print(c1)
    subprocess.call(c1, shell=True)


stage = stages.stage4

input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

print('Begin Stage ', stage.number, ' : ', stage.name)

assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


# process files
for f in os.listdir(input_folder):
    # print (i);
    filePath = os.path.join(input_folder, f)

    # process csv files
    if (filePath[-4:] == '.csv'):
        print('Processing: ', filePath)

        do_face_cropping(f, filePath, output_folder);

print('End Stage ', stage.number, ' : ', stage.name)
