#*************************************************
# Stage 5 - Face Alignment
#*************************************************


import os
import sys
import subprocess
import cv2
import csv
from argparse import ArgumentParser

import stages

stage = stages.stage5
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


def do_face_alignment(filename, filePath, output_folder, alignmentMethod, outputVideoSize):
    """
    Execute face alignment program using custom landmark detection
    """
    program_location = stage.dependencies['faceAlign_manual_script_path']
    libsPath = stage.dependencies['dlib_faceLandmarks_model_path']
    c1 = 'python ' + program_location + ' \'' + filePath + '\' ' + output_folder + ' ' + alignmentMethod + ' ' + str(
        outputVideoSize) + ' \'' + libsPath + '\' ';

    # execute command

    print(c1)
    subprocess.call(c1, shell=True)



def do_face_alignment_openface(input_folder, output_folder):
    """
    Execute face alignment program using OpenFace's landmark detection
    """
    program_location = stage.dependencies['faceAlign_openface_script_path']
    c1 = 'python '+program_location+' '+openface_FaceLandmarkVidMulti_path+' '+input_folder+' '+output_folder;

    # execute command
    print(c1)
    subprocess.call(c1, shell=True)



argparse = ArgumentParser()

argparse.add_argument('--method', type=str, help='Landmark/method to use for face alignment. Valid options : \'dlib\', \'mtcnn\', \'openface\'' , default='openface')
argparse.add_argument('--output_size', type=str, help='Size of aligned square face video', default='224')

args = argparse.parse_args() 


input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

alignmentMethod = args.method
outputVideoSize = int(args.output_size)

print('Begin Stage ', stage.number, ' : ', stage.name)

assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


if (alignmentMethod == 'openface'):
    do_face_alignment_openface(input_folder, output_folder)
else:
    # process files
    for f in os.listdir(input_folder):
        filePath = os.path.join(input_folder, f)

        # process video files

        if (f[-4:].lower() == '.mp4' or f[-4:].lower() == '.avi'):
            print('Processing: ', filePath)

            do_face_alignment(f, filePath, output_folder, alignmentMethod, outputVideoSize);


print('End Stage ', stage.number, ' : ', stage.name)