#*************************************************
# Program to perform feature extraction in aligned face videos using OpenFace
#*************************************************


import os
import sys
import subprocess
import cv2

BATCH_SIZE = 1;
OUTPUT_VIDEO_FRAME_SIZE = 224;


def PV(arr):
    for x in arr:
        print(x);


def processVideos(outPath, files, openfacePath):
    # Prepare OpenFace command
    videoPaths = "";
    c1 = '"{openfacePath}" {videos}-out_dir ' + outPath + ' -oc mp4v -2Dfp -pose -gaze -pdmparams -aus -multi_view 1';

    # add names of videos in batch
    for f in range(0, len(files)):
        videoPaths += '-f "' + files[f] + '" ';

    # execute OpenFace command
    com1 = c1.format(videos=videoPaths, openfacePath=openfacePath);
    print(com1)
    subprocess.call(com1, shell=True)

    print(" ")


def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


if (len(sys.argv) < 4):
    print(
        'Usage: python processVideosWithOpenFace.py <FaceLandmarkVidMulti program path> <input videos folder> <output folder>')
    print('Example : python processVideosWithOpenFace.py OpenFace/build/bin/FaceLandmarkVidMulti videos processed');
    sys.exit();

files = [];
openface_program_path = removeTrailingBackslash(sys.argv[1]);
input_folder = removeTrailingBackslash(sys.argv[2]);
output_folder = removeTrailingBackslash(sys.argv[3]);

# process videos in batches
for i in os.listdir(input_folder):

    if (i[-4:].lower() == '.mp4' or i[-4:].lower() == '.avi'):
        files.append(input_folder + '/' + i)

    if (len(files) >= BATCH_SIZE):
        processVideos(output_folder, files, openface_program_path);
        files = [];

# process remaining files in batch
if (len(files) > 0):
    processVideos(output_folder, files, openface_program_path);
    files = [];

print('\nProcessing complete.\n')
