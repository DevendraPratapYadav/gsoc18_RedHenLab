#*************************************************
# Stage 1 - Shot segmentation
#*************************************************

import os
import sys
import subprocess
import cv2

import stages

def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


#*************************************************
# Pyannote-Video Shot segmentation
#*************************************************

from pyannote.video import Video
from pyannote.video import Shot, Thread

def pyannote_shot(videoPath):
    """
    Perform shot segmentation using pyannote-video and return output
    Input : videoPath - Path to video file
    Output : shotBoundaries - list of [shot start, shot end]

    """
    video = Video(videoPath)
    FPS = video._fps
    shots = Shot(video)
    shotBoundaries = []

    for x in shots:
        shotBoundaries.append([int(FPS * x.start), int(FPS * x.end)])

    return shotBoundaries

def do_shot_segmentation(filename, filePath, input_folder, output_folder):
    """
    Process video file and write output
    Input : filename - name of video
            filePath - path to video file
            input_folder - input folder for current stage
            output_folder - output folder for current stage

    """

    # call any shot segmentation function here:
    
    # perform shot segmentation using Pyannote-video
    output = pyannote_shot(filePath)


    # process shot segmentation output and write it to file
    outFileName = filename + '_shots.csv'
    outFile = open(os.path.join(output_folder, outFileName), 'w')

    # write stage information to output file
    outFile.write(','.join(['StageName', 'Stage_Number', 'InputFile', 'OutputFile']) + '\n');

    outFile.write(','.join([stage.name, str(stage.number), os.path.join(input_folder, filename),
                            os.path.join(output_folder, outFileName)]) + '\n');

    # write shot segmentation output
    outFile.write('\n');
    outFile.write(','.join(['Shot start', 'Shot end']) + '\n');
    for segment in output:
        outFile.write(','.join([str(segment[0]), str(segment[1])]) + '\n');


# Get stage information
stage = stages.stage1

input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

print('Begin Stage ', stage.number, ' : ', stage.name)

assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


# process videos one by one
for f in os.listdir(input_folder):
    filePath = os.path.join(input_folder, f)
    print('Processing: ', filePath)

    do_shot_segmentation(f, filePath, input_folder, output_folder);

print('End Stage ', stage.number, ' : ', stage.name)
