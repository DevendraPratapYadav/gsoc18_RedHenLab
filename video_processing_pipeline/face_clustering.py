#*************************************************
# Stage 7 - Face Clustering
#*************************************************


import sys
import os
import dlib
import cv2
import pandas as pd
import numpy as np
import stages

from pyannote.core import Segment, Annotation
from pyannote.video.face.clustering import FaceClustering


CLUSTERING_THRESHOLD = 0.6
NUM_FRAMES_PER_VIDEO = 6

def showImg(img, wait):
    cv2.imshow('image', img)
    cv2.waitKey(wait)


def PV(arr):
    for x in arr:
        print(x);

def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


def getVideoEmbeddings(videoPath):
    """
    Process video to extract embeddings using face recognition model 'facerec'
    Input : videoPath - path to video file
    Output : [videoPath, videoEmbeddings] - videoEmbeddings contains list of embeddings for video frames
    """

    videoEmbeddings = []
    vid = cv2.VideoCapture(videoPath);
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    print(videoPath)
    # print("height:", frame_height, ", width:", frame_width, ", total_frames:", total_frames, ", FPS:", FPS)

    if (FPS < 1 or total_frames < NUM_FRAMES_PER_VIDEO):
        vid.release()
        return []

    # Now process frames from video
    for frameNo in range(3, total_frames-3):

        if (int(total_frames/NUM_FRAMES_PER_VIDEO) == 0):
            return []
            
        if (frameNo % int(total_frames/NUM_FRAMES_PER_VIDEO) !=0 ):
            continue
        # print(frameNo)

        vid.set(1, frameNo)
        readOK, img = vid.read()
        if not readOK:
            print('Cannot read more frames.')
            continue;

        # win.clear_overlay()
        # win.set_image(img)

        # Get the landmarks/parts for the face in box d.
        d = dlib.rectangle( 0, 0 , frame_width, frame_height)

        # print ('Detection:', d)
        shape = sp(img, d)
        # Draw the face landmarks on the screen so we can see what face is currently being processed.
        
        # win.clear_overlay()
        # win.add_overlay(d)
        # win.add_overlay(shape)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        videoEmbeddings.append( list(face_descriptor) )

    vid.release()
    return [videoPath, videoEmbeddings]



def convertToPandasDataFrame(embeddings):
    """
    Convert embeddings to pandas data frame for use with Pyannote-video
    Input : embeddings - list of embeddings in format [videoPath, videoEmbeddings]
    Output: trackToVideo - dict to get videoPath from track number
            dataFrame - pandas data frame with all video embeddings. 
                        Each video is given a unique track number. 
                        Each frame in video is given a unique time stamp

    """
    names = ['time', 'track']
    for i in range(128):
        names += ['d{0}'.format(i)]

    d = {}
    for nn in names:
        d[nn] = []

    trackToVideo = {}

    for ind in range(len(embeddings)):

        trackNumber = ind;

        trackToVideo [trackNumber] =  embeddings[ind][0]
        embedding = embeddings[ind][1]

        for ei in range(len(embedding)):
            
            d[ names[0] ].append( float(trackNumber*10 + ei) ) # time
            d[ names[1] ].append( trackNumber ) # track number
            
            for ii in range( len(embedding[ei]) ):
                d[ names[2+ii] ].append( embedding[ei][ii] )

    dataFrame = pd.DataFrame(d)
    dataFrame = pd.DataFrame(dataFrame, columns=names)

    dataFrame.sort_values(by=['track', 'time'], inplace=True)

    return trackToVideo, dataFrame



def getTrackStartingPoints(data):
    """
    Get track start points from pandas data frame 
    """
    def _to_segment(group):
            return Segment(np.min(group.time), np.max(group.time))

    starting_point = Annotation(modality='face')
    for track, segment in data.groupby('track').apply(_to_segment).iteritems():
        if not segment:
            continue
        starting_point[segment, track] = track

    return starting_point


def getEmbeddingForVideosInFolder(input_folder):
    """
    Get face embeddings for videos in folder
    """
    allVideoEmbeddings = []

    for f in os.listdir(input_folder):
        filePath = input_folder+'/'+f

        if ( f[-4:].lower() == '.avi' or f[-4:].lower() == '.mp4' ):
            videoEmbeddings = getVideoEmbeddings(filePath)
            
            if (len(videoEmbeddings) > 0):
                allVideoEmbeddings.append(videoEmbeddings)

    return allVideoEmbeddings


def makeIdentitiesConsecutive(clusteringOutput):
    """
    Make identity numbers consecutive
    """
    identities = []

    for xx in clusteringOutput:
        identities.append(xx[1])

    identities = np.unique(identities)

    iDict = {}
    for ind in range(len(identities)):
        iDict[identities[ind]] = ind

    for ind in range(len(clusteringOutput)):
        clusteringOutput [ind][1] = 1+iDict[ clusteringOutput [ind][1] ]

    return clusteringOutput




def writeIdentityCsvFiles(clusteringOutput, output_folder):
    """
    Write clustering output to csv file
    """

    outputFiles = {}

    for vid in clusteringOutput:
        
        vidName = vid[0].split('/')[-1]
        vidName = vidName.split('_aligned')[0][:-4]

        vidInfo = vidName.split('_')[-2:]
        inputVideoName = '_'.join(vidName.split('_')[:-2])

        shotNum = int(vidInfo[0])
        face_id = int(vidInfo[1])

        identity = vid[1]

        print(inputVideoName, shotNum, face_id, identity)

        if (inputVideoName not in outputFiles):
            outputFiles[inputVideoName] = [ [shotNum, face_id, identity] ]
        else:
            outputFiles[inputVideoName].append( [shotNum, face_id, identity] )


    for videoName, identities in outputFiles.items():
        csvFileName = output_folder+'/'+videoName+'_identities.csv';
        csvFile = open(csvFileName, 'w');

        sourceVideoPath = input_stage0.outputLocation+'/'+videoName

        # write stage information to output file
        csvFile.write(','.join(['StageName', 'Stage_Number', 'InputFile', 'OutputFile']) + '\n');

        csvFile.write(
            ','.join([stage.name, str(stage.number), sourceVideoPath, csvFileName]) + '\n');

        csvFile.write('\n')

        csvFile.write( ','.join( ['Shot Number','Face id', 'Identity']) + '\n' )

        for ii in identities:
            csvFile.write( ','.join(  list(map(str,ii)) ) +'\n')


# script start


# Get stage information
stage = stages.stage7

input_stage0 =  stages.stage0

input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

print('Begin Stage ', stage.number, ' : ', stage.name)


assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


# initialize shape predictor and face recognition models
# libsPath = output_folder.split('/')[0]+'/libs'

predictor_path = stage.dependencies['dlib_faceLandmarks_model_path']
face_rec_model_path = stage.dependencies['dlib_faceRecognition_model_path']

sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# win = dlib.image_window()

allVideoEmbeddings = getEmbeddingForVideosInFolder(input_folder)

trackNumberToVideo, embeddingData = convertToPandasDataFrame(allVideoEmbeddings)

face_tracks = getTrackStartingPoints(embeddingData)


# perform clustering using Pyannote-videos

clustering = FaceClustering(threshold=CLUSTERING_THRESHOLD)

face_tracks.get_timeline()

result = clustering(face_tracks, features=embeddingData)

clusteringOutput = []
for _, track_id, cluster in result.itertracks(yield_label=True):
    # print(trackNumberToVideo[track_id] ,' :- ', cluster )
    clusteringOutput.append( [trackNumberToVideo[track_id], cluster] )


clusteringOutput = makeIdentitiesConsecutive(clusteringOutput)

# PV(clusteringOutput)

writeIdentityCsvFiles(clusteringOutput, output_folder)


print('End Stage ', stage.number, ' : ', stage.name)



