#*************************************************
# Program to perform face tracking in video using face detection output
#*************************************************


import os
import sys
import subprocess
import cv2
import csv
import pdb

import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

TRACKING_FAILURE_THRESHOLD = 10;
TRACKING_IOU_THRESHOLD = 0.3
MIN_TRACK_LENGTH = 10


def PV(arr):
    for x in arr:
        print(x);
        print('\n')


def showImg(img, wait):
    cv2.imshow('image', img)
    cv2.waitKey(wait)


def getIOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute area of intersection
    intersectionArea = (xB - xA + 1) * (yB - yA + 1)

    # compute area
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)


    nonOverlappingArea = float(boxAArea + boxBArea - intersectionArea) + 1
    iou = intersectionArea / nonOverlappingArea

    return iou


def getTracker(trackerIndex):
    # Set up tracker and return object
    # Input - index of tracking method as per list below

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # tracker_type = tracker_types[2]
    tracker_type = tracker_types[trackerIndex]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    return tracker


def fixROI(roi, frame_width, frame_height):
    nroi = roi[:];

    nroi[0] = max(roi[0], 0);
    nroi[1] = max(roi[1], 0);
    nroi[2] = min(roi[2], frame_width);
    nroi[3] = min(roi[3], frame_height);

    return nroi


def performTracking(faceDetections, videoPath):
    """
    Perform face tracking on video

    Input : faceDetections - list of face detections in format : [frameNumber, [bbox1, bbox2 ...], shotNumber]
            videoPath - path to video file

    Output: tracks - list of face tracks. Each face track is list of - [frameNumber, bbox], where bbox = [xmin,ymin,xmax,ymax]

    """
    tracks = [];

    if (len(faceDetections) < 1):
        return tracks

    print(videoPath)
    # Read video
    video = cv2.VideoCapture(videoPath)

    total_frames = int(video.get(7))
    FPS = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    print('Total frames:', total_frames, ', FPS:', FPS)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        return [];

    for det in faceDetections:

        # print('Detection :', det)

        if (len(det[1]) == 0):
            continue;

        frameNoOrg = det[0];
        shotEndFrame = det[2];

        for face in det[1]:

            if (len(face) < 1):
                # print('No face detection')
                continue;

            print('Face:', face, ', Start frame number:', frameNoOrg)
            face = fixROI(face, frame_width, frame_height)

            frameNo = frameNoOrg;

            # video.set(1,0);
            video.set(1, frameNo);
            bbox = (face[0], face[1], face[2] - face[0] + 1, face[3] - face[1] + 1)

            # read first frame in track
            ok, frame = video.read()

            # Initialize tracker with first frame and bounding box
            tracker = getTracker(2); # provide index of tracking method as per list ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
            ok = tracker.init(frame, bbox)

            newTrack = [];  # stores face tracking output
            newTrack.append([frameNo, face]);

            frameNo += 1;

            fails = 0;
            while True:
                # print('frame: ', frameNo);

                # Read a new frame
                ok, frame = video.read()
                if not ok:
                    print('Cannot read more frames')
                    break

                # Start timer
                timer = cv2.getTickCount()

                # Update tracker
                ok, bbox = tracker.update(frame) # NOTE : bbox format : x1,y1, w,h

                # print(bbox)
                trackBox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                if (ok):
                    newTrack.append([frameNo, trackBox]);
                    # remove detected faces overlapping with our track
                    for ii in range(len(faceDetections)):
                        if (faceDetections[ii][0] == frameNo and ok):
                            for jj in range(len(faceDetections[ii][1])):
                                detBox = faceDetections[ii][1][jj];

                                if (len(detBox) < 1):
                                    continue;
                                
                                if (getIOU(trackBox, detBox) > TRACKING_IOU_THRESHOLD):
                                    # print('removing:', detBox)
                                    faceDetections[ii][1][jj] = [];

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                # print (ok,bbox)
                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    fails = 0;
                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                    fails += 1

                
                if (fails > TRACKING_FAILURE_THRESHOLD or frameNo >= shotEndFrame):
                    break;

                frameNo += 1;


            cv2.destroyAllWindows()
            if (len(newTrack) >= MIN_TRACK_LENGTH):
                tracks.append(newTrack);

    video.release();
    return tracks



faceDetectionsFile = sys.argv[1];
outputPath = sys.argv[2];


def get_csv_data(csvFile):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data


dets = get_csv_data(faceDetectionsFile);


stage2_data = dets[0:2];
dets = dets[4:];

videoFile = stage2_data[1][2];
print(videoFile)

shotCount = 0;
for d in dets:
    shotCount = max(shotCount, int(d[1]));

shots = []
for ii in range(shotCount):
    shots.append([]);

for det in dets:
    shotNo = int(det[1])

    frameNo = int(det[0])
    shotEndFrame = int(det[2])

    faces = [];
    for bb in det[3:]:
        bb = bb.split('_');
        bb = list(map(int, bb))

        faces.append(bb);

    if ( len(shots[shotNo - 1]) < 1 ):
        shots[shotNo - 1] = [[frameNo, faces, shotEndFrame]]
    else:
        shots[shotNo - 1].append([frameNo, faces, shotEndFrame])


videoName = videoFile.split('/')[-1];

outFilePath = outputPath + '/' + videoName + '_faceTracks.csv';
outFile = open(outFilePath, 'w')

# write stage information to output file
outFile.write(','.join(['StageName', 'Stage_Number', 'InputFile', 'OutputFile']) + '\n');

outFile.write(','.join(['Face Tracking', '3', videoFile, outFilePath]) + '\n');

# write face detection output
outFile.write('\n');

outFile.write(','.join(['Frame number', 'Shot number', 'Face id', 'x_min', 'y_min', 'x_max', 'y_max']) + '\n');

tracks = []
trackId = 1;


# perform face tracking for each shot
for ind in range(len(shots)):

    print('----------------------------------------')
    print('Shot : ', ind + 1)
    print('----------------------------------------\n')

    detections = shots[ind];
    tracks = performTracking(detections, videoFile)
    # PV(tracks)
    for tt in tracks:
        for face in tt:
            frameNo = str(face[0])
            face_id = str(trackId);
            shotNo = str(ind + 1);
            box = list(map(str, face[1]));
            outArr = [frameNo] + [shotNo] + [face_id] + box
            # print (outArr)
            outFile.write(','.join(outArr) + '\n');

        trackId += 1

outFile.close();
