#*************************************************
# Program to perform face cropping in video using face tracking output
#*************************************************


import numpy as np
import cv2
import os
import sys
import pdb
import csv

MIN_VIDEO_LENGTH = 3  # in seconds. face segments smaller than this are ignored
# MIN_NUM_FRAMES = 5; 

TRACK_SMOOTH_NUMFRAMES = 2  # number of past frames to look at for smoothing tracking

ROI_SCALE = 1.3; # scale face detection by this amount before cropping

# SKIP_FRAMES_NUM = 5; 


def show(img, wait):
    cv2.imshow('image', img)
    cv2.waitKey(wait)


def PV(arr):
    for x in arr:
        print(x);


def PNN(ss):
    print(ss, end=', ', flush=True)

NEXT_FRAME_TO_READ = 0
def getFrame(cap, i, show):
    global NEXT_FRAME_TO_READ

    if (i!=NEXT_FRAME_TO_READ):
        cap.set(1, i)
    ret, frame = cap.read()
    # print('read frame: ',i)
    if (show == 1):
        cv2.imshow('frame', frame);
        cv2.waitKey(0);
    NEXT_FRAME_TO_READ = i+1
    return frame;


def readFile(fil):
    lines = [];
    with open(fil) as f:
        lines = f.readlines()
    return lines;


def viewVideo(vid):
    frameNo = 0;

    while (True):
        # vid.set(cv2.CAP_PROP_POS_MSEC ,frameNo*1000.0/24.0)
        ret, frame = vid.read()

        if ret == True:
            frameNo += 1;
            print('read frame : ',frameNo)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


def cropFrame(vid, frameNo, roi):
    # roi is in [1, num_pixels]

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    if (frameNo < 0 or frameNo > total_frames):
        print("ERROR : frameNo:", frameNo, " is invalid.")
        return None;

    frame = getFrame(vid, frameNo, 0);

    roi = list(map(int, roi))

    xmin = roi[0];
    xmax = roi[2];
    ymin = roi[1];
    ymax = roi[3];

    if (xmin < 0 or xmax > frame_width or ymin < 0 or ymax > frame_height):
        print("ERROR : ymin,ymax, xmin,xmax : ", ymin, ", ", ymax, ", ", xmin, ", ", xmax);
        return None

    cropped = frame[ymin:ymax, xmin:xmax, :]

    return cropped;


def convertToPixels(roi, vid):
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    if (xmax > 1):
        return roi

    xmin = int(roi[0] * frame_width);
    xmax = int(roi[2] * frame_width);
    ymin = int(roi[1] * frame_height);
    ymax = int(roi[3] * frame_height);

    roi = [xmin, ymin, xmax, ymax];
    return roi


def getMeanRoi(rois, vid):
    rois = np.array(rois);
    rois = rois[:, 1:];  # remove frame no.

    meanROI = np.mean(rois, 0)

    meanROI = list(map(int, meanROI));

    return meanROI


def checkROI(roi, vid):
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    if (roi[0] < 0 or roi[2] > frame_width or roi[1] < 0 or roi[3] > frame_height):
        return 0

    return 1;


def fixROI(roi, vid):
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    nroi = roi[:];

    nroi[0] = max(roi[0], 0);
    nroi[1] = max(roi[1], 0);
    nroi[2] = min(roi[2], frame_width);

    nroi[3] = min(roi[3], frame_height);

    return nroi


def getROIPixelCenter(roi, vid):
    # roi = convertToPixels(roi,vid);

    cx = int((roi[0] + roi[2]) / 2)
    cy = int((roi[1] + roi[3]) / 2)

    return cx, cy


def getROIRelativeCenter(roi, vid):
    cx = (roi[0] + roi[2]) / 2.0
    cy = (roi[1] + roi[3]) / 2.0

    return cx, cy


def resizeROI(roi, meanROI, smoothCenters, roi_scale, vid):
    # roi is in [0,1]
    # meanROI is in [1,num_pixels]

    roi = np.array(roi);
    meanROI = np.array(meanROI);

    MW = int((meanROI[2] - meanROI[0]) * roi_scale);  # mean width
    MH = int((meanROI[3] - meanROI[1]) * roi_scale);  # mean height

    SC = np.array(smoothCenters)

    # get smoothened center
    cx = int(np.mean(SC, 0)[0]);
    cy = int(np.mean(SC, 0)[1]);

    xmin = cx - MW / 2
    ymin = cy - MH / 2
    xmax = cx + MW / 2
    ymax = cy + MH / 2

    resizedROI = [xmin, ymin, xmax, ymax];

    resizedROI = fixROI(resizedROI, vid);

    return resizedROI;


def makeROISquare(roi):
    MW = int(meanROI[2] - meanROI[0]);  # mean width
    MH = int(meanROI[3] - meanROI[1]);  # mean height

    squareSize = max(MW, MH)
    cx, cy = getROIPixelCenter(roi, None)

    xmin = cx - squareSize / 2
    ymin = cy - squareSize / 2
    xmax = cx + squareSize / 2
    ymax = cy + squareSize / 2

    squareROI = [xmin, ymin, xmax, ymax];

    return squareROI


def get_csv_data(csvFile):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data


# START PROGRAM

if (len(sys.argv) < 3):
    print("Usage: python faceCrop <tracks.csv> <output_folder>")

trackFile = sys.argv[1];
output_folder = sys.argv[2];

tracks = get_csv_data(trackFile);
stage3_data = tracks[0:2];
tracks = tracks[4:];

videoFile = stage3_data[1][2];
print(videoFile)

print(videoFile, '\n', trackFile, '\n', output_folder);

# READ VIDEO

vid = cv2.VideoCapture(videoFile);

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
total_frames = int(vid.get(7))
FPS = vid.get(cv2.CAP_PROP_FPS)

print("height:", frame_height, ", width:", frame_width, ", total_frames:", total_frames, ", FPS:", FPS)


# READ TRACK
lines = tracks;

# dict containing all detection for each face. indexed by face number
faces = {};

for x in lines:

    x = list(map(float, x));
    x = list(map(int, x));

    face_id = x[2]

    if (face_id not in faces):
        faces[face_id] = []

    faces[face_id].append(x)

    # break;

croppedFacesPath = output_folder

# process each face track to crop it
for f in faces.items():

    face_id = f[0]
    face_data = f[1]

    print("Face :", face_id);

    face_roi = np.array(face_data);
    frameNums = np.array(face_roi[:, 0]).reshape(-1, 1);

    face_roi = face_roi[:, -4:]

    face_roi = np.hstack((frameNums, face_roi));

    shotNo = face_data[0][1]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    videoName = croppedFacesPath + "/" + videoFile.split('/')[-1] + "_" + str(shotNo) + "_" + str(face_id) + '.mp4';
    csvFileName = croppedFacesPath + "/" + videoFile.split('/')[-1] + "_" + str(shotNo) + "_" + str(face_id) + '.csv';

    csvFile = open(csvFileName, 'w')

    csvFile.write(','.join(['StageName', 'Stage_Number', 'InputFile', 'OutputFile']) + '\n');

    csvFile.write(','.join(['Face Cropping', '4', videoFile, videoName]) + '\n');

    csvFile.write('\n');

    csvFile.write(
        ','.join(['Frame no', 'Shot no', 'Face id', 'Crop_x_min', 'Crop_y_min', 'Crop_x_max', 'Crop_y_max']) + '\n');

    meanROI = getMeanRoi(face_roi, vid);
    meanROI = makeROISquare(meanROI)

    oMW = int(meanROI[2] - meanROI[0]);  # mean width
    oMH = int(meanROI[3] - meanROI[1]);  # mean height
    print("meanROI: ", meanROI, ' , W:', oMW, ', H:', oMH);

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    # Scale ROI region
    MW = int(oMW * ROI_SCALE)
    MH = int(oMH * ROI_SCALE)

    roi_scale = ROI_SCALE;

    if (MW < 0 or MH < 0 or MW > frame_width or MH > frame_height):
        print('Over bounds of video')
        roi_scale = 1.0;
        MW = oMW;
        MH = oMH;

    outVideoSize = (MW, MH)

    out = cv2.VideoWriter(videoName, fourcc, FPS, outVideoSize)

    smoothCenters = [];

    segmentFrames = len(face_roi);
    frameNum = -1;

    for fr in face_roi:
        frameNum += 1;

        cx, cy = getROIPixelCenter(fr[1:], vid);

        if (len(smoothCenters) < 1):
            for ind in range(0, TRACK_SMOOTH_NUMFRAMES):
                smoothCenters.append([cx, cy])

        # remove oldest center and add current
        temp = smoothCenters.pop(0);
        smoothCenters.append([cx, cy]);
        # print(len(smoothCenters));

        # resize roi to crop region of mean width and height
        resizedROI = resizeROI(fr[1:], meanROI, smoothCenters, roi_scale, vid);

        cropped = cropFrame(vid, fr[0], resizedROI);

        # check if roi or frameNo is invalid. If yes, remove its file
        if (cropped is None):
            print("Face:", f[0], "bounds error(roi or frameNo).")
            print("deleting file:", videoName)
            out.release()
            os.remove(videoName);
            break;

        cropped = cv2.resize(cropped, outVideoSize)  # resize if roi size different from video size

        # print("Cropping frame:",f[0]," - ", fr);
        # show(cropped,0);

        out.write(cropped)

        csvFile.write(','.join(
            [str(frameNum), str(shotNo), str(face_id), str(resizedROI[0]), str(resizedROI[1]), str(resizedROI[2]),
             str(resizedROI[3])]) + '\n');

    cv2.destroyAllWindows()
    out.release()
    csvFile.close();

# viewVideo(vid);

vid.release()
# out.release()

# Close all the frames
cv2.destroyAllWindows()
