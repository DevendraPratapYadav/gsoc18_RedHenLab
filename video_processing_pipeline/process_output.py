#*************************************************
# Stage 8 - Process Output
#*************************************************


import os
import sys
import subprocess
import cv2
import csv
import numpy as np
import math

import stages

FEATURE_CONFIDENCE_THRESHOLD = 0.75 # If feature detection confidence is below this, detection is considered erroneous

AU_INTENSITY_THRESHOLD = 1.1 # threshold for AU to be considered present

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


def showImg(img, wait):
    cv2.imshow('image', img)
    cv2.waitKey(wait)

def getFrame(cap, i, show):
    cap.set(1, i)
    ret, frame = cap.read()
    if (show == 1):
        cv2.imshow('frame', frame);
        cv2.waitKey(0);
    return frame;


def createOutputFile( faceTracks_file, output_folder, faceCrop_folder, featureExtraction_folder, faceIdentity_folder, depthOutput_folder):
    trackFile = faceTracks_file;

    tracks = get_csv_data(trackFile);
    stage3_data = tracks[0:2];
    tracks = tracks[4:];

    if (len(tracks)< 1):
        return ''

    videoFileName = stage3_data[1][2].split('/')[-1];
    videoFilePath = stage3_data[1][2]

    # print(videoFileName, '\n', trackFile, '\n', output_folder);

    faceDict = {}

    outputFileData = {}

    outputColumnNames = []
    featureColumnNames = []
    faceDetections = {}

    faceIdentities = {}

    for tt in tracks:

        frameNum = int(tt[0])
        face_id = int(tt[2])
        shotNum = int(tt[1])

        bbox = tt[3:7];
        # print(bbox)
        bbox = [ int(float(xx)) for xx in bbox ]
        
        if (shotNum, face_id) in faceDict:
            faceDict[(shotNum, face_id)].append([shotNum, face_id, frameNum])
        else:
            faceDict[(shotNum, face_id)] = [[shotNum, face_id, frameNum]]

        faceDetections[ (frameNum, shotNum, face_id) ] = bbox

    # PV(sorted(faceDict.keys()))

    faceIdentitiesFile = faceIdentity_folder+'/'+videoFileName+'_identities.csv'
    
    if (os.path.exists(faceIdentitiesFile) == True):
        identities = get_csv_data(faceIdentitiesFile);
        stage7_data = identities[0:2];
        identities = identities[4:];

        for idt in identities:
            shotNum_ = int(idt[0])
            face_id_ = int(idt[1])
            identity_ = int(idt[2])

            faceIdentities[ (shotNum_, face_id_) ] = identity_

    # get depth data
    depth_file = depthOutput_folder + '/' + videoFileName+'_depth.mp4'
    depthDataAvailable = False
    depthVideo = None
    if (os.path.exists(depth_file) == False):
        print(depth_file, ' does not exist.')
    else:
        depthDataAvailable = True

    if (depthDataAvailable == True):
        depthVideo = cv2.VideoCapture(depth_file);

        depthVideo_frame_width = int(depthVideo.get(3))
        depthVideo_frame_height = int(depthVideo.get(4))
        depthVideo_total_frames = int(depthVideo.get(7))
        depthVideo_FPS = depthVideo.get(cv2.CAP_PROP_FPS)

        print("Depth video \nheight:", depthVideo_frame_height, ", width:", depthVideo_frame_width, ", total_frames:", depthVideo_total_frames, ", FPS:", depthVideo_FPS)


    inputVideo = cv2.VideoCapture(videoFilePath);

    inputVideo_frame_width = int(inputVideo.get(3))
    inputVideo_frame_height = int(inputVideo.get(4))
    inputVideo_total_frames = int(inputVideo.get(7))
    inputVideo_FPS = inputVideo.get(cv2.CAP_PROP_FPS)

    print("Input video \nheight:", inputVideo_frame_height, ", width:", inputVideo_frame_width, ", total_frames:", inputVideo_total_frames, ", FPS:", inputVideo_FPS)

    def freeRescaleRoi(roi, currentSize, newSize):
        # rescale roi coordinates from currentSize video to newSize video

        # convert x coordinates
        roi[0] = int(roi[0] * newSize[0] / currentSize[0])
        roi[2] = int(roi[2] * newSize[0] / currentSize[0])

        # convert y coordinates
        roi[1] = int(roi[1] * newSize[1] / currentSize[1])
        roi[3] = int(roi[3] * newSize[1] / currentSize[1])

        return roi


    for face_, frame in sorted(faceDict.items()):

        shotNum = int(frame[0][0])
        face_id = int(frame[0][1])
        faceTrackStartFrame = int(frame[0][2])

        faceCrop_file = faceCrop_folder + '/' + videoFileName+'_'+str(shotNum)+'_'+str(face_id)+'.csv'

        if (os.path.exists(faceCrop_file) == False):
            print(faceCrop_file, ' does not exist. Most likely, features could not be extracted for face.')
            continue;

        cropDataArray = get_csv_data(faceCrop_file)[4:];

        faceTrackFrameNums = [ int(xx[2]) for xx in frame ]
        faceTrackFrameNums = sorted(faceTrackFrameNums)

        # create frame-wise cropData dict
        cropData = {}
        for dd in cropDataArray:

            crop_shotNum = int(dd[1])
            crop_face_id = int(dd[2])
            crop_frameNum = int(dd[0])

            cropData[ crop_frameNum ] = list(map(float,dd[3:7]))

        features_file = featureExtraction_folder + '/' + videoFileName+'_'+str(shotNum)+'_'+str(face_id)+'.csv'
        featureData = []

        if (os.path.exists(features_file) == False):
            print(features_file, ' does not exist. Most likely, features could not be extracted for face.')
            # continue;
        else:
            featureData = get_csv_data(features_file);

            colNames = featureData[0]
            colNames = [ elem.strip() for elem in colNames ]

            if (len(featureColumnNames)<1):
                featureColumnNames = colNames[:]

            featureData = featureData[1:];


        def findIndex(colName):
            return featureColumnNames.index(colName);

        def listMapInt(arr):
            arrInt = list(map(int, list(map(float ,arr))))
            return arrInt

        def addNumToList(arr, addNum, ranges):
            # ranges points are inclusive of all elements to be changed
            for rr in ranges:

                for rInd in range(rr[0], 1+rr[1]):
                    arr[rInd]+= addNum;

            return arr;

        featureDataNoFrameSkipped = []
        featureDataIndex = 0;

        emptyFeatureData = [];

        for ind in range(len(featureColumnNames)):
            emptyFeatureData.append(0.0)

        if (len(emptyFeatureData) < 1):
            continue;

        for ind in range(len(cropData.keys())):
            
            if (featureDataIndex >= len(featureData)):
                featureDataFrame = -1;
            else:
                featureDataFrame = int(featureData [featureDataIndex][ findIndex('frame') ])-1

            if (featureDataFrame == ind):
                featureDataNoFrameSkipped.append(featureData [featureDataIndex] )
                featureDataIndex+=1
            else:
                newEmptyFeat = emptyFeatureData[:]
                newEmptyFeat[0] = ind
                featureDataNoFrameSkipped.append( newEmptyFeat )
                

        featureData = featureDataNoFrameSkipped[:]

        cropFrameNumber = -1
        
        print('Face :', shotNum, face_id)


        # Active speaker detection code
        auStackLen = 10;
        auStack = {}

        SPEAKING_SD_THRESHOLD = 0.12
        speakingStack={}
        speakingStackLen = 20


        # for feat in featureData:
        for kk, vv in sorted( cropData.items() ):

            cropFrameNumber+=1
            feat = featureData[cropFrameNumber];
            
            feat = list(map(float, feat))

            frameNum = kk

            cropBox = vv[:]

            cropOrigin = [ cropBox[0], cropBox[1]] # Origin for crop (xmin, ymin)

            # add face track start to get actual frame number
            frameNum += faceTrackStartFrame

            if (cropFrameNumber < len(faceTrackFrameNums) ):
                frameNum = faceTrackFrameNums[cropFrameNumber]
            else:
                continue

            xRanges = []; yRanges = [];
            
            xRanges.append( [findIndex('eye_lmk_x_0'), findIndex('eye_lmk_x_55')] )
            yRanges.append( [findIndex('eye_lmk_y_0'), findIndex('eye_lmk_y_55')] )

            xRanges.append( [findIndex('x_0'), findIndex('x_67')] )
            yRanges.append( [findIndex('y_0'), findIndex('y_67')] )

            feat = addNumToList(feat, cropOrigin[0], xRanges)
            feat = addNumToList(feat, cropOrigin[1], yRanges)

            if (frameNum, shotNum, face_id) not in faceDetections:
                continue;  

            faceBbox = faceDetections[ (frameNum, shotNum, face_id) ]

            # Use AU25 (lips part) to detect speaking person
            currFaceId = face_id
            if ((shotNum, currFaceId) not in auStack):
                auStack[(shotNum, currFaceId)] = [0.0] * auStackLen
            
            if ((shotNum, currFaceId) not in speakingStack):
                speakingStack[(shotNum, currFaceId)] = [0.0] * speakingStackLen

            if ('AU25_r' in featureColumnNames):
                auStack[ (shotNum, currFaceId) ].pop(0)
                auStack[ (shotNum, currFaceId) ].append( feat[ findIndex('AU25_r') ] )

            lipsPartSD = np.std(auStack[ (shotNum, currFaceId) ]);

            speakingStack[ (shotNum, currFaceId) ].pop(0)
            speakingStack[ (shotNum, currFaceId) ].append( lipsPartSD )

            averageSD = np.mean( speakingStack[ (shotNum, currFaceId) ] )
            isSpeaking = 0;
            if (averageSD > SPEAKING_SD_THRESHOLD):
                isSpeaking = 1;



            finalOutput = []

            finalOutput+= [frameNum, shotNum, face_id]
            
            clusterIdentity = -1
            if ( (shotNum, face_id) in faceIdentities ):
                clusterIdentity = faceIdentities[ (shotNum, face_id) ]

            finalOutput+= [clusterIdentity]

            finalOutput+= faceBbox
            
            # set all featues as 0 is 'detection confidence' is 0.0
            if (feat[3] < 0.01):
                feat = list( np.array(feat)*0.0 )

            finalOutput+= feat[3:]
            finalOutput+= [isSpeaking]


            # get depth information
            depthData = -1
            if (depthDataAvailable == True):
                depthFrame = getFrame(depthVideo, frameNum, 0)
                depth_bbox = freeRescaleRoi(faceBbox, [inputVideo_frame_width, inputVideo_frame_height], [depthVideo_frame_width, depthVideo_frame_height])

                faceDepthData = depthFrame[depth_bbox[1]:depth_bbox[3], depth_bbox[0]:depth_bbox[2]]
                if (faceDepthData.shape[0] > 0 and  faceDepthData.shape[1] > 0 ):
                    depthData = np.mean(faceDepthData)/255.0

            finalOutput+= [depthData]

            if frameNum in outputFileData:
                outputFileData[ frameNum ].append(finalOutput);
            else:
                outputFileData[ frameNum ] = [finalOutput];


    outputFilePath = output_folder + '/' + videoFileName +'_output.csv'  
    
    outputFile = open(outputFilePath, 'w');

    outputFile.write(','.join(['StageName', 'Stage_Number', 'Video_File', 'OutputFile']) + '\n');

    outputFile.write(','.join([stage.name, str(stage.number), videoFilePath,
                            outputFilePath]) + '\n');

    # write shot segmentation output
    outputFile.write('\n');

    outputColumnNames += ['frame_number', 'shot_number', 'face_id', 'identity']
    outputColumnNames += ['face_xmin', 'face_ymin', 'face_xmax', 'face_ymax']
    outputColumnNames += featureColumnNames[3:]
    outputColumnNames += ['is_speaking']
    outputColumnNames += ['face_depth']

    outputFile.write(','.join(outputColumnNames) + '\n');


    for frameNumber, frameData in sorted(outputFileData.items()):

        for frameDetection in frameData:
            
            frameDetection = list(map(str, frameDetection));
            outputFile.write(','.join(frameDetection) + '\n');


    inputVideo.release();
    if (depthDataAvailable == True):
        depthVideo.release();

    return outputFilePath







# Visualize output utility functions

AUS = {}

AUS["AU01"] ="Inner Brow Raiser"
AUS["AU02"] ="Outer Brow Raiser"
AUS["AU04"] ="Brow Lowerer"
AUS["AU06"] ="Cheek Raiser"
AUS["AU09"] ="Nose Wrinkler"
AUS["AU12"] ="Lip Corner Puller"

AUS["AU15"] ="Lip Corner Depressor"

AUS["AU20"] ="Lip stretcher"
AUS["AU23"] ="Lip Tightener"
AUS["AU25"] ="Lips part"

AUS["AU45"] ="Blink"


def project2d(pt, angleX, angleY, mag):
    projX = int(mag * math.sin(angleX))
    projY = int(mag * math.sin(angleY))

    projPoint = [ int(projX+pt[0]) , int(projY+pt[1]) ]

    return projPoint


def getROICenter(roi):

    cx = int((roi[0] + roi[2]) / 2)
    cy = int((roi[1] + roi[3]) / 2)

    return [cx, cy]



def resizeROI( roi, roi_scale):

    roi = np.array(roi);

    MW = int((roi[2] - roi[0]) * roi_scale);  # mean width
    MH = int((roi[3] - roi[1]) * roi_scale);  # mean height

    SC = np.array(getROICenter(roi))

    # get smoothened center
    cx = SC[0]
    cy = SC[1]

    xmin = cx - MW / 2
    ymin = cy - MH / 2
    xmax = cx + MW / 2
    ymax = cy + MH / 2

    resizedROI = list(map(int, [xmin, ymin, xmax, ymax] ));

    return resizedROI;

def resizeROIOnlyWidth( roi, roi_scale):

    roi = np.array(roi);

    MW = int((roi[2] - roi[0]) * roi_scale);  # mean width
    MH = int((roi[3] - roi[1]));  # mean height
   
    SC = np.array(getROICenter(roi))

    # get smoothened center
    cx = SC[0]
    cy = SC[1]

    xmin = cx - MW / 2
    ymin = cy - MH / 2
    xmax = cx + MW / 2
    ymax = cy + MH / 2

    resizedROI = list(map(int, [xmin, ymin, xmax, ymax] ));

    return resizedROI;

def convertRectTo4Points(roi):
    points = [ [ roi[0],roi[1] ],
                    [ roi[2],roi[1] ],
                    [ roi[2],roi[3] ],
                    [ roi[0],roi[3] ]
                ]
    return points


def rotate4PointROI(roi, angle):

    cx = (roi[0][0] + roi[2][0])/2
    cy = (roi[0][1] + roi[2][1])/2

    newRoi = []

    for pt in roi:
        px = pt[0]-cx
        py = pt[1]-cy

        rx = px*math.cos(angle) - py*math.sin(angle)
        ry = px*math.sin(angle) + py*math.cos(angle)

        newRoi.append( [int(cx+rx), int(cy+ry)] )

    return newRoi


def visualizeOutput(outputCsvFile, output_folder):
    
    if (os.path.exists(outputCsvFile) == False):
        print('Output csv file not created.')
        return;

    features = get_csv_data(outputCsvFile); 

    stage7_data = features[0:2];
    videoFilePath = stage7_data[1][2];
    
    colNames = features[3]
    features = features[4:];
    colNames = [ elem.strip() for elem in colNames ]
    # remove column names from csv data

    featureDict = {}

    # print (colNames)
    
    def findIndex(colName):
        if (colName in colNames):
            return colNames.index(colName);
        print ('ERROR : feature ', colName, ' not found in extracted features file.')
        return -1;

    def listMapInt(arr):
        arrInt = list(map(int, list(map(float ,arr))))
        return arrInt

    for tt in features:

        frameNum = int(tt[ findIndex('frame_number') ])
        faceID = int(tt[ findIndex('face_id') ])
        faceIdentity = int(tt[ findIndex('identity') ])
        featureConfidence = float(tt[ findIndex('confidence') ])

        shot_number = int(tt[ findIndex('shot_number') ])

        # gaze features
        gazeX = float(tt[ findIndex('gaze_angle_x') ])
        gazeY = float(tt[findIndex('gaze_angle_y')])
        gazeVectors = list(map(float ,  tt[findIndex('gaze_0_x') : 1+findIndex('gaze_1_z')]  ))

        eyeLX = np.mean(list(map(float,tt[ findIndex('eye_lmk_x_20') : 1+findIndex('eye_lmk_x_27') ])))
        eyeLY = np.mean(list(map(float,tt[ findIndex('eye_lmk_y_20') : 1+findIndex('eye_lmk_y_27') ])))

        eyeRX = np.mean(list(map(float,tt[ findIndex('eye_lmk_x_48') : 1+findIndex('eye_lmk_x_55') ])))
        eyeRY = np.mean(list(map(float,tt[ findIndex('eye_lmk_y_48') : 1+findIndex('eye_lmk_y_55') ])))
        

        # head pose features
        poseAngle = list(map(float ,  tt[findIndex('pose_Rx') : 1+findIndex('pose_Rz')]  ))
        
        noseTip = listMapInt([ tt[findIndex('x_33')], tt[findIndex('y_33')] ] )


        faceDetectionBox = listMapInt( tt[findIndex('face_xmin') : 1+findIndex('face_ymax')]  )

        headLeft = listMapInt([ tt[findIndex('x_36')], tt[findIndex('y_36')] ] )
        headRight = listMapInt([ tt[findIndex('x_45')], tt[findIndex('y_45')] ] )
        headTop = listMapInt([ tt[findIndex('x_24')], tt[findIndex('y_24')] ] )
        headBottom = listMapInt([ tt[findIndex('x_8')], tt[findIndex('y_8')] ] )


        headBox = [ min(headLeft[0], headRight[0], headTop[0], headBottom[0]) ,
                    min(headLeft[1], headRight[1], headTop[1], headBottom[1]),
                    max(headLeft[0], headRight[0], headTop[0], headBottom[0]),
                    max(headLeft[1], headRight[1], headTop[1], headBottom[1])    ]

        actionUnits = [ colNames[findIndex('AU01_r') : 1+findIndex('AU45_r')] ,
                       list(map(float, tt[ findIndex('AU01_r') : 1+findIndex('AU45_r') ]  ) ) 
                       ]

        landmarks = [ listMapInt([ tt[findIndex('x_0')], tt[findIndex('y_0')] ] ),
                      listMapInt([ tt[findIndex('x_16')], tt[findIndex('y_16')] ] ),
                      listMapInt([ tt[findIndex('x_4')], tt[findIndex('y_4')] ] ),
                      listMapInt([ tt[findIndex('x_12')], tt[findIndex('y_12')] ] ),
                      listMapInt([ tt[findIndex('x_8')], tt[findIndex('y_8')] ] ),
                      listMapInt([ tt[findIndex('x_48')], tt[findIndex('y_48')] ] ),
                      listMapInt([ tt[findIndex('x_54')], tt[findIndex('y_54')] ] ),
                      listMapInt([ tt[findIndex('x_33')], tt[findIndex('y_33')] ] )
                    ]

        headBox = resizeROIOnlyWidth(headBox, 1.4)
        headBox = [ [ headBox[0],headBox[1] ],
                    [ headBox[2],headBox[1] ],
                    [ headBox[2],headBox[3] ],
                    [ headBox[0],headBox[3] ]
                ]

        eyes = [ eyeLX, eyeLY, eyeRX, eyeRY ]
        gaze = [gazeX , gazeY]

        isSpeaking = int(tt[ findIndex('is_speaking') ])

        face_depth = float(tt[ findIndex('face_depth') ])

        if frameNum in featureDict:
            featureDict[frameNum].append([gaze, eyes, noseTip, headBox, poseAngle, landmarks, actionUnits, [faceID] , faceDetectionBox, featureConfidence, shot_number, faceIdentity, gazeVectors, isSpeaking, face_depth])
        else:
            featureDict[frameNum] = [ [gaze, eyes, noseTip, headBox, poseAngle, landmarks, actionUnits, [faceID], faceDetectionBox, featureConfidence, shot_number , faceIdentity, gazeVectors, isSpeaking, face_depth] ]

    # PV(sorted(frameDict.items()))

    print('videoFilePath : ', videoFilePath)
    vid = cv2.VideoCapture(videoFilePath);

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    print("height:", frame_height, ", width:", frame_width, ", total_frames:", total_frames, ", FPS:", FPS)

    outputVideoPath = output_folder + "/" + videoFilePath.split('/')[-1] + '_visualization.mp4';
    
    outputVideoGroupGazePath = output_folder + "/" + videoFilePath.split('/')[-1] + '_gaze_visualization.mp4';
    
    # outputVideoDepthPath = output_folder + "/" + videoFilePath.split('/')[-1] + '_depthView.mp4';
    
    print('Visualization :', outputVideoPath)
    print('Visualization Group Gaze:', outputVideoGroupGazePath)

    outputVideoSize = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    outputVideo = cv2.VideoWriter(outputVideoPath, fourcc, FPS, outputVideoSize)
    
    outputVideoGroupGaze = cv2.VideoWriter(outputVideoGroupGazePath, fourcc, FPS, outputVideoSize)

    # outputVideoDepth = cv2.VideoWriter(outputVideoDepthPath, fourcc, FPS, outputVideoSize)
    
    checkpointFrame = int(total_frames / 10)


    # display output for each frame
    for ind in range(total_frames):

        if (ind % checkpointFrame == 0):
            print(str(int(ind / checkpointFrame)) + '0%')

        ret, frame = vid.read()

        if (ret == False):
            continue;

        groupGazeFrame = np.copy(frame)

        depthFrame = 0*np.copy(frame)

        gazes = {}


        if ind in featureDict:
            # overlay output

            for face_ in featureDict[ind]:

                gazeAngle = face_[0]
                eyeL = tuple(map(int,[face_[1][0], face_[1][1]]))
                eyeR = tuple(map(int,[face_[1][2], face_[1][3]]))
                eyeCenter = tuple(map(int, [ (eyeL[0]+eyeR[0])/2 , (eyeL[1]+eyeR[1])/2] ) )

                noseTip = face_[2]

                headBox = face_[3]

                poseAngle = face_[4]

                landmarks = face_[5]

                actionUnits = face_[6]

                currFaceId = face_[7][0]

                faceDetectBox = face_[8] 
                featureConfidence = face_[9];

                shotNum = face_[10];

                faceIdentity = face_[11];

                gazeVector = face_[12];

                isSpeaking = face_[13];

                faceDepth = face_[14];


                if (featureConfidence > FEATURE_CONFIDENCE_THRESHOLD) :

                    headRect = headBox[0]+headBox[2]

                    headLength = int(headRect[3]-headRect[1]);
                    gazeMag = headLength/3;
                    boxMag = headLength/1.5;

                    scaledHeadBox = resizeROI( headRect, 1.3)
                    scaledHeadBox = convertRectTo4Points(scaledHeadBox);

                    headBox = rotate4PointROI(headBox, poseAngle[2])
                    scaledHeadBox = rotate4PointROI(scaledHeadBox, poseAngle[2])

                    projectedHeadBox = [ project2d(pnt, -poseAngle[1], poseAngle[0], boxMag)  for pnt in scaledHeadBox ]
                    

                    # draw action units
                    auDrawSeparation = 15; # distance between drawn AU names in pixels

                    if (auDrawSeparation * 6 < headLength):
                        auPoint = [headBox[1][0], headBox[1][1]];

                        for auInd in range(len(actionUnits[0])):
                            if (actionUnits[0][auInd][:-2] in AUS):
                                
                                auName = AUS[ actionUnits[0][auInd][:-2] ]
                                if (actionUnits[1][auInd] > AU_INTENSITY_THRESHOLD):
                                    cv2.putText(frame, auName, tuple(auPoint), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                                    cv2.putText(frame, auName, tuple(auPoint), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                                auPoint[1]+=auDrawSeparation


                    
                    if (isSpeaking == 1):
                        cv2.putText(frame, ')))', (projectedHeadBox[1][0], projectedHeadBox[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, ')))', (projectedHeadBox[1][0], projectedHeadBox[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 230, 10), 2, cv2.LINE_AA)
                    

                    # display depth information
                    # cv2.putText(frame, str( round(faceDepth,2) ), (projectedHeadBox[2][0], projectedHeadBox[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 230, 10), 2, cv2.LINE_AA)
                    xcor = int((headRect[0]+ headRect[2])/2 )
                    zcor = int(faceDepth*frame_height)
                    cv2.circle(depthFrame, (xcor, zcor) ,10, (0, 50, 250), 2, cv2.LINE_AA)

                    # draw short gaze arrow
                    depth_projX = int(frame_width * math.sin(gazeAngle[0]))
                    depth_projY = int(frame_width * math.cos(gazeAngle[0]))

                    # draw eye gaze
                    cv2.line(depthFrame, (xcor,zcor) , (xcor+depth_projX, zcor+depth_projY) ,(0,200,50),2, cv2.LINE_AA)


                    # draw face landmarks

                    for lmrk in landmarks:
                        cv2.circle(frame, tuple(lmrk) ,1, (0, 50, 25), 5, cv2.LINE_AA)
                        cv2.circle(frame, tuple(lmrk) ,1, (0, 200, 50), 2, cv2.LINE_AA)

                    

                    # draw eye centers
                    cv2.circle(frame, eyeL ,2, (0, 255, 100), 2, cv2.LINE_AA)
                    cv2.circle(frame, eyeR ,2, (0, 255, 100), 2, cv2.LINE_AA)

                    # draw gaze vector
                    gaze_projX = int(gazeMag * math.sin(gazeAngle[0]))
                    gaze_projY = int(gazeMag * math.sin(gazeAngle[1]))

                    # draw eye gaze
                    cv2.line(frame,eyeL, (eyeL[0]+gaze_projX, eyeL[1]+gaze_projY) ,(0,50,200),4, cv2.LINE_AA)
                    cv2.line(frame,eyeR, (eyeR[0]+gaze_projX, eyeR[1]+gaze_projY) ,(0,50,200),4, cv2.LINE_AA)

                    # -------------------------------
                    # Group Gaze features
                    # -------------------------------
                    
                    # draw short gaze arrow
                    projX = int(frame_width * math.sin(gazeAngle[0]))
                    projY = int(frame_width * math.sin(gazeAngle[1]))

                    # Joint attention using gaze vectors
                    vec = gazeVector;
                    vec = list( np.mean( [vec[0:3], vec[3:] ],0) )
                    gazes[ currFaceId ] = [currFaceId, list(eyeCenter)+[zcor], vec, headLength, noseTip, projX, projY]


                    # draw farther head box
                    cv2.polylines(frame,[np.array(headBox).reshape((-1,1,2))],True,(150, 50, 0), 2, cv2.LINE_AA)

                    # draw connecting lines
                    for ptNo in range(4):
                        cv2.line(frame, tuple(headBox[ptNo]), tuple(projectedHeadBox[ptNo]) ,(200, 100, 0),2, cv2.LINE_AA)
                    
                    # draw closer box 
                    cv2.polylines(frame,[np.array(projectedHeadBox).reshape((-1,1,2))],True,(250, 150, 0),2, cv2.LINE_AA)

                    # write face_id
                    # cv2.putText(frame, str(currFaceId), (projectedHeadBox[0][0], projectedHeadBox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    # cv2.putText(frame, str(currFaceId), (projectedHeadBox[0][0], projectedHeadBox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)
                    
                    # write face identity
                    cv2.putText(frame, str(faceIdentity), (projectedHeadBox[0][0], projectedHeadBox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(faceIdentity), (projectedHeadBox[0][0], projectedHeadBox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)
                    


                else:
                    cv2.rectangle(frame,(faceDetectBox[0],faceDetectBox[1]), (faceDetectBox[2],faceDetectBox[3]) , (150, 50, 0), 2 , cv2.LINE_AA)

                    # write face_id
                    cv2.putText(frame, str(faceIdentity), (faceDetectBox[0],faceDetectBox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, str(faceIdentity), (faceDetectBox[0],faceDetectBox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)
                
                # write shot number
                cv2.putText(frame, 'Shot ' + str(shotNum), (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5,
                            cv2.LINE_AA)
                cv2.putText(frame, 'Shot ' + str(shotNum), (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2,
                            cv2.LINE_AA)
                


        # find gazes that are looking in same direction

        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'::

                    >>> angle_between((1, 0, 0), (0, 1, 0))
                    1.5707963267948966
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def groupGazes(allGaze):
        # allGaze - dict with entries [currFaceId, eyeCenter, vec, headLength, noseTip]
            gazeGroups = {}
            gazeFocus = {}
            for kk, gg in allGaze.items():
                gazeGroups[ gg[0] ] = [gg[0]]
                gazeFocus[ gg[0] ] = []

            for kk, g1 in sorted(allGaze.items()):

                for kk_, g2 in sorted(allGaze.items()):
                    
                    if (g1[0] == g2[0]):
                        continue
                    # check if vectors diverge or converge
                    c1c2 = np.array(g2[1]) - np.array(g1[1])
                    v1 = np.array(g1[2])
                    v2 = np.array(g2[2])
                    angle_v1_c1c2 = angle_between(v1, c1c2)
                    

                    # find gaze focus on person
                    c2c1 = np.array(g1[1]) - np.array(g2[1])
                    pointDist = np.linalg.norm( np.cross(c2c1, v1) ) / np.linalg.norm(v1);
                    headLen = g2[3]
                    if (pointDist < (headLen/2) ):
                        gazeFocus[ g1[0] ].append( g2[0] )

                    
                    # find gaze groups
                    if (angle_v1_c1c2 > math.pi/2.0):
                        # diverging gaze vectors
                        continue

                    normalVec = np.cross(v1,v2)
                    shortestDistance = abs(np.dot(c1c2, normalVec))

                    minDist = (g1[3]+g2[3])/2
                    # print(g1[0], '-',g2[0] ,' : ',shortestDistance, ' vs ', minDist)

                    if (shortestDistance < minDist and len(gazeGroups[ g1[0] ]) > 0):
                        gazeGroups[g1[0]].append(g2[0])
                        gazeGroups[g2[0]] = []


            groupNum = {}
            groupID = 1;
            for kk,vv in gazeGroups.items():
                if (len(vv) < 1):
                    continue
                if (len(vv) == 1):
                    groupNum[vv[0]] = 0
                    continue

                for ff in vv:
                    groupNum[ff] = groupID

                groupID+=1;

            return groupNum, gazeFocus


        gazeGroupID, gazeFocusFaces = groupGazes(gazes)


        myColors = [ (0,0,0), (0,0,200), (200,0,0), (0,200,0), (200,100,0), (100,200,0), (100,0,200), (200,0,100), (0,200,100), (0,100,200)]

        for faceID_, grpNum in sorted(gazeGroupID.items()):
            color_ = myColors[ grpNum%len(myColors) ]
            nosePnt = gazes[faceID_][4]
            projX = int(5*gazes[faceID_][5])
            projY = int(5*gazes[faceID_][6])

            
            if (grpNum > 0):
                cv2.line(groupGazeFrame,tuple(nosePnt), (nosePnt[0]+projX, nosePnt[1]+projY), (255,255,255), 2, cv2.LINE_AA)
            cv2.line(groupGazeFrame,tuple(nosePnt), (nosePnt[0]+projX, nosePnt[1]+projY), color_, 1, cv2.LINE_AA)

        # find if a gaze is directed at a face
        for faceID_, gazeFocusPerson in sorted(gazeFocusFaces.items()):
            nosePnt_1 = gazes[faceID_][4]

            if (len(gazeFocusPerson) < 1):
                continue

            for focusFace in gazeFocusPerson:
                nosePnt_2 = gazes[focusFace][4]
                headLen_2 = gazes[focusFace][3]
                cv2.line(groupGazeFrame,tuple(nosePnt_1), tuple(nosePnt_2), (0,150,0), 2, cv2.LINE_AA)
                cv2.circle(groupGazeFrame, tuple(nosePnt_2) ,int(headLen_2/8), (0,150,0), 4, cv2.LINE_AA)



        # showImg(frame, 1)
        # showImg(groupGazeFrame, 1)
        # showImg(depthFrame, 1)
        outputVideo.write(frame)
        outputVideoGroupGaze.write(groupGazeFrame)
        # outputVideoDepth.write(depthFrame)


    outputVideo.release()
    outputVideoGroupGaze.release()
    # outputVideoDepth.release()

    vid.release()

stage = stages.stage8

input_folder = removeTrailingBackslash(stage.inputLocation)
output_folder = removeTrailingBackslash(stage.outputLocation)

print('Begin Stage ', stage.number, ' : ', stage.name)

assert os.path.exists(input_folder), "Stage input folder : " + input_folder + " not found."
assert os.path.exists(output_folder), "Stage output folder : " + output_folder + " not found."


faceCropStage = stages.stage4
featureExtractionStage = stages.stage6
faceIdentityStage = stages.stage7
depthOutputStage = stages.stage9

# process files
for f in os.listdir(input_folder):
    filePath = os.path.join(input_folder, f)

    # process csv files
    if (filePath[-4:] == '.csv'):
        print('Processing: ', filePath)

        outputFile = createOutputFile( filePath, output_folder, faceCropStage.outputLocation, featureExtractionStage.outputLocation, faceIdentityStage.outputLocation, depthOutputStage.outputLocation)
        
        visualizeOutput(outputFile, output_folder);

print('End Stage ', stage.number, ' : ', stage.name)
