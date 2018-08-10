#*************************************************
# Program to perform face alignment in video using OpenFace
#*************************************************


import os
import sys
import subprocess
import cv2


# number of videos to process at once with openface. 
# batch size of 1 is preferred since error with any video in batch stops processing of other videos as well
# Set with consideration of disk size of extracted .bmp video frames 
BATCH_SIZE = 1; 
OUTPUT_VIDEO_FRAME_SIZE = 224;
OUTPUT_FPS = 24


def PV(arr):
    for x in arr:
        print (x);


def processVideos(outPath, files, openfacePath):
    
    # Prepare OpenFace command
    videoPaths = "";
    c1 = '"{openfacePath}" {videos}-out_dir '+ outPath +' -simalign -nomask -simsize '+str(OUTPUT_VIDEO_FRAME_SIZE);

    # add names of videos in batch
    for f in range(0,len(files)):
        videoPaths+='-f "'+files[f]+'" ';

    # execute OpenFace command
    com1 = c1.format(videos= videoPaths , openfacePath = openfacePath);
    print(com1)
    subprocess.call(com1, shell=True)

    print(" ")


    # process each video one by one
    for f in range(0,len(files)):
        
        
        alignedVideoPath = outPath+'/'+files[f].split('/')[-1][:-4]+'_aligned'
        print("alignedVideoPath >> ",alignedVideoPath)
        
        if (os.path.exists(alignedVideoPath) == False):
            print("ERROR : alignedVideoPath-",alignedVideoPath, 'does not exist. Possibly OpenFace program was terminated before completion due to lack of memory.')
            continue;
            
        # find number of faces in video shot
        numFaces = 0;
        for i in os.listdir(alignedVideoPath):
            numFaces = max(numFaces , int(i.split('_')[2]));
        numFaces+=1;
        # print("numFace>>", numFaces)
    
        outputVideos = [None]*(numFaces);
        
        # create video files
        for v in range(numFaces):

            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            videoName = outPath+ '/'+ files[f].split('/')[-1] +'_aligned.mp4';
            outVideoSize = (OUTPUT_VIDEO_FRAME_SIZE,OUTPUT_VIDEO_FRAME_SIZE)
            FPS = OUTPUT_FPS
            out = cv2.VideoWriter(videoName,fourcc, FPS, outVideoSize)
            
            outputVideos[v] = out


        frames = os.listdir(alignedVideoPath);

        # writing video from images
        for i in sorted(frames):
            img = cv2.imread(alignedVideoPath +'/'+ i);
            numFace = int(i.split('_')[2])
            
            outputVideos[numFace].write(img)

        for v in range(numFaces):
            outputVideos[v].release()

        # remove bmp images
        # com2 = 'rm -r "' + alignedVideoPath+'"/*.bmp';
        com2 = 'rm -r "' + alignedVideoPath+'"'; # 
        print(com2);
        print('INFO : Deleting temporary directory - ', alignedVideoPath)
        subprocess.call(com2, shell=True)


        for f in os.listdir(outPath):
            if (f[-4:] == '.csv' or f[-4:] == '.txt'):
                fileRemovePath = outPath+'/'+f
                os.remove(fileRemovePath)


def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];
        
    return path;


if (len(sys.argv)<4):
    print('Usage: python alignFace_OpenFace.py <FaceLandmarkVidMulti program path> <input videos folder> <output folder>')
    print('Example : python alignFace_OpenFace.py OpenFace/build/bin/FaceLandmarkVidMulti videos processed');
    sys.exit();


files = [];
openface_program_path = removeTrailingBackslash( sys.argv[1] );
input_folder = removeTrailingBackslash( sys.argv[2] );
output_folder = removeTrailingBackslash( sys.argv[3] );

# openface_program_path = './'+openface_program_path;


# process videos in batches
for f in os.listdir(input_folder):
    # print (f);

    if (f[-4:] != '.csv'):
        files.append(input_folder+'/'+f)

    if (len(files)>=BATCH_SIZE):
        processVideos(output_folder, files, openface_program_path);
        files = [];

# process remaining files in batch
if (len(files)>0):
    processVideos(output_folder, files, openface_program_path);
    files = [];

print ('\nProcessing complete.\n')


