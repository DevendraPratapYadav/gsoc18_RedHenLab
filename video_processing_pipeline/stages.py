#*************************************************
# Dependencies for video processing pipeline modules
#*************************************************


import os
import pathlib
import csv


OpenFace_FaceLandmarkVidMulti_path = '/home/deeplearning/Desktop/GSOC_18/modules/OpenFace_scripts/OpenFace/build/bin/FaceLandmarkVidMulti'


class Stage:
    name = None
    number = None
    inputLocation = None
    outputLocation = None
    dependencies = {}

    def __init__(self,name,number):
        self.name = name
        self.number = number

def printStage(stage):
    print('Stage Name : ',stage.name)
    print('Stage Number : ', stage.number)
    print('Input Location : ', stage.inputLocation)
    print('Output Location : ', stage.outputLocation)
    print('Dependencies : ', stage.dependencies)
    print()

def get_csv_data(csvFile):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data

def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


"""
argparse = ArgumentParser()

argparse.add_argument('--i', type=str, help='Path to input folder with videos' , default='')
argparse.add_argument('--o', type=str, help='Path to output folder', default='')

args = argparse.parse_args() 
"""



# GLOBAL PARAMETERS

# current working directory where all script files are placed. Intermediate working folders for each stage are created within this directory.
SCRIPT_WORKING_DIRECTORY = os.getcwd()

MAIN_INPUT_DIR = ''
MAIN_OUTPUT_DIR = ''

config_file_path = os.path.join(SCRIPT_WORKING_DIRECTORY,'config.csv')

if (os.path.exists(  config_file_path  )):
    folderConfig = get_csv_data(config_file_path)
    if (len(folderConfig) > 0):
        if (len(folderConfig[0]) > 0):
            MAIN_INPUT_DIR = removeTrailingBackslash(folderConfig[0][0])
        if (len(folderConfig[0]) > 1):
            MAIN_OUTPUT_DIR = removeTrailingBackslash(folderConfig[0][1])
    



# STAGE WISE PARAMETERS

stage0 = Stage('Input', 0)
stage0.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'0_input')
if(MAIN_INPUT_DIR != ''):
    stage0.outputLocation = MAIN_INPUT_DIR

stage1 = Stage('Shot Segmentation', 1)
stage1.inputLocation = stage0.outputLocation
stage1.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'1_shot_segmentation/output')

stage2 = Stage('Face Detection', 2)
stage2.inputLocation = stage1.outputLocation
stage2.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'2_face_detection/output')
stage2.dependencies['tinyfaces_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'2_face_detection/libs/tiny_faces')
stage2.dependencies['tinyfaces_weights_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'2_face_detection/libs/tiny_faces/hr_res101.pkl')


stage3 = Stage('Face Tracking', 3)
stage3.inputLocation = stage2.outputLocation
stage3.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'3_face_tracking/output')
stage3.dependencies['face_track_script_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'3_face_tracking/face_track.py')


stage4 = Stage('Face Cropping', 4)
stage4.inputLocation = stage3.outputLocation
stage4.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'4_face_cropping/output')
stage4.dependencies['face_crop_script_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'4_face_cropping/faceCropVideo.py')

stage5 = Stage('Face Alignment', 5)
stage5.inputLocation = stage4.outputLocation
stage5.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'5_face_alignment/output')
stage5.dependencies['faceAlign_openface_script_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'5_face_alignment/alignFace_OpenFace.py')
stage5.dependencies['faceAlign_manual_script_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'5_face_alignment/alignFace.py')
stage5.dependencies['dlib_faceLandmarks_model_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'5_face_alignment/libs/shape_predictor_68_face_landmarks.dat')
stage5.dependencies['openface_FaceLandmarkVidMulti_path'] = OpenFace_FaceLandmarkVidMulti_path

stage6 = Stage('Feature Extraction', 6)
stage6.inputLocation = stage4.outputLocation
stage6.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'6_feature_extraction/output')
stage6.dependencies['feature_extraction_script_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'6_feature_extraction/processVideosWithOpenFace.py')
stage6.dependencies['openface_FaceLandmarkVidMulti_path'] = OpenFace_FaceLandmarkVidMulti_path

stage7 = Stage('Face Clustering', 7)
stage7.inputLocation = stage5.outputLocation
stage7.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'7_face_clustering/output')
stage7.dependencies['dlib_faceLandmarks_model_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'5_face_alignment/libs/shape_predictor_68_face_landmarks.dat')
stage7.dependencies['dlib_faceRecognition_model_path'] = os.path.join(SCRIPT_WORKING_DIRECTORY,'7_face_clustering/libs/dlib_face_recognition_resnet_model_v1.dat')

stage8 = Stage('Process Output', 8)
stage8.inputLocation = stage3.outputLocation
stage8.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'8_process_output/output')
if(MAIN_OUTPUT_DIR != ''):
    stage8.outputLocation = MAIN_OUTPUT_DIR

stage9 = Stage('Depth Output', 9)
stage9.outputLocation = os.path.join(SCRIPT_WORKING_DIRECTORY,'9_depth_output')


# check existence of all paths
allStages = [stage0, stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9]
print('Checking paths for all stages.')

for st in allStages:

    # print('Stage ', st.number);
    iPath = st.inputLocation;
    oPath = st.outputLocation;

    if (iPath!=None and os.path.exists(iPath) == False):
        print(iPath, 'does not exist. Creating folders.')
        pathlib.Path(iPath).mkdir(parents=True, exist_ok=True) 

    if (oPath!=None and os.path.exists(oPath) == False):
        print(oPath, 'does not exist. Creating folders.')
        pathlib.Path(oPath).mkdir(parents=True, exist_ok=True) 

    for depName, depPath in st.dependencies.items():
        if (os.path.exists(depPath) == False):
            print('ERROR : ', depPath, 'does not exist.')

