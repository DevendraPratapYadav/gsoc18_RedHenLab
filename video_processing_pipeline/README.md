# Video Processing Pipeline

## Overview

This project contains a modular pipeline to extract several face related features from videos. It takes a folder with videos as input and generates csv output and its visualization videos in a specified output folder.
Each stage places its output in its folder, which is used by the later stages. Stage folders are named as `stageNumber_stageName`.

The pipeline consists of 8 modules/stages :

1. Shot segmentation - Determine shot boundaries in video   
2. Face detection - Detect bounding box for faces in video at some interval  
3. Face tracking - Track the detected faces across subsequent frames  
4. Face cropping - Crop the tracked faces to obtain a video  
5. Face alignment - Perform face alignment for each frame in cropped face video  
6. Feature extraction - Extract face features such as face landmarks, eye gaze direction, Action Units, head pose etc. using OpenFace library  
7. Face clustering - Perform face clustering to determine unique identity for each aligned face video obtained in stage 5 - Face alignment  
8. Process output - Process outputs obtained from previous stages to create single csv file containing extracted face features. Create a video visualizing these features.  

### Features extracted

68 face landmarks, eye gaze direction, head pose, speaker state (is a person speaking), person identity and selected action units



## Dependencies:

1. **Linux / Ubuntu**  

2. **Python 3.5 (Anaconda distribution preferred)**  

3. **OpenFace**  
    Install using instructions given here:
    https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation    
  
4. **OpenCV-Python 3.3 or above**  
    Preferred command : conda install -c menpo opencv


5. **Pyannote-video**  
    Link : https://github.com/pyannote/pyannote-video


6. **MTCNN - Tensorflow**  
    https://github.com/ipazc/mtcnn


7. **Finding Tiny Faces - Tensorflow**  
    https://github.com/cydonia999/Tiny_Faces_in_Tensorflow.  
    NOTE : Download trained model from link given in setup instructions


8. **ffmpeg**  
    Preferred command : conda install -c menpo ffmpeg 


9. **dlib**  
    Models needed : 'shape_predictor_68_face_landmarks.dat', 'dlib_face_recognition_resnet_model_v1.dat'  
    Download from : https://github.com/davisking/dlib-models. Provide path to models in 'stages.py'  


10. **Other libraries** : numpy, scipy, pandas, pylab  


## Dependency installation :

### Using setup script
1. Run the setup script 'setup.sh' to install needed libraries. 

2. Go to the newly created 'OpenFace' folder in current directory and install OpenFace as per instructions on https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation. Update the path to 'FaceLandmarkVidMulti' in 'stages.py' as string 'OpenFace_FaceLandmarkVidMulti_path'

### Manual setup

If you encounter any errors in using the setup script, you can manually install dependencies as follows: 

1\. Download and install Anaconda  

2\. Create new anaconda virtual environment  
```
conda create -n videopipelineenv python=3.5 anaconda
source activate videopipelineenv
```
3\. Install OpenCV python  
```
conda install -c menpo opencv
pip install opencv-contrib-python
```
4\. Install keras with tensorflow backend  
```
conda install -c anaconda tensorflow-gpu
conda install keras
```
5\. Install MTCNN  
```
pip install mtcnn
```
6\. Install pyannote-video  
```
pip install pyannote-video
conda update sortedcollections
conda install sortedcontainers==1.5.9
```
7\. Install OpenFace  
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
```

Please run ./download_models.sh, then run ./install.sh to setup OpenFace. This requires sudo access to install.  
FaceLandmarkVidMulti program will be built in <OpenFace_Directory>/build/bin. Provide the path to this program in the dependencies file stages.py as string 'OpenFace_FaceLandmarkVidMulti_path'  


### Download required models and configure script

1\. Tiny-faces trained weights:  
Link : https://drive.google.com/file/d/1HtvrinoiNHTOquF5SHBWxXjUHlC8XbL4/view?usp=sharing  
Place at : 2_face_detection/libs/tiny_faces/hr_res101.pkl  


2\. Dlib shape predictor/landmark detector:  
Link : https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2  
Place at : 5_face_alignment/libs/shape_predictor_68_face_landmarks.dat  

3\. Dlib face recognition model:  
Link : https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2  
Place at : 7_face_clustering/libs/dlib_face_recognition_resnet_model_v1.dat  


4\. Set name of conda virtual environment in which dependencies have been setup in scripts 'execute_pipeline.sh' and 'execute_pipeline_timed.sh'. Edit the first line `source activate <your_env_name>`  

5\. Provide the path to `FaceLandmarkVidMulti` program built with OpenFace in the dependencies file 'stages.py' as string 'OpenFace_FaceLandmarkVidMulti_path'  


## Script description:
`execute_pipeline.sh` takes input and output folder as argument and executes each pipeline stage sequentially. `execute_pipeline_timed.sh` works similarly but also writes execution time for each stage in `runtime.txt`. Any missing output folders for each stage are created automatically. Missing dependencies (eg. dlib model) are reported as errors. All output files are placed in the specified output folder. Output folder location and library locations for each stage are configured in 'stages.py'

## Run as:

Activate the conda environment or Python virtualenv where dependencies have been setup.

Then execute the script:
```
$ ./execute_pipeline.sh <input_folder_path> <output_folder_path>
```
NOTE : Each stage stores its outputs in `stageNumber_stageName/output`. Upon pipeline execution, `cleanOutput.sh` renames these intermediate output folders for each stage by adding current time as suffix and creates new empty `output` folders. After pipeline processing is done, you may delete all folders with prefix `output_` to free space. You can use `removeIntermediateOutputs.sh` to delete these folders.

### Stage processing options

The following arguments can be modified as per user requirements : 
#### 1\. Face detection
--method : Algorithm for face detection. Options: `mtcnn`, `tiny_faces`  
--tinyFaces_scale : Width to which image should be resized before detection when using `tiny_faces` method. Smaller values are faster but less accurate  
--detection_interval : (float) At what interval(in seconds) should we perform face detection  

#### 2\. Face alignment
--method : Landmark/method to use for face alignment. Options : `dlib`, `mtcnn`, `openface`  
--output_size : Size of aligned square face video  



## NOTE:
1\. For face alignment with OpenFace, all video frames are extracted as .bmp files while processing. Keep sufficient free disk space for these files. 
Approximate Size calculation : `(Num_frames * Num_faces_per_frame * 150) KB`. 
After processing with OpenFace is completed, the script automatically deletes these bmp images and creates a video for each face.  

2\. Please remove any spaces in folder and file names




## Code structure

```
.
├── 0_input
├── 1_shot_segmentation
│   └── output
├── 2_face_detection
│   ├── libs
│   │   └── tiny_faces
│   │       ├── tiny_face_model.py
│   │       └── util.py
│   └── output
├── 3_face_tracking
│   ├── face_track.py
│   └── output
├── 4_face_cropping
│   ├── faceCropVideo.py
│   └── output
├── 5_face_alignment
│   ├── alignFace_OpenFace.py
│   ├── alignFace.py
│   ├── libs
│   └── output
├── 6_feature_extraction
│   ├── output
│   └── processVideosWithOpenFace.py
├── 7_face_clustering
│   ├── libs
│   └── output
├── 8_process_output
│   └── output
├── 9_depth_output
├── cleanOutput.sh
├── config.csv
├── docs
│   └── Stage_interfaces.pdf
├── execute_pipeline.sh
├── execute_pipeline_timed.sh
├── face_alignment.py
├── face_clustering.py
├── face_cropping.py
├── face_detection.py
├── face_tracking.py
├── feature_extraction.py
├── process_output.py
├── README.md
├── removeIntermediateOutputs.sh
├── setup.sh
├── shot_segmentation.py
└── stages.py

```


