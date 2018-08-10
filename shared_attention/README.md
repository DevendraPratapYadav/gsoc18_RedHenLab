This project aims to implement a deep neural network model to detect a person's gaze location in an image.  
The models are based on the papers 'Where are they looking?' by Recasens et al and 'Inferring Shared Attention in Social Scene Videos?' by Fan et al.

## Dependencies
1\. Keras with tensorflow backend
2\. OpenCV python
3\. Python 3.5 or above. Python Anaconda distribution preferred

## Setup:

1. VGG16_hybrid_places_1365 pre-trained model-  
https://github.com/GKalliatakis/Keras-VGG16-places365
Link : https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'
Place at : Same directory as vgg16_hybrid_places_1365.py

2. GazeFollow dataset:  
http://gazefollow.csail.mit.edu/download.html  
NOTE : The dataset provides bounding box for person instead of head. We must crop 30% of the percent of area around eye location from person bounding box to extract head image

3. VideoCoAtt dataset:  
http://www.stat.ucla.edu/~lifengfan/shared_attention  
The original dataset provides bounding box for region of shared attention. The dataset annotations have been converted to similar format as GazeFollow dataset and these annotation files are present in 'VideoCoAtt_Dataset' folder.  

For both dataset image paths in annotations are assumed to be relative to annotation file directory. Please place dataset folders such that annotation files contain relative path to images. Eg. place 'images', 'videos' folders of VideoCoAtt dataset along with annotation file 'train_list.txt'



## Usage:

Place the model definition and training code in root directory with dataset folders and vgg16_hybrid_places_1365.py.
Some models implemented in Keras are available in `models` folders

### Training
```
$ python train_gazeFollow.py <train/test> <train_annotations_file_path> <validation_annotations_file_path> <test_annotations_file_path> <pre-trained_model_weights_path>
```

Example : 
```
 $ python train_gazeFollow.py train VideoCoAtt_Dataset/train_list.txt VideoCoAtt_Dataset/validate_list.txt VideoCoAtt_Dataset/test_list.txt data/checkpoints/best_03-42-18_noconv_88acc.hdf5
```


### Test and output heatmap visualization:
```
$ python test_gazeFollow.py test <train_annotations_file_path> <validation_annotations_file_path> <test_annotations_file_path> <pre-trained_model_weights_path>
```

Heatmap images are placed in `visualization_heatmap` folder. Image order from top to bottom : input_image, gaze_heatmap, saliency_heatmap, multiplied_heatmap, ground_truth_heatmap. In input image, head location (blue), true gaze (red) and predicted gaze (green) are marked with circles


## Head detection

Both GazeFollow and VideoCoAtt datasets provide head bounding box annotation. However, for testing on new images, we need head bounding box for the person whose gaze location we want to predict.  
A head detector can be easily trained using code available at : https://github.com/pranoyr/head-detection-using-yolo  
Using InceptionV3 as backend provides good performance with reasonable speed. (around 30 ms per image on GTX Titan V)


## Future work

The current implemented models overfit, with high training accuracy and very low test accuracy. As of now, the model outputs are not usable. More testing and model modifications need to be done to improve performance.

Current training/testing is done on the converted VideoCoAtt dataset. The generator function must be modified to read annotations of GazeFollow dataset.

## Code structure

.
├── data
│   └── checkpoints
│   
├── models
│   ├── compact_model
│   │   ├── compact_model.py
│   │   ├── model.png
│   │   └── train_compact_model.py
│   ├── gazefollow_model_3losses
│   │   ├── gazefollow_keras_3losses.py
│   │   ├── model.png
│   │   ├── test_gazeFollow_3losses.py
│   │   ├── train_gazeFollow_3losses.py
│   │   └── visualization
│   │       ├── visualization_heatmap_test
│   │       └── visualization_heatmap_train
│   └── gazefollow_paper_model
│       ├── gazefollow_paper.py
│       ├── model.png
│       ├── test_gazeFollow_paper.py
│       ├── train_gazeFollow_paper.py
│       ├── train_log.png
│       ├── train_log.txt
│       └── visualization
│           ├── visualization_heatmap_test
│           └── visualization_heatmap_train
│
├── README.md
├── vgg16_hybrid_places_1365.py
├── VideoCoAtt_Dataset
│   ├── test_list.txt
│   ├── train_list.txt
│   └── validate_list.txt
└── visualization_heatmap
