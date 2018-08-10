source activate your_env_name
echo $1','$2 > config.csv
bash cleanOutput.sh
python shot_segmentation.py
python face_detection.py --method=mtcnn --tinyFaces_scale=360 --detection_interval=0.5
python face_tracking.py
python face_cropping.py
python face_alignment.py --method=openface --output_size=224
python feature_extraction.py
python face_clustering.py
python process_output.py
