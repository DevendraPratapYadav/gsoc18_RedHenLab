source activate your_env_name

echo $1','$2 > config.csv
bash cleanOutput.sh

echo 'Runtimes : ' > runtime.txt
start=`date +%s`
python shot_segmentation.py
end=`date +%s`
echo 'Time taken - shot_segmentation : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python face_detection.py --method=mtcnn --tinyFaces_scale=360 --detection_interval=0.5
end=`date +%s`
echo 'Time taken - face_detection : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python face_tracking.py
end=`date +%s`
echo 'Time taken - face_tracking : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python face_cropping.py
end=`date +%s`
echo 'Time taken - face_cropping : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python face_alignment.py --method=openface --output_size=224
end=`date +%s`
echo 'Time taken - face_alignment : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python feature_extraction.py
end=`date +%s`
echo 'Time taken - feature_extraction : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python face_clustering.py
end=`date +%s`
echo 'Time taken - face_clustering : ' $((end-start)) ' sec' >> runtime.txt

start=`date +%s`
python process_output.py
end=`date +%s`
echo 'Time taken - process_output : ' $((end-start)) ' sec' >> runtime.txt

