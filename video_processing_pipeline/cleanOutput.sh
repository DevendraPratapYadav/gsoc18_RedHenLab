#!/bin/bash

today=`date '+%Y_%m_%d__%H_%M_%S'`;
# today='8vids'
echo $today

mv 1_shot_segmentation/output 1_shot_segmentation/output_$today
mkdir 1_shot_segmentation/output
echo 'Renamed to 1_shot_segmentation/output_'$today 

mv 2_face_detection/output 2_face_detection/output_$today
mkdir 2_face_detection/output
echo 'Renamed to 2_face_detection/output_'$today 

mv 3_face_tracking/output 3_face_tracking/output_$today
mkdir 3_face_tracking/output
echo 'Renamed to 3_face_tracking/output_'$today 

mv 4_face_cropping/output 4_face_cropping/output_$today
mkdir 4_face_cropping/output
echo 'Renamed to 4_face_cropping/output_'$today 

mv 5_face_alignment/output 5_face_alignment/output_$today
mkdir 5_face_alignment/output
echo 'Renamed to 5_face_alignment/output_'$today 

mv 6_feature_extraction/output 6_feature_extraction/output_$today
mkdir 6_feature_extraction/output
echo 'Renamed to 6_feature_extraction/output_'$today 

mv 7_face_clustering/output 7_face_clustering/output_$today
mkdir 7_face_clustering/output
echo 'Renamed to 7_face_clustering/output_'$today 

# mv 8_process_output/output 8_process_output/output_$today
# mkdir 8_process_output/output
# echo 'Renamed to 8_process_output/output_'$today 

