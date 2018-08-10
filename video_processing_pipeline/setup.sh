echo 'Installing dependencies for Video Pipeline scripts'

# Create new anaconda virtual environment
conda create -y -n videopipelineenv python=3.5 anaconda
source activate videopipelineenv

# install OpenCV python
conda install -y -c menpo opencv
yes | pip install opencv-contrib-python

# install keras with tensorflow backend
conda install -y -c anaconda tensorflow-gpu
conda install -y keras

# Install MTCNN
yes | pip install mtcnn

# Install pyannote-video :
yes | pip install pyannote-video
conda update -y sortedcollections
conda install -y sortedcontainers==1.5.9

# Install OpenFace
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
echo "OpenFace installation files have been downloaded in OpenFace folder."
echo "Please run ./download_models.sh, then run ./install.sh to setup OpenFace. This requires sudo access to install."

echo "NOTE : FaceLandmarkVidMulti program will be built in <OpenFace_Directory>/build/bin. Provide the path to this program in the dependencies file stages.py as string OpenFace_FaceLandmarkVidMulti_path"