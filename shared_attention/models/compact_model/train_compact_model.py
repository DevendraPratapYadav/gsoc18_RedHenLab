import time
import os
import os.path
import numpy as np
import sys
import pdb
import csv
import cv2
from random import shuffle


# USE CPU ONLY
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""



from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
np.set_printoptions(threshold=np.inf)







import compact_model



OUTPUT_GRID_SIZE = 5
INTERMEDIATE_GRID_SIZE = 14


INPUT_IMAGE_SHAPE = (224, 224, 3)
HEAD_IMAGE_SHAPE = (224, 224, 3)
HEAD_LOCATION_SHAPE = (INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE,1)


def PV(arr):
    for x in arr:
        print (x);

def get_csv_data(csvFile, split = -1, pos = 'b'):
    with open(csvFile, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)


    if (split>0):
        if (pos == 'b'):
            data = data[0:int(len(data)*split)]
        else:
            data = data[int(len(data)*split):]

    return data


def showImg(img, wait, windowName = 'image'):
    cv2.imshow(windowName, img)
    cv2.waitKey(wait)

def shuffleTogether4(arr1, arr2, arr3, arr4):
    
    # Shuffle arrays together
    shuf = np.arange(0,len(arr1));
    np.random.shuffle(shuf);

    comb=zip(shuf,arr1);
    comb=sorted(comb);
    arr1=[x[1] for x in comb];

    comb=zip(shuf,arr2);
    comb=sorted(comb);
    arr2=[x[1] for x in comb];

    comb=zip(shuf,arr3);
    comb=sorted(comb);
    arr3=[x[1] for x in comb];

    comb=zip(shuf,arr4);
    comb=sorted(comb);
    arr4=[x[1] for x in comb];
    
    return np.array(arr1), np.array(arr2), np.array(arr3), np.array(arr4)



def shuffleTogether(arr1, arr2, arr3, arr4, arr5):
    
    # Shuffle arrays together
    shuf = np.arange(0,len(arr1));
    np.random.shuffle(shuf);

    comb=zip(shuf,arr1);
    comb=sorted(comb);
    arr1=[x[1] for x in comb];

    comb=zip(shuf,arr2);
    comb=sorted(comb);
    arr2=[x[1] for x in comb];

    comb=zip(shuf,arr3);
    comb=sorted(comb);
    arr3=[x[1] for x in comb];

    comb=zip(shuf,arr4);
    comb=sorted(comb);
    arr4=[x[1] for x in comb];
    
    comb=zip(shuf,arr5);
    comb=sorted(comb);
    arr5=[x[1] for x in comb];

    return np.array(arr1), np.array(arr2), np.array(arr3), np.array(arr4), np.array(arr5)






def getGridCellNumber(point, gridSize):
    cellSize = 1.0/float(gridSize)

    cellX = int(float(point[0])/cellSize)
    cellY = int(float(point[1])/cellSize)

    cellNumber = cellX + gridSize*cellY
    return cellNumber



def getGridLocation(cellNumber, gridSize):
    cellSize = 1.0/float(gridSize)

    cellX = (cellNumber%gridSize)*cellSize + cellSize/2.0
    cellY = (int(cellNumber/gridSize) )*cellSize + cellSize/2.0
    return [cellX,cellY]


def cropImage(img, bbox):

    height = img.shape[0]
    width = img.shape[1]
    bbox = bbox[:]
    bbox[0] = max(0, bbox[0]*width)
    bbox[2] = min(width, bbox[2]*width)
    bbox[1] = max(0, bbox[1]*height)
    bbox[3] = min(height, bbox[3]*height)

    bbox = list(map(int,bbox))
    # print(bbox)
    # return img[ bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2] ]
    new_img = np.copy(img)
    return new_img[ bbox[1]:bbox[3], bbox[0]:bbox[2] ]




def loadVGGImage(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x[0]


def preprocess_VGG_Imagenet(img):
    img[:,:,0] -= 124 # B
    img[:,:,1] -= 117 # G
    img[:,:,2] -= 104 # R

    return img

def preprocess_VGG_Places(img):
    img[:,:,0] -= 123 # B
    img[:,:,1] -= 117 # G
    img[:,:,2] -= 104 # R

    return img


def createGazeMap(eye, gaze):
    gazeMap = np.zeros((INTERMEDIATE_GRID_SIZE, INTERMEDIATE_GRID_SIZE))

    eye = np.array(np.copy(eye))
    gaze = np.array(np.copy(gaze))
    
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        if (np.linalg.norm(vector) < 0.0000001):
            print('zero norm vector encountered')
            return vector
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    for yy in range(INTERMEDIATE_GRID_SIZE):
        for xx in range(INTERMEDIATE_GRID_SIZE):
            cellSize = 1.0/float(INTERMEDIATE_GRID_SIZE)
            pnt = np.array([ (xx+0.5)*cellSize , (yy+0.5)*cellSize ])
            angle = angle_between( (eye-gaze), (eye-pnt) )

            value = gaussian( angle, 0.0, 0.25 )
            gazeMap[yy][xx] = round(value,3)

    # print(np.max(gazeMap))
    return gazeMap

def train_generator(data_file, split = -1, pos = 'b', doShuffle = False, batchSize = -1):
    # for VideoCoAtt dataset
    # functions as test generator as well using appropriate arguments
    # data_file : txt file with list of training images and their annotations
    


    data = get_csv_data(data_file, split, pos)

    root_folder = '/'.join(data_file.split('/')[:-1])
    ind = 0
    num_samples = len(data)

    if (doShuffle == True):
        print('Shuffling data')
        shuffle(data)

    if (batchSize > 0):
        BATCH_SIZE = batchSize

    while True:
        
        batch_input_images = []
        batch_head_images = []
        batch_head_locations = []
        batch_gaze_locations = []
        batch_gaze_heatmaps = []
        batch_int_gaze_heatmaps = []

        for ii in range(ind, ind+BATCH_SIZE):

            idx = ii%num_samples
            # print(data[idx])
            image_path = os.path.join(root_folder, data[idx][2])
            image_number = int(data[idx][0])
            head_bbox = list(map(float,data[idx][7:11]))
            gaze_coordinates = list(map(float,data[idx][3:5]))
            eye_coordinates = list(map(float,data[idx][5:7]))

            # print(image_path)
            input_image = cv2.imread(image_path)
            # input_image = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
            # input_image = input_image.astype('uint8')
            # print(input_image[100:110, 100:110,:])
            head_image = cropImage(input_image, head_bbox)
            head_location = np.array(to_categorical( getGridCellNumber(eye_coordinates, INTERMEDIATE_GRID_SIZE),
                                             INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE ) )
            head_location = np.array(head_location).reshape(-1,1)
            gaze_location = getGridCellNumber(gaze_coordinates, OUTPUT_GRID_SIZE)
            # gaze_location = to_categorical(gaze_location, )
            
            # gaze_heatmap = np.zeros((INTERMEDIATE_GRID_SIZE,INTERMEDIATE_GRID_SIZE))
            # intermediate_gaze_location = getGridCellNumber(gaze_coordinates, INTERMEDIATE_GRID_SIZE)
            # gaze_heatmap[ int(intermediate_gaze_location/INTERMEDIATE_GRID_SIZE) ] [int(intermediate_gaze_location%INTERMEDIATE_GRID_SIZE)] = 1.0
            gaze_heatmap = to_categorical( getGridCellNumber(gaze_coordinates, INTERMEDIATE_GRID_SIZE),
                                             INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE )
            vis_gaze_heatmap = gaze_heatmap.reshape((INTERMEDIATE_GRID_SIZE, INTERMEDIATE_GRID_SIZE))
            vis_gaze_heatmap = cv2.GaussianBlur(vis_gaze_heatmap,(3,3),0)
            vis_gaze_heatmap = vis_gaze_heatmap/np.max(vis_gaze_heatmap)
            gaze_heatmap = np.copy(vis_gaze_heatmap.flatten())
            # vis_gaze_heatmap = cv2.resize(vis_gaze_heatmap, (INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[0]), interpolation = cv2.INTER_AREA)
            # showImg( vis_gaze_heatmap, 0, 'gazeMap')

            # int_gaze_heatmap = createGazeMap(eye_coordinates, gaze_coordinates).flatten()
            int_gaze_heatmap = createGazeMap(eye_coordinates, gaze_coordinates)


            """
            ### TEST gaze map output ###################
            showImg( cv2.resize(int_gaze_heatmap, (INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[0]), interpolation = cv2.INTER_AREA), 0, 'gazeMap')

            cv2.line(input_image,(int(eye_coordinates[0]*input_image.shape[1]), int(eye_coordinates[1]*input_image.shape[0]) ), 
                (int(gaze_coordinates[0]*input_image.shape[1]) ,  int(gaze_coordinates[1]*input_image.shape[0]) ), 
                (0,50,200),4, cv2.LINE_AA)

            cellLoc = getGridLocation(gaze_location, OUTPUT_GRID_SIZE)
            cv2.circle(input_image,(int(cellLoc[0]*input_image.shape[1]), int(cellLoc[1]*input_image.shape[0]) ),
                5, 
                (0,200,50),5, cv2.LINE_AA)

            showImg( cv2.resize(input_image, (INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[0]), interpolation = cv2.INTER_AREA), 0, 'input')

            ###########################################33
            """

            # image preprocessing
            input_image = cv2.resize(input_image, (INPUT_IMAGE_SHAPE[1], INPUT_IMAGE_SHAPE[0]), interpolation = cv2.INTER_AREA)
            head_image = cv2.resize(head_image, (HEAD_IMAGE_SHAPE[1], HEAD_IMAGE_SHAPE[0]))
            
            input_image = preprocess_VGG_Places(input_image.astype('float32'))
            head_image = preprocess_VGG_Imagenet(head_image.astype('float32'))

            # print(np.min(input_image), np.max(input_image))
            # print(np.min(head_image), np.max(head_image))


            batch_input_images.append(input_image)
            batch_head_images.append(head_image)
            batch_head_locations.append(head_location)
            batch_gaze_locations.append(gaze_location)
            batch_gaze_heatmaps.append(gaze_heatmap)
            batch_int_gaze_heatmaps.append(int_gaze_heatmap)

            """
            cv2.line(input_image,(int(eye_coordinates[0]*input_image.shape[1]), int(eye_coordinates[1]*input_image.shape[0]) ), 
                (int(gaze_coordinates[0]*input_image.shape[1]) ,  int(gaze_coordinates[1]*input_image.shape[0]) ), 
                (0,50,200),4, cv2.LINE_AA)

            cellLoc = getGridLocation(gaze_location, OUTPUT_GRID_SIZE)
            cv2.circle(input_image,(int(cellLoc[0]*input_image.shape[1]), int(cellLoc[1]*input_image.shape[0]) ),
                5, 
                (0,200,50),5, cv2.LINE_AA)

            showImg(input_image[:,:,2],0,'image')
            showImg(head_image,0,'head')
            """

        ind+=BATCH_SIZE

        if (ind >= num_samples):
            ind = ind%num_samples;
            if (doShuffle == True):
                print('Shuffling data')
                shuffle(data)

        batch_input_images, batch_head_images, batch_head_locations, batch_gaze_locations, batch_gaze_heatmaps = shuffleTogether(batch_input_images, batch_head_images, batch_head_locations, batch_gaze_locations, batch_gaze_heatmaps)

        batch_gaze_locations = to_categorical(batch_gaze_locations, OUTPUT_GRID_SIZE*OUTPUT_GRID_SIZE)


        # convert image pixels to float [0.0 to 1.0]
        batch_head_images = np.array(batch_head_images)/255.0
        batch_input_images = np.array(batch_input_images)/255.0

        # print( 'Shapes:\n', batch_input_images.shape, batch_head_images.shape, batch_head_locations.shape, batch_gaze_locations.shape, batch_gaze_heatmaps.shape)

        # print(getGridLocation(np.argmax(batch_gaze_locations), OUTPUT_GRID_SIZE), getGridLocation(np.argmax(batch_gaze_heatmaps), INTERMEDIATE_GRID_SIZE) )

        # yield [np.array(batch_input_images), np.array(batch_head_images), np.array(batch_head_locations)] , [np.array(batch_gaze_locations), np.array(batch_gaze_heatmaps), np.array(batch_int_gaze_heatmaps)]
        yield [np.array(batch_input_images), np.array(batch_int_gaze_heatmaps)] , [np.array(batch_gaze_locations)]


def train(data_file, val_file, test_file, mode = 'train', batch_size=4, nb_epoch=10, weights = '', myLearningRate = 1e-3 ):


    # tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

    early_stopper = EarlyStopping(monitor='loss', patience=7, verbose=1, min_delta=0.01)

    timestamp = time.time()
    currTime = time.strftime("%H-%M-%S")
    
    rm = compact_model.GazeFollowModel( myWeights = weights, learning_rate = myLearningRate )


    best_weights_file = os.path.join('data', 'checkpoints', 'best_'+currTime+'--{epoch:02d}.hdf5')
    # best_weights_file = os.path.join('data', 'checkpoints', 'best_'+currTime+'.hdf5')

    checkpointer = ModelCheckpoint(
        best_weights_file,
        monitor='val_loss',
        save_weights_only=False,
        verbose=1, save_best_only=False)

    num_train_samples = len(get_csv_data(data_file))
    num_train_batches = int(num_train_samples/batch_size)

    num_val_samples = len(get_csv_data(val_file))
    # num_val_batches = int(num_val_samples/batch_size)

    split_ratio = 0.7

    if (mode == 'train'):
        hist = rm.fit_generator( 
            train_generator(data_file, split_ratio, 'b', doShuffle = True, batchSize = BATCH_SIZE),
            validation_data=train_generator(data_file, split_ratio, 'a', doShuffle = True, batchSize = 1),
            verbose=1,
            shuffle = True,
            callbacks=[ checkpointer],
            epochs=nb_epoch*10,
            steps_per_epoch = num_train_batches/10, 
            validation_steps = num_val_samples/50
            )



        print ('Training history:');
        PV(sorted(hist.history.items()))

    # PREDICT TEST OUTPUT
    num_test_samples = len(get_csv_data(test_file))
    # num_test_batches = int(num_test_samples/batch_size)


    print('Number of test samples :', num_test_samples)


    # loss, accuracy = rm.evaluate_generator( test_generator(test_file), steps = num_test_samples)

    # print ('Test Loss:',loss, '\n','Test Accuracy:',accuracy,'\n')

    predictions = rm.predict_generator( train_generator(test_file), steps = num_test_samples, verbose = 1)
    y_pred = np.argmax(predictions, axis=1)

    y_test = [] 
    testGen = train_generator(test_file)
    # for idx in range(len(y_pred)):
    for idx in range(num_test_samples):
        X_, Y_ = next(testGen)
        y_test.append(Y_[0])
    y_test=np.array(y_test)

    y_test = np.argmax(y_test,axis=1)

    cm = confusion_matrix(y_test,y_pred)
    print('\nConfusion Matrix:\n', cm)

    print(classification_report(y_test, y_pred))






BATCH_SIZE = 8
nb_epoch = 30
learning_rate = 1e-3

dropout_val_ = 0.4


def main():

        if (len(sys.argv)<5):
            print('Usage: python train_gazeFollow.py <train/test> <train_list.txt> <validate_list.txt> <test_list.txt> <weights.hdf5>')

        print ('Batch Size = ',BATCH_SIZE, '\n', 
            'No. epochs = ',nb_epoch, '\n',
            'Learning Rate = ',learning_rate, '\n',)

        mode = sys.argv[1]

        data_csv_file = sys.argv[2];
        val_file = sys.argv[3];
        test_file = sys.argv[4];

 
        weights = ""
        if (len(sys.argv)>5):
            weights = sys.argv[5];
            print ('Using pretrained weights : \n', weights)


        train(data_csv_file, val_file, test_file, mode = mode,  batch_size=BATCH_SIZE, nb_epoch=nb_epoch, weights = weights , myLearningRate = learning_rate)

if __name__ == '__main__':
    main()