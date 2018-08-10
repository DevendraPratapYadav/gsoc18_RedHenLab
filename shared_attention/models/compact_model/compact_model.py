import sys


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras import optimizers
from keras.models import Sequential
# from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Flatten, Dense, Dropout, Reshape
from keras.layers import multiply, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import cv2, numpy as np

from keras.applications import VGG16
import vgg16_hybrid_places_1365


VGG16_HYBRID_PLACES_1365_WEIGHTS = 'vgg16-hybrid1365_weights_tf_dim_ordering_tf_kernels_notop.h5'


OUTPUT_GRID_SIZE = 5
INTERMEDIATE_GRID_SIZE = 14

INPUT_IMAGE_SHAPE = (224, 224, 3)
HEAD_IMAGE_SHAPE = (224, 224, 3)
HEAD_LOCATION_SHAPE = (INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE,1)
GAZE_HEATMAP_SHAPE = (INTERMEDIATE_GRID_SIZE, INTERMEDIATE_GRID_SIZE)

dropout_rate = 0.3
LeakyReLU_alpha = 0.3

def GazeFollowModel( myWeights = '' , learning_rate = 1e-3):
    """
    gaze_vgg = VGG16(weights = 'imagenet',include_top=False, input_shape=HEAD_IMAGE_SHAPE)
    for layer_ in gaze_vgg.layers[:-4]:
        layer_.trainable = False
    """
    # for layer_ in gaze_vgg.layers:
    #     print(layer_, layer_.trainable)



    # saliency_vgg = vgg16_hybrid_places_1365.VGG16_Hybrid_1365(include_top=False, input_shape = INPUT_IMAGE_SHAPE, weights=VGG16_HYBRID_PLACES_1365_WEIGHTS)
    saliency_vgg = VGG16(weights = 'imagenet',include_top=False, input_shape=HEAD_IMAGE_SHAPE)
    saliency_vgg.layers.pop()
    saliency_vgg = Model(saliency_vgg.input, saliency_vgg.layers[-1].output, name=saliency_vgg.name)
    for layer_ in saliency_vgg.layers[:-3]:
        layer_.trainable = False

    # for layer_ in saliency_vgg.layers:
    #     print(layer_, layer_.trainable)


    input_image = Input(shape=INPUT_IMAGE_SHAPE , name = 'input_image')

    # true_gaze_heatmap = Input(shape=GAZE_HEATMAP_SHAPE , name = 'true_gaze_heatmap')

    # build saliency pathway using output of VGG-Places model
    saliency_vgg_output = saliency_vgg(input_image)

    """
    saliency_conv1 = Conv2D(1, (1, 1), use_bias=False, name = 'saliency_conv1')(saliency_vgg_output)
    saliency_heatmap = BatchNormalization()(saliency_conv1)
    saliency_heatmap = Activation('sigmoid', name = 'saliency_heatmap')(saliency_heatmap)
    """
    saliency_conv1 = Conv2D(1, (1, 1), name = 'saliency_conv1')(saliency_vgg_output)
    saliency_heatmap = Activation('sigmoid', name = 'saliency_heatmap')(saliency_conv1)

    # saliency_heatmap = Flatten(name = 'saliency_map_flattened')(saliency_conv1)


    # build gaze pathway using output of VGG-16 model
    """
    gaze_vgg_output = gaze_vgg(head_image)
    gaze_vgg_output = Conv2D(128, (1, 1), name = 'gaze_conv1')(gaze_vgg_output)
    gaze_flatten1 = Flatten(name = 'gaze_flatten1')(gaze_vgg_output)

    gaze_dense1 = Dense(400, use_bias=False, name = 'gaze_dense1')(gaze_flatten1)
    gaze_dense1 = LeakyReLU(LeakyReLU_alpha)(gaze_dense1)
    gaze_dense1 = BatchNormalization()(gaze_dense1)
    gaze_dense1 = Dropout(dropout_rate)(gaze_dense1)


    head_loc = Flatten( name = 'head_location_flattened')(head_location) # head location input
    gaze_net_merged = concatenate([gaze_dense1, head_loc], name = 'merged_gaze_features')

    
    gaze_dense2 = Dense(300, use_bias=False, name = 'gaze_dense2')(gaze_net_merged)
    gaze_dense2 = LeakyReLU(LeakyReLU_alpha)(gaze_dense2)
    gaze_dense2 = BatchNormalization()(gaze_dense2)
    gaze_dense2 = Dropout(dropout_rate)(gaze_dense2)
    """

    """
    gaze_dense3 = Dense(250, use_bias=False, name = 'gaze_dense3')(gaze_dense2)
    gaze_dense3 = LeakyReLU(LeakyReLU_alpha)(gaze_dense3)
    gaze_dense3 = BatchNormalization()(gaze_dense3)
    gaze_dense3 = Dropout(dropout_rate)(gaze_dense3)
    """

    # gaze_heatmap = Dense(INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE, use_bias=False, name = 'gaze_dense4')(gaze_net_merged)#(gaze_dense3)
    # gaze_heatmap = BatchNormalization()(gaze_heatmap)
    
    """
    int_gaze_heatmap = Dense(INTERMEDIATE_GRID_SIZE*INTERMEDIATE_GRID_SIZE, name = 'gaze_dense4')(gaze_dense2)#(gaze_dense3)
    int_gaze_heatmap = Activation('sigmoid', name = 'intermediate_gaze_heatmap')(int_gaze_heatmap) # use sigmoid for gaze importance map
    gaze_heatmap = Dropout(dropout_rate)(int_gaze_heatmap)
    
    gaze_heatmap = Reshape((INTERMEDIATE_GRID_SIZE,INTERMEDIATE_GRID_SIZE,1), name = 'gaze_heatmap')(gaze_heatmap)
    """
    gaze_heatmap = Input(shape=(INTERMEDIATE_GRID_SIZE,INTERMEDIATE_GRID_SIZE) , name = 'gaze_heatmap')
    gaze_heatmap_reshaped = Reshape((INTERMEDIATE_GRID_SIZE,INTERMEDIATE_GRID_SIZE,1))(gaze_heatmap)

    # multiply gaze and saliency heatmaps
    multiplied_heatmaps = multiply( [gaze_heatmap_reshaped,saliency_heatmap] , name = 'combined_heatmaps')

    heatmap_conv1 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same',
                     # kernel_regularizer=l2(0.0002),
                     activation='relu', name='heatmap_conv1')(multiplied_heatmaps)
    # heatmap_conv1 = Dropout(dropout_rate)(heatmap_conv1)

    heatmap_conv2 = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same',
                     # kernel_regularizer=l2(0.0002),
                     activation='relu', name='heatmap_conv2')(heatmap_conv1)
    # heatmap_conv2 = Dropout(dropout_rate)(heatmap_conv2)

    heatmap_conv3 = Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding='same',
                     # kernel_regularizer=l2(0.0002),
                     activation='relu', name='heatmap_conv3')(heatmap_conv2)
    # heatmap_conv3 = Dropout(dropout_rate)(heatmap_conv3)

    heatmap_conv4 = Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same',
                     # kernel_regularizer=l2(0.0002),
                     activation='sigmoid', name='heatmap_conv4')(heatmap_conv3)
    # heatmap_conv4 = Dropout(dropout_rate)(heatmap_conv4)

    # multiplied_heatmaps = Reshape((INTERMEDIATE_GRID_SIZE,INTERMEDIATE_GRID_SIZE))(multiplied_heatmaps)
    final_heatmap_flattened = Flatten(name = 'final_heatmap_output')(heatmap_conv4)

    # add a final dense layer for output
    final_output = Dense(OUTPUT_GRID_SIZE*OUTPUT_GRID_SIZE, activation='softmax', name = 'location_output')(final_heatmap_flattened)

    gazeFollow = Model(inputs=[input_image, gaze_heatmap], outputs=[final_output], name = 'GazeFollow')

    losses = {
    # "final_heatmap_output": "mse",
    # "intermediate_gaze_heatmap": "mse",
    "location_output": "categorical_crossentropy"
    }

    # lossWeights = {"final_heatmap_output": 100.0, "location_output": 0.5, "intermediate_gaze_heatmap": 30.0}
    lossWeights = {"location_output": 1.0}

    print('Loss weights :\n', lossWeights)


    gazeFollow.compile(loss=losses, loss_weights=lossWeights,
                  # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.0),
                  metrics=['accuracy'])


    if(myWeights != ''):
        gazeFollow.load_weights(myWeights)


    print(gazeFollow.name)
    print(gazeFollow.summary())
    print(gazeFollow.output)

    plot_model(gazeFollow, to_file='model.png' , show_shapes=True)
    """
    plot_model(gazeFollow, to_file='model.png' , show_shapes=False)
    modelVis = cv2.imread('model.png')
    cv2.imshow('Model visualization',modelVis)
    cv2.waitKey(0)
    """
    
    return gazeFollow




