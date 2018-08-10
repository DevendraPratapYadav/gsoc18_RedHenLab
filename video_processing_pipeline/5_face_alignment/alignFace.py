#*************************************************
# Program to perform face alignment in video using custom face landmark detection like mtcnn or dlib
#*************************************************


import cv2
import numpy as np
import sys
import pdb

from mtcnn.mtcnn import MTCNN
import dlib


def showImg(img, waitTime):
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)

def removeTrailingBackslash(path):
    if (path[-1] == '/'):
        path = path[:-1];

    return path;


# template for aligned face. 68 points
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

#: Landmark indices.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

pastH = []


def align(imgDim, rgbImg,
          detectedLandmarks, templateLandmarks):
    assert imgDim is not None
    assert rgbImg is not None
    assert templateLandmarks is not None

    detectedLandmarks = np.float32(np.array(detectedLandmarks))
    templateLandmarks = np.float32(np.int32(np.array(templateLandmarks)))

    # if landmarks are not detected, use previous frames' warp matrix
    if (len(detectedLandmarks) > 0):
        H = cv2.estimateRigidTransform(detectedLandmarks, templateLandmarks, True)
    else:
        H = None

    SMOOTH_NUM = 3

    if (len(pastH) < SMOOTH_NUM):
        if H is None:
            return []

        for ii in range(SMOOTH_NUM):
            pastH.append(H);

    smoothH = np.zeros((2, 3))

    if H is not None:
        pastH.pop()
        pastH.insert(0, H)

    # Take average of last 'SMOOTH_NUM' warp matrices to reduce jitter
    for ii in range(SMOOTH_NUM):
        smoothH = smoothH + pastH[ii]

    smoothH = smoothH / SMOOTH_NUM

    transformed = cv2.warpAffine(rgbImg, smoothH, (imgDim, imgDim))

    return transformed


def alignFace_dlib(image):
    # Input image must be square

    imgDim = image.shape[0]

    template_landmarks = get68TemplateLandmarks(imgDim);

    detected_landmarks = getDlibLandmarks(image);

    if (len(detected_landmarks) < 1):
        # print('No face detected')
        return align(imgDim, image, [], template_landmarks);

    RIGID_POINTS = [39, 42, 8, 36, 45,
                    33]  # leftEye_leftmost, leftEye_rightmost, chin_bottom, rightEye_leftmost, tightEye_rightmost, nose_tip

    template_landmarks = np.array(template_landmarks)[RIGID_POINTS]
    detected_landmarks = np.array(detected_landmarks)[RIGID_POINTS]

    """
    # visualize landmarks
    for lm in detected_landmarks:
        cv2.circle(image,tuple(lm), 2, (0,255,100), 2)
    """

    alignedImg = align(imgDim, image, detected_landmarks, template_landmarks);

    return alignedImg


def get68TemplateLandmarks(imgDim):
    template_landmarks = []

    for ind in range(68):
        lm = imgDim * MINMAX_TEMPLATE[ind]

        template_landmarks.append(lm)

    return template_landmarks


def getDlibLandmarks(image):
    dets = dlib_detector(image, 1)

    if (len(dets) < 1):
        return []

    landmarks = []

    for k, d in enumerate(dets):

        shape = dlib_predictor(image, d)

        for ind in range(68):
            landmarks.append([shape.part(ind).x, shape.part(ind).y])

    # print('Dlib landmarks:\n', landmarks)


    return landmarks


def getDlibLandmarksDirectly(image):
    landmarks = []

    myDet = dlib.rectangle(20, 20, 200, 200)

    shape = dlib_predictor(image, myDet)

    for ind in range(68):
        landmarks.append([shape.part(ind).x, shape.part(ind).y])

    # print('Dlib landmarks:\n', landmarks)

    return landmarks


def alignFace_MTCNN(image):
    # Input image must be square

    imgDim = image.shape[0]

    template_landmarks = get4TemplateLandmarks(imgDim);
    detected_landmarks = getMTCNNLandmarks(image);

    if (len(detected_landmarks) < 1):
        # print('No face detected')
        return align(imgDim, image, [], template_landmarks);

    """
    # visualize landmarks
    for lm in detected_landmarks:
        cv2.circle(image,tuple(lm), 2, (0,255,100), 2)
    """

    alignedImg = align(imgDim, image, detected_landmarks, template_landmarks);

    return alignedImg


def getMTCNNLandmarks(image):
    # image = cv2.imread(sys.argv[1])
    result = MTCNN_detector.detect_faces(image)

    if (len(result) < 1):
        return []

    # take first face's landmarks
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    MTCNN_LIP_CENTER = [(keypoints['mouth_left'][0] + keypoints['mouth_right'][0]) / 2,
                        (keypoints['mouth_left'][1] + keypoints['mouth_right'][1]) / 2]

    MTCNN_landmarks = [list(keypoints['left_eye']), list(keypoints['right_eye']), list(keypoints['nose']),
                       MTCNN_LIP_CENTER]

    MTCNN_landmarks = [list(map(int, elem)) for elem in MTCNN_landmarks]

    return MTCNN_landmarks


def get4TemplateLandmarks(imgDim):
    T_LEYE_LEFT = imgDim * MINMAX_TEMPLATE[36]
    T_LEYE_RIGHT = imgDim * MINMAX_TEMPLATE[39]
    T_REYE_LEFT = imgDim * MINMAX_TEMPLATE[42]
    T_REYE_RIGHT = imgDim * MINMAX_TEMPLATE[45]

    T_NOSE = list(imgDim * MINMAX_TEMPLATE[33])
    T_LIP_LEFT = list(imgDim * MINMAX_TEMPLATE[48])
    T_LIP_RIGHT = list(imgDim * MINMAX_TEMPLATE[54])

    T_LIP_TOP = list(imgDim * MINMAX_TEMPLATE[51])
    T_LIP_BOTTOM = list(imgDim * MINMAX_TEMPLATE[57])

    T_LIP_CENTER = [(T_LIP_TOP[0] + T_LIP_BOTTOM[0]) / 2, (T_LIP_TOP[1] + T_LIP_BOTTOM[1]) / 2]

    T_LEYE_CENTER = [(T_LEYE_LEFT[0] + T_LEYE_RIGHT[0]) / 2, (T_LEYE_LEFT[1] + T_LEYE_RIGHT[1]) / 2]
    T_REYE_CENTER = [(T_REYE_LEFT[0] + T_REYE_RIGHT[0]) / 2, (T_REYE_LEFT[1] + T_REYE_RIGHT[1]) / 2]

    template_landmarks = [T_LEYE_CENTER, T_REYE_CENTER, T_NOSE, T_LIP_CENTER]

    return template_landmarks


def processVideo(videoPath, alignmentMethod, outputVideoSize, outputFolder):
    vid = cv2.VideoCapture(videoPath);

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)

    print(videoPath, " : total_frames:", total_frames, ", FPS:", FPS)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # videoName = croppedFacesPath+str(f[0])+'.avi';

    videoName = videoPath.split('/')[-1]

    outputVideoName = outputFolder + '/' + videoName + '_aligned.mp4';
    outVideoSize = (outputVideoSize, outputVideoSize)

    outputVideo = cv2.VideoWriter(outputVideoName, fourcc, FPS, outVideoSize)

    for ind in range(0, total_frames):

        ret, frame = vid.read()

        if (ret == False):
            print('Error reading frame ', ind, ' in ', videoPath);
            continue;

        frame = cv2.resize(frame, outVideoSize)

        # NOTE : Add more face alignment methods here. 
        #        Use the 'align' function with any custom landmark detector 

        alignedFrame = []

        if (alignmentMethod == 'mtcnn'):
            alignedFrame = alignFace_MTCNN(frame)
        elif (alignmentMethod == 'dlib'):
            alignedFrame = alignFace_dlib(frame)

        if (len(alignedFrame) == 0):
            print('Error : Alignment error in Frame Number:', ind + 1)
            continue;

        # showImg(alignedFrame, 33)
        outputVideo.write(alignedFrame)

    cv2.destroyAllWindows()
    outputVideo.release()


if (len(sys.argv)<6):
    print('Usage: python alignFace.py <inputVideoPath> <output_folder> <landmark_detection_method (\'mtcnn\' or \'dlib\') <aligned_video_size> <dlib_shape_predictor_model_path>')
    print('Example : python alignFace.py vid.mp4 output mtcnn 224 5_face_alignment/libs/model.mat');
    sys.exit();

inputVideo = sys.argv[1]

outputFolder = sys.argv[2]

alignmentMethod = sys.argv[3]

outputVideoSize = int(float(sys.argv[4]))

dlib_shape_predictor_path = removeTrailingBackslash(sys.argv[5])

# dlib_shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
# dlib_shape_predictor_path = libsPath + '/' + dlib_shape_predictor_path


dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(dlib_shape_predictor_path)

MTCNN_detector = MTCNN()

processVideo(inputVideo, alignmentMethod, outputVideoSize, outputFolder)
