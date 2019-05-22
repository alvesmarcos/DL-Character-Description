##
## @spmallick
## thanks https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison
##
import cv2
import sys
import time
import tensorflow as tf
import numpy as np
from PIL import Image

WIDTH = HEIGHT = 100

def label_gender(pred):
    return 'Male' if pred <= 0.5 else 'Female'

def label_ethnicity(pred):
    labels = ['White', 'Black', 'Asian', 'Indian', 'Hispanic or Latino or Middle Eastern']
    return labels[pred[0].argmax()]

def label_age(pred):
    labels = ['(0, 2)', '(4, 6)', '(8, 13)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    return labels[pred[0].argmax()]

def image_to_arr(frame, x1, x2, y1, y2):
    frameDescription = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frameDescription)
    image = image.crop((x1,y1,x2,y2))
    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    X = np.array(image, dtype=np.float32)
    X /= 255
    inp = np.array([X])
    return inp

def detectFaceOpenCVDnn(net, frame, genderModel, ethnicityModel, ageModel):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])

            image = image_to_arr(frame, x1, x2, y1, y2)
            # predictions
            pred_gender = genderModel.predict(image)
            pred_age = ageModel.predict(image)
            pred_eth = ethnicityModel.predict(image)

            cv2.putText(frameOpencvDnn, '{:}, {:}, {:}'.format(label_gender(pred_gender), label_age(pred_age), label_ethnicity(pred_eth)), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

if __name__ == "__main__" :
    genderModel = tf.keras.models.load_model('models/best_model_gender.hdf5')
    ethnicityModel = tf.keras.models.load_model('models/best_model_ethnicity.hdf5')
    ageModel = tf.keras.models.load_model('models/best_model_age.hdf5')
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    cap = cv2.VideoCapture(source)
    hasFrame, frame = cap.read()

    #vid_writer = cv2.VideoWriter('output-dnn-{}.avi'.format(str(source).split(".")[0]),cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

    frame_count = 0
    tt_opencvDnn = 0
    while(True):
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame, genderModel, ethnicityModel, ageModel)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Character Description", outOpencvDnn)

        #vid_writer.write(outOpencvDnn)
        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    #vid_writer.release()
