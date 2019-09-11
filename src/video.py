from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import math
import cv2
import numpy as np
import align.detect_face
import facenet
from facenet import crop
import tensorflow as tf
import pickle
from PIL import Image,ImageDraw,ImageFont
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                           datefmt = '%m/%d/%Y %H:%M:%S',
                           level = logging.INFO)
logger = logging.getLogger(__name__)


# 加载opencv人脸识别模型
face_cascade = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

def opencvDetect(img):
    faces = face_cascade.detectMultiScale(img,1.3,5)#faces返回人脸位置坐标
    if len(faces) > 0 :
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)#画矩形框标记人脸,框大点
    return faces

# 加载mtcnn
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
logger.info('加载MTCNN模型')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
logger.info('加载完成')

def mtcnnDetect(img):
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes

def main(args):
    # 加载svm
    classifier_filename_exp = os.path.expanduser(args.classifier_filename)
    logger.info('加载分类器"%s"' % classifier_filename_exp)
    with open(classifier_filename_exp, 'rb') as infile:
        (classifier_model, class_names) = pickle.load(infile)
        logger.info('分类器识别类别有:{}'.format(class_names))
    logger.info('加载完成')


    with tf.Graph().as_default():
        with tf.Session() as sess:
            logger.info('加载Facenet进行人脸识别')
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            logger.info('加载完成')
            cap = cv2.VideoCapture(0)
            num = 1
            while(1):
                key = cv2.waitKey(1)
                # get a frame
                ret, frame = cap.read()
                if(num%args.interval == 0):
                    logger.info('识别第%d帧' %num)
                    if frame.ndim == 3:
                        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame
                    if gray.ndim == 2:
                        img = facenet.to_rgb(gray)

                    bounding_boxes = mtcnnDetect(img)
                    nrof_faces = bounding_boxes.shape[0]  # number of faces
                    crop_faces=np.zeros((len(bounding_boxes), args.image_size, args.image_size, 3))
                    label=[]
                    if nrof_faces < 1:
                        logger.info('未检测到人脸')
                        pass
                    else:
                        logger.info('检测到%d个人脸'%nrof_faces)
                        for i,face_position in enumerate(bounding_boxes):
                            face_position = face_position.astype(int)
                            cv2.rectangle(frame, (face_position[0],
                                                  face_position[1]),
                                          (face_position[2], face_position[3]),
                                          (0, 255, 0), 2)
                            #将图像reshape成输入进facenet模型的结构
                            face = frame[face_position[1]:face_position[3],face_position[0]:face_position[2], ]
                            face = cv2.resize(face, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)#crop shape 96*96*3
                            data = face.reshape(-1, args.image_size, args.image_size, 3) #reshape image
                            feed_dict = {images_placeholder: data, phase_train_placeholder: False}
                            #得到图像embedding
                            emb_array = sess.run(embeddings,feed_dict=feed_dict)
                            #embedding输入到svm分类
                            predictions = classifier_model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            for i in range(len(best_class_indices)):
                                label.append(class_names[best_class_indices[i]])
                            results = set(zip(label,best_class_probabilities))
                            logger.info('识别出:{}'.format(results))

                            cv2.putText(frame, 'detected:{}'.format(results), (50, 100),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0),
                                        thickness=2, lineType=2)

                cv2.imshow("capture", frame)
                num += 1 #帧数+1
                # 键盘输入q则退出,ord返回ascii码
                if key == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--interval',type=int,help='Frame by second',default=1)

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
