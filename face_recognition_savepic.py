# -*- coding:utf-8 -*-
import os
import json
import scipy.misc
import tensorflow as tf
import src.facenet
import src.align.detect_face
import numpy as np
from scipy import misc
import face_mysql
import cv2
import imageio
# cancel the warnings when code is running
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class face_reconition:
    def __init__(self):
        pass

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def get_image_paths(self, inpath):
        paths = []
        for file in os.listdir(inpath):
            if os.path.isfile(os.path.join(inpath, file)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) is False:
                    continue

                paths.append(os.path.join(inpath, file))

        return (paths)

    def images_to_vectors(self, inpath, outjson_path, modelpath):
        results = dict()

        with tf.Graph().as_default():
            with tf.Session() as sess:
                src.facenet.load_model(modelpath)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                image_paths = self.get_image_paths(inpath)

                print('Image paths:', image_paths)
                for image_path in image_paths:
                    # get the face numbers of a pic
                    img = scipy.misc.imread(os.path.expanduser(image_path), mode='RGB')
                    images = self.image_array_align_data(img,image_path)

                    print('Shape of images:', images.shape)

                    # Determine if the face is detected and if it is not detected,
                    # jump out of this loop

                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    if images.shape[0] < 1 :
                        print('No face detected.')
                    continue

                try:
                    print('Running the session ... ')
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    print('Session finished running.')
                    print('Shape of emb_array:', emb_array)
                    filename_base, file_extension = os.path.splitext(image_path)
                    for j in range(0, len(emb_array)):
                        results[filename_base + "_" + str(j)] = emb_array[j].tolist()
                        face_mysql_instant = face_mysql.face_mysql()
                        face_mysql_instant.insert_facejson\
                        (filename_base + "_" + str(j),
                                    ",".join(str(li)
                                    for li in emb_array[j].tolist()))
                except Exception as e:
                      print('Error occurred:', e)

        # Print out the results before writing to the JSON file
        print(results)
        # All done, save for later!
        json.dump(results, open(outjson_path, "w"))


    def image_array_align_data(self, image_arr,image_path, image_size=160, margin=32, gpu_memory_fraction=1.0,
                               detect_multiple_faces=True):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)
        img = image_arr
        bounding_boxes, _ = src.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        nrof_successfully_aligned = 0

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    det_arr.append(det[index, :])
            else:
                det_arr.append(np.squeeze(det))

            images = np.zeros((nrof_faces, image_size, image_size, 3))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = scipy.misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                nrof_successfully_aligned += 1
                # print(scaled)
                # scaled=self.prewhiten(scaled)
                # save avatar
                filename_base = './img/'
                filename = os.path.basename(image_path)
                filename_name, file_extension = os.path.splitext(filename)
                output_filename_n = "{}/{}_{}{}".format(filename_base, filename_name, i, file_extension)
                misc.imsave(output_filename_n, scaled)
                scaled = src.facenet.prewhiten(scaled)
                scaled = src.facenet.crop(scaled, False, 160)
                scaled = src.facenet.flip(scaled, False)
                images[i] = scaled
        if nrof_faces > 0:
            return images
        else:
            return np.zeros((1,3))


'''
    def capture_image(self, image_path):
        # open the camera
        cap = cv2.VideoCapture(0)
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError('Can not open Webcam')
        ret, frame = cap.read()
        cv2.imwrite(image_path, frame)
        cap.release()
'''


if __name__ == "__main__":
    face_reconition = face_reconition()
    images_path = '/Users/hubertlee/Desktop/Demos/face_login_project/img'
    #captured_image_path = images_path + '/captured_image.jpg'
    #face_reconition.capture_image(captured_image_path)
    # model path
    modelpath = "/Users/hubertlee/Desktop/Demos/face_login_project/models/facenet/20230810-000613"
    out_path = '/Users/hubertlee/Desktop/Demos/face_login_project/img/pic.json'
    face_reconition.images_to_vectors(images_path, out_path, modelpath)
