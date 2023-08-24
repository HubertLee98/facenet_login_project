# -*- coding:utf-8 -*-
from flask import Flask, jsonify, abort, make_response, request, url_for,render_template
from flask_httpauth import HTTPBasicAuth
import json
import os
import ntpath
import argparse
import face_mysql
import tensorflow as tf
import src.facenet
import src.align.detect_face
import numpy as np
from scipy import misc
import matrix_fun
import urllib
from src.align import detect_face

app = Flask(__name__)
# The maxmium size of pic is 16M
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
auth = HTTPBasicAuth()

# based on database of lfw
MAX_DISTINCT=1.22

# find the device
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# from werkzeug import secure_filename
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'pic_tmp/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    pass
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Find the device (CPU or GPU)

with tf.Graph().as_default():
    with tf.device('/cpu:0'):
        sess = tf.Session()
        with sess.as_default():
            #pnet, rnet, onet = src.align.detect_face(sess, None) # Treated as a moudle not a func
            pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)

        modelpath = "/Users/hubertlee/Desktop/Demos/face_login_project/models/facenet/20230810-000613"

        pwd = os.getcwd()

        father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
        with sess.as_default():

    # src.facenet.load_model(modelpath)
    # Loading model....
            meta_file, ckpt_file = src.facenet.get_model_filenames(modelpath)
            saver = tf.train.import_meta_graph(os.path.join(modelpath, meta_file))
            saver.restore(sess, os.path.join(modelpath, ckpt_file))
    # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # face recognition and loading
            print('Creating networks and loading parameters')
    # get the pic from post and return the id of database


    @app.route('/')
    def index():
        return render_template("face_login.html")
    @app.route('/face_insert_html')
    def face_insert_html():
        return render_template("face_insert.html")
    @app.route('/face_query_html')
    def face_query_html():
        return render_template("face_login.html")
    @app.route('/face/insert', methods=['POST'])
    def face_insert():
        uid = request.form['uid']
        ugroup = request.form['ugroup']
        upload_files = request.files['imagefile']
        file = upload_files
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(image_path)

        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        # 'Flase' detects only one face in the graph, while 'True' detects multiple faces

        '''
        Generally, only one face is detected when entering the library,
        and multiple faces are detected when querying
        '''

        images = image_array_align_data(img, image_path, pnet, rnet, onet, detect_multiple_faces=False)
        with tf.device('/cpu:0'):
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            emb_array = sess.run(embeddings, feed_dict=feed_dict)

        filename_base, file_extension = os.path.splitext(image_path)
        id_list = []

        # save in sql
        for j in range(0, len(emb_array)):
            face_mysql_instant = face_mysql.face_mysql()
            last_id = face_mysql_instant.insert_facejson(filename_base + "_" + str(j),
                                                         ",".join(str(li) for li in emb_array[j].tolist()), uid, ugroup)
            id_list.append(str(last_id))

        # edit type of return
        request_result = {}
        request_result['id'] = ",".join(id_list)
        if len(id_list) > 0:
            request_result['state'] = 'sucess'
        else:
            request_result['state'] = 'error'

        print(request_result)
        return json.dumps(request_result)


    @app.route('/face/query', methods=['POST'])
    def face_query():
        ugroup = request.form['ugroup']
        upload_files = request.files['imagefile']
        file = upload_files
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(image_path)

        # loading
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        images = image_array_align_data(img, image_path, pnet, rnet, onet)

        # Determine if no face is detected in the figure and return directly
        if len(images.shape) < 4: return json.dumps({'error': "not found face"})
        with  tf.device('/cpu:0'):
         feed_dict = {images_placeholder: images, phase_train_placeholder: False}
         emb_array = sess.run(embeddings, feed_dict=feed_dict)
         face_query = matrix_fun.matrix()

        # pic_min_scores - Face distance in the database
        # pic_min_names - name of the file that was saved when it was stored
        # pic_min_uid - The corresponding user id

         pic_min_scores, pic_min_names, pic_min_uid = face_query.get_socres(emb_array, ugroup)

        # if there s no group, return
        if len(pic_min_scores) == 0: return json.dumps({'error': "not found user group"})
        result = []
        for i in range(0, len(pic_min_scores)):
            if pic_min_scores[i]<MAX_DISTINCT:
                rdict = {'uid': pic_min_uid[i],
                         'distance': pic_min_scores[i],
                         'pic_name': pic_min_names[i],
                         'state':'Recognized successfully!'}
                result.append(rdict)
        print(result)
        if len(result) == 0 :
            return json.dumps({"state":"success, but not match face"})
        else:
            return json.dumps(result)


def image_array_align_data(image_arr, image_path, pnet, rnet, onet, image_size=160, margin=32, gpu_memory_fraction=1.0,
                           detect_multiple_faces=True):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    img = image_arr
    with tf.device('/cpu:0'):
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

        images = np.zeros((len(det_arr), image_size, image_size, 3))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            #  cv2.resize(img,(w,h))
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            nrof_successfully_aligned += 1
            # save the pic
            filename_base = 'pic_tmp'
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
        return np.zeros((1, 3))



def get_url_imgae(picurl):
    response = urllib.urlopen(picurl)
    pic = response.read()
    pic_name = "pic_tmp/" + os.path.basename(picurl)
    with open(pic_name, 'wb') as f:
        f.write(pic)
    return pic_name
@auth.get_password
def get_password(username):
    if username == 'root':
        return 'root'
    return None
@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Invalid data!'}), 400)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
