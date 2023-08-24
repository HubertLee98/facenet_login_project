import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def load_pb_model(pb_file):
    with tf.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph


from sklearn.model_selection import KFold


def evaluate_on_lfw(embeddings1, embeddings2, actual_issame, distance_metric=0, nrof_folds=10):
    # 计算距离
    if distance_metric == 0:
        dist = np.sqrt(np.sum(np.square(embeddings1 - embeddings2), 1))
    else:
        dist = np.sum(embeddings1 * embeddings2, 1)

    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, dist, actual_issame, nrof_folds=nrof_folds)

    return tpr, fpr, accuracy


def calculate_roc(thresholds, dist, actual_issame, nrof_folds=10):
    nrof_pairs = min(len(actual_issame), dist.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(dist)):
        # 计算每个阈值的TPR和FPR
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx] = calculate_accuracy(threshold, dist[test_set],
                                                                                              actual_issame[test_set])
        accuracy[fold_idx] = np.mean(np.equal(dist[test_set] <= np.mean(thresholds), actual_issame[test_set]))

    mean_tpr = np.mean(tprs, 0)
    mean_fpr = np.mean(fprs, 0)
    mean_acc = np.mean(accuracy)
    return mean_tpr, mean_fpr, mean_acc


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    return tpr, fpr


def plot_roc(fpr, tpr, figure_name="roc.png"):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.savefig(figure_name)


if __name__ == "__main__":
    pb_model_path = '/Users/hubertlee/Desktop/facenet-master/models/facenet/20230819-101932/softmax.pb'
    lfw_pairs_path = './data/pairs.txt'

    graph = load_pb_model(pb_model_path)

    with graph.as_default():
        with tf.Session() as sess:
            # 获取输入输出张量
            images_placeholder = graph.get_tensor_by_name("input:0")
            embeddings = graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

            # 模拟LFW数据获取
            # 假设pairs是你的LFW验证数据集，形如[img_path1, img_path2, is_same]，其中is_same是0或1
            pairs = []
            with open(lfw_pairs_path, 'r') as f:
                pairs = f.readlines()

            emb_array1 = []
            emb_array2 = []
            actual_issame = []

            for pair in pairs:
                img_path1, img_path2, is_same = pair.strip().split()  # 根据你的数据格式进行适当调整
                img1 = preprocess_image(img_path1)  # 你需要定义预处理函数来处理LFW图像
                img2 = preprocess_image(img_path2)

                emb1 = sess.run(embeddings, feed_dict={images_placeholder: img1, phase_train_placeholder: False})
                emb2 = sess.run(embeddings, feed_dict={images_placeholder: img2, phase_train_placeholder: False})

                emb_array1.append(emb1)
                emb_array2.append(emb2)
                actual_issame.append(is_same)

            tpr, fpr, _ = evaluate_on_lfw(np.array(emb_array1), np.array(emb_array2), actual_issame)
            plot_roc(fpr, tpr)
