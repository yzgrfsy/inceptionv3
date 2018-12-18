import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from keras.models import Model

import config
import util
import matplotlib.pyplot as plt


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=32, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def decode_predictions_custom(preds, top=3):

    CLASS_CUSTOM = ["3147", "3148", "3182", "3184", "3191", "3193", "3201", "3202", "3203", "3205", "3206", "3207",
                    "3208", "3209", "3211", "3212", "3214", "3215", "3216", "3217", "3218", "3220", "3226", "3227",
                    "3228", "3229", "3230", "3238", "3240", "3249", "3253", "3281", "3283", "3284", "3285", "3286",
                    "3287", "3288", "3290", "3291", "3292", "3293", "3298", "3299", "3300"]

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(CLASS_CUSTOM[i] + ":" + str("%.2f" % (pred[i] * 100))) for i in top_indices]
        results.append(result)
    return results


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)
    return y_true, inputs


def predict(path):
    files = get_files(path)
    n_files = len(files)
    #print('Found {} files'.format(n_files))

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)
        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        # np.save("000", inputs)
        y_trues += y_true
        if not args.store_activations:
            global out, out2

            out = model.predict(np.array(inputs))
            print("out:", out)
            print("111111111111111111111111")
            # np.save("111", out)
            out2 = decode_predictions_custom(out, top=3)
            print("out2:", out2)
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            # print("np.argmaxout:", )
        if not args.store_activations:
            for i, p in enumerate(predictions):
                recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
                print(' picture:{}   top3:{}'.format(y_trues[i], out2[i]))
                # print(' picture:{}   in {} file files be predicted as {} class  , score : {}'.format(y_trues[i], files[i].split(os.sep)[-2],
                #                                                                                     recognized_class, ("%.2f" % (100 * np.max(
                #                                                                                                 out[i])))))


if __name__ == '__main__':
    args = parse_args()
    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()
    classes_in_keras_format = util.get_classes_in_keras_format()
    predict(args.path)