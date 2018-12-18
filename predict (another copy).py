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
    parser.add_argument('--batch_size', default=100, type=int, help='How many files to predict on at once')
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
        # print("pred:", pred)
        # print(" pred.argsort():", pred.argsort())
        #print(len(pred.argsort()))
        top_indices = pred.argsort()[-top:][::-1]
        #print("top_indices:", top_indices)
        # for i in top_indices:
        #     print("i:", i)

        #     #print("CLASS_CUSTOM1:", CLASS_CUSTOM[i])
        #
        #
        #     print((CLASS_CUSTOM[i]+":"+str(pred[i]*100)))
            #print(tuple(CLASS_CUSTOM[i])+pred[i]*100)
        #result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,)for i in top_indices]
        result = [(CLASS_CUSTOM[i] + ":" + str("%.2f" % (pred[i] * 100))) for i in top_indices]

        results.append(result)
        # print("results:", results)
        #print(results[0])
    return results

def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        #print(i)
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            #print(i.split(os.sep))
            #print("image_class:", image_class)
            # print("wwwwwwwwwwwwwwwwwwwwwwwww1")
            keras_class = int(classes_in_keras_format[image_class])
            print("keras_class:", keras_class)
            # print(keras_class)
            y_true.append(keras_class)
        except Exception:
            # print("os.path.split(i):", (os.path.split(i)[1]))
            y_true.append(os.path.split(i)[1])

        inputs.append(x)
        # print(inputs)
        # print(len(inputs))
        # print("y_true:", y_true)
        # print(len(y_true))
        #
        # print("qwerrt")
    return y_true, inputs


def predict(path):
    files = get_files(path)
    #print("files:", files)
    n_files = len(files)
    print('Found {} files'.format(n_files))

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    #print("nb_batch:", nb_batch)
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)
        # print(n_from)
        # print(n_to)
        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true
        # print("1111111111111111")
        #print(len(y_trues))
        # print("y_trues:", y_trues)
        #print(len(inputs))

        #print(inputs)

        # print("22222222222222222")
        # input = np.array(inputs)
        # print(input.shape)
        # print("3333333333333333333333333")

        if not args.store_activations:
            # Warm up the model
            # if n == 0:
            #     model.predict(np.array([inputs[0]]))
            #     print(model.predict(np.array([inputs[0]])))
            #     print(np.argmax(model.predict(np.array([inputs[0]])), axis=1))
            #     print("n=0")
            global out, out2

            out = model.predict(np.array(inputs))
            # print("out:", out)
            # out1 = np.array(out)
            # print("out.shape:", out1.shape)
            out2 = decode_predictions_custom(out, top=3)

            # print("top3:", out2)
            # print(len(out2))
            # out3 = np.array(out2)
            # print("out3.shape:", out3.shape)
            # print(np.argmax(out, axis=1))
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            #print("predictions[n_from:n_to]:", predictions[n_from:n_to])


        if not args.store_activations:
            for i, p in enumerate(predictions):
                print("predictions:", predictions)
                #print("enumerate(predictions):", enumerate(predictions))
                print("i:", i)
                print("p:", p)
                print("class00:", classes_in_keras_format)
                print("keys:", list(classes_in_keras_format.keys()))
                print("key1:", list(classes_in_keras_format.keys())[1])
                print("values:", list(classes_in_keras_format.values()))
                print("class index(p):", list(classes_in_keras_format.values()).index(p))
                #print("out[i]", out[i])
                recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
                #print('recognized_class:', recognized_class)
                print(' picture:{}   top3:{}'.format(y_trues[i], out2[i]))
                print(' picture:{}   in {} file files be predicted as {} class  , score : {}'.format(y_trues[i], files[i].split(os.sep)[-2],
                                                                                                    recognized_class, ("%.2f" % (100 * np.max(
                                                                                                                out[i])))))


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