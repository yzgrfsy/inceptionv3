import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
 
import os
from keras.models import load_model
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
#file_path = '/home/yuzhg/6/3148.jpg'



def decode_predictions_custom(preds, top=3):
    CLASS_CUSTOM = ["3147", "3148", "3182", "3184", "3191", "3193", "3201", "3202", "3203", "3205", "3206", "3207",
                    "3208", "3209", "3211", "3212", "3214", "3215", "3216", "3217", "3218", "3220", "3226", "3227",
                    "3228", "3229", "3230", "3238", "3240", "3249", "3253", "3281", "3283", "3284", "3285", "3286",
                    "3287", "3288", "3290", "3291", "3292", "3293", "3298", "3299", "3300"]

    results = []
    for pred in preds:
        #print("pred:", pred)
        # print(" pred.argsort():", pred.argsort())
        # print(len(pred.argsort()))
        top_indices = pred.argsort()[-top:][::-1]
        # print("top_indices:", top_indices)
        result = [(CLASS_CUSTOM[i] + ":" + str("%.2f" % (pred[i] * 100))) for i in top_indices]
        results.append(result)
        #print(results)
    return results

'''
def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(file_path + '*.jpg')
    elif file_path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files

'''
file_path = "/home/yuzhg/3/"
f_names = glob.glob(file_path + "*.jpg")
img = []
print(len(f_names))
for i in range(len(f_names)):
    images = image.load_img(f_names[i], target_size=(299, 299))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    #print(x)
    # x /= 255.
    # x -= 0.5
    # x *= 2.
    #print("qqqqqqqqqqqqqqqq")
    #print(x)
    img.append(x)

img1 = np.array(img)
print(img1.shape)
model = load_model('/home/yuzhg/Inception-v3/trained/last-model-inception_v3.h5')
for i in range(len(f_names)):
    x = np.concatenate([x for x in img])

    y = model.predict(x)
    global out2

    out2 = decode_predictions_custom(y, top=3)
    print('top3:{}'.format(out2[i]))
    # print('top3:', decode_predictions_custom(y, top=3)[0])


