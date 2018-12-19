from __future__ import print_function
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
import os
from keras.callbacks import *
# import visualize
import config
import util
import tensorflow as tf
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
    train_data = get_train_datagen(rotation_range=30., shear_range=0.2,
                                            zoom_range=0.2, horizontal_flip=True)
    checkpoint_dir = os.path.join(os.path.abspath('.'), 'checkpoint')
    callbacks = get_callbacks(config.get_fine_tuned_weights_path(), checkpoint_dir,
                                       patience=fine_tuning_patience)

    # input_target = Input(shape=(None,))
    # centers = Embedding(len(config.classes), 4096)(input_target)  # Embedding层用来存放中心
    # print('center:', centers)
    # center_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='center_loss')(
    #     [feature, centers])
    '''
    centers = get_center_loss(0.2,len(config.classes),x,x.shape[1])
    center_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [x, centers])
    '''
    # center_model = Model(inputs=[base_model.input, input_target], outputs=[predictions, center_loss])

    center_model = load_model('/home/yuzhg/keras-transfer-learning-for-oxford102/trained/last-model-vgg16.h5')

    center_model.compile(loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                       loss_weights=[1, 0.2], metrics=['accuracy'], optimizer=Adam(lr=1e-5))
    center_model.summary()
    history = center_model.fit_generator(
        util.clone_y_generator(train_data),
        steps_per_epoch=655535 / 32),
        epochs=2,
        validation_data=util.clone_y_generator(get_validation_datagen()),
        validation_steps=2,)
        center_model.save('last-vgg16.h5')


    def train(self):
        print("Creating model...")
        self._create()
        print("the number of class is %s", config.classes)
        print(len(config.classes))
        print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        #self._fine_tuning_()
        self.save_classes()
        print("Classes are saved")

    def load(self):
        print("Creating model")
        self.load_classes()
        self._create()
        self.model.load_weights(config.get_fine_tuned_weights_path())
        return self.model

    @staticmethod
    def save_classes():
        joblib.dump(config.classes, config.get_classes_path())

    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, dir_path, patience=30, monitor='val_loss'):
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        tensorboard = TensorBoard(log_dir=dir_path, histogram_freq=0, write_graph=True, write_images=False)
        # histories = Histories(config.isCenterLoss)
        return [early_stopping, model_checkpoint, tensorboard]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the datasets mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def load_classes():
        config.classes = joblib.load(config.get_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.classes)
    '''
    def get_train_datagen_(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow(x=x_train, y=y_train, batch_size=32, shuffle=True)
    '''
    def get_validation_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.validation_dir, target_size=self.img_size, classes=config.classes)

