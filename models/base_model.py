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
#from keras.layers import Lambda
import triplet_loss
from keras import backend as K

class BaseModel(object):
    def __init__(self,
                 class_weight=None,
                 nb_epoch=1000,
                 batch_size=32,
                 freeze_layers_number=None):
        self.model = None

        self.center_model = None
        self.triplet_model = None

        self.class_weight = class_weight
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.freeze_layers_number = freeze_layers_number
        self.fine_tuning_patience = 20

        self.img_size = (299, 299)
        self.history = None

    def _create(self):
        return NotImplementedError('subclasses must override _create()')
        print('subclasses must override _create()')

    def hard_triplet_loss(self, y_true, y_pred):
        print(y_true)
        print(y_true.shape)
        y_true = y_true[:, 0]
        print(y_true)
        print(y_true.shape)
        return triplet_loss.batch_hard_triplet_loss(y_true, y_pred, margin=0.6, squared=False)

    def _fine_tuning(self):
        self.freeze_top_layers1()
        train_data = self.get_train_datagen(rotation_range=30., shear_range=0.2,
                                            zoom_range=0.2, horizontal_flip=True,
                                            preprocessing_function=self.preprocess_input)
        checkpoint_dir = os.path.join(os.path.abspath('.'), 'checkpoint')
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(), checkpoint_dir,
                                       patience=self.fine_tuning_patience)

        if util.is_keras2():
            if config.isCenterLoss:
                self.center_model.load_weights('/home/yuzhg/Inception-v3/trained/fine-tuned-best-inception-weights.h5' ,
                                               by_name=True)
                self.center_model.compile(loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred],
                                          loss_weights=[1, 0.2], metrics=['accuracy'], optimizer=Adam(lr=1e-5))
                self.center_model.summary()
                self.history = self.center_model.fit_generator(
                    util.clone_y_generator(train_data),
                    steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                    epochs=self.nb_epoch,
                    validation_data=util.clone_y_generator(self.get_validation_datagen()),
                    validation_steps=config.nb_validation_samples / float(self.batch_size),
                    callbacks=callbacks,
                    class_weight=self.class_weight
                )
            elif config.isTripletLoss:
                self.triplet_model.load_weights('/home/yuzhg/Inception-v3/trained/fine-tuned-best-inception-weights.h5',
                                                by_name=True)
                #self.triplet_model.compile(loss=self.hard_triplet_loss, optimizer=Adam(lr=1e-5), metrics=['accuracy'])
                self.triplet_model.compile(optimizer=Adam(lr=1e-5), loss=['categorical_crossentropy', self.hard_triplet_loss],
                                           loss_weights=[1.0, 1.0], metrics=['accuracy'])
                self.triplet_model.summary()
                valid_data = self.get_validation_datagen(rotation_range=30., shear_range=0.2, zoom_range=0.2,
                                                         horizontal_flip=True, preprocessing_function=self.preprocess_input)

                # util.clone_y_generator1(train_data),
                self.history = self.triplet_model.fit_generator(
                    #util.triplet_transformed_generator(train_data, 4096),
                    util.clone_y_generator1(train_data),
                    steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                    epochs=self.nb_epoch,
                    #validation_data=util.triplet_transformed_generator(valid_data, 4096),
                    validation_data=util.clone_y_generator1(valid_data),
                    validation_steps=config.nb_validation_samples / float(self.batch_size),
                    callbacks=callbacks,
                    class_weight=self.class_weight
                )
            else:
                self.model.load_weights('/home/yuzhg/Inception-v3/trained/fine-tuned-best-inception-weights.h5',
                                        by_name=True)
                self.model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(lr=1e-5),
                    metrics=['accuracy']
                )

                self.model.summary()
                self.history = self.model.fit_generator(
                    train_data,
                    steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                    epochs=self.nb_epoch,
                    validation_data=self.get_validation_datagen(rotation_range=30., shear_range=0.2,
                                                                zoom_range=0.2, horizontal_flip=True,
                                                                preprocessing_function=self.preprocess_input),
                    validation_steps=config.nb_validation_samples / float(self.batch_size),
                    callbacks=callbacks,
                    class_weight=self.class_weight
                )


        # else:
        #     if config.isCenterLoss:
        #         self.center_model.compile(loss=['categorical_crossentropy', lambda y_true, y_pred:y_pred],
        #                            loss_weights=[1, 0.2], metrics=['accuracy'],
        #                            optimizer=Adam(lr=1e-5))
        #         self.center_model.summary()
        #         self.history = self.center_model.fit_generator(
        #             util.clone_y_generator(train_data),
        #             samples_per_epoch=config.nb_train_samples,
        #             nb_epoch=self.nb_epoch,
        #             validation_data=util.clone_y_generator(self.get_validation_datagen()),
        #             nb_val_samples=config.nb_validation_samples,
        #             callbacks=callbacks,
        #             class_weight=self.class_weight)
        #     elif config.isTripletLoss:
        #         self.triplet_model.compile(loss=triplet_loss, optimizer=Adam(lr=1e-5))
        #         self.triplet_model.summary()
        #         self.history = self.triplet_model.fit_generator(
        #             util.clone_y_generator(train_data),
        #             steps_per_epoch=config.nb_train_samples / float(self.batch_size),
        #             epochs=self.nb_epoch,
        #             validation_data=util.clone_y_generator(self.get_validation_datagen()),
        #             validation_steps=config.nb_validation_samples / float(self.batch_size),
        #             callbacks=callbacks,
        #             class_weight=self.class_weight
        #         )
        #     else:
        #         self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
        #         self.model.summary()
        #         self.history = self.model.fit_generator(
        #             train_data,
        #             steps_per_epoch=config.nb_train_samples / float(self.batch_size),
        #             epochs=self.nb_epoch,
        #             validation_data=self.get_validation_datagen(),
        #             validation_steps=config.nb_validation_samples / float(self.batch_size),
        #             callbacks=callbacks,
        #             class_weight=self.class_weight
        #         )

        if config.isCenterLoss:
            #self.center_model.save_weights('vgg16-model-weights.h5')
            self.center_model.save(config.get_model_path())
            util.save_history(self.history, self.center_model)
        elif config.isTripletLoss:
            self.triplet_model.save(config.get_model_path())
            util.save_history(self.history, self.triplet_model)
        else:
            self.model.save(config.get_model_path())
            util.save_history(self.history, self.model)

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
        #print("config.get_fine_tuned_weights_path()", config.get_fine_tuned_weights_path())
        self.model.load_weights(config.get_fine_tuned_weights_path())
        print("")
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
            print("Freezing1 {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    def freeze_top_layers1(self):
        if self.freeze_layers_number:
            if config.isCenterLoss:
                print("CenterLoss Freezing {} layers".format(self.freeze_layers_number))
                for layer in self.center_model.layers[:self.freeze_layers_number]:
                    layer.trainable = False
                for layer in self.center_model.layers[self.freeze_layers_number:]:
                    layer.trainable = True
            elif config.isTripletLoss:
                print("TripletLoss Freezing {} layers".format(self.freeze_layers_number))
                for layer in self.triplet_model.layers[:self.freeze_layers_number]:
                    layer.trainable = False
                for layer in self.triplet_model.layers[self.freeze_layers_number:]:
                    layer.trainable = True
            else:
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
        print("config.get_classes_path()", config.get_classes_path())
        config.classes = joblib.load(config.get_classes_path())

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        print("qwe1")
        print(x)

        print(x.dtype)
        return x

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        print("img0:", img)
        print(img.mode)
        img = np.array(img)
        print(img.shape)
        #print("imgdtype:", img.dtype)
        x = image.img_to_array(img)
        print("img1dtype:", x.dtype)
        print("img1:", x)
        # x = x[:, :, ::-1]
        #
        # print("img2:", x)
        # print(x.ndim)
        # print(x.shape)
        x = np.expand_dims(x, axis=0)
        print("1111111111111111111111111111")
        print(x)
        # print(x.ndim)
        # print("2222222222222222222222")

        return self.preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.classes)

    def get_validation_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        self.apply_mean(idg)
        return idg.flow_from_directory(config.validation_dir, target_size=self.img_size, classes=config.classes)

    def triplet_loss(self, y_true, y_pred):
        y_pred = K.l2_normalize(y_pred, axis=1)
        batch = self.batch_size
        ref1 = y_pred[0:batch, :]
        pos1 = y_pred[batch:batch + batch, :]
        neg1 = y_pred[batch + batch:3 * batch, :]
        dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
        dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
        dis_pos = K.sqrt(dis_pos)
        dis_neg = K.sqrt(dis_neg)
        a1 = 0.6
        d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
        return K.mean(d1)

    def triplet_hard_loss(self, y_true, y_pred):
        global SN
        global PN
        feat_num = SN * PN  # images num
        y_pred = K.l2_normalize(y_pred, axis=1)
        feat1 = K.tile(K.expand_dims(y_pred, axis=0), [feat_num, 1, 1])
        feat2 = K.tile(K.expand_dims(y_pred, axis=1), [1, feat_num, 1])
        delta = feat1 - feat2
        dis_mat = K.sum(K.square(delta), axis=2) + K.epsilon()  # Avoid gradients becoming NAN
        dis_mat = K.sqrt(dis_mat)
        positive = dis_mat[0:SN, 0:SN]
        negative = dis_mat[0:SN, SN:]
        for i in range(1, PN):
            positive = tf.concat([positive, dis_mat[i * SN:(i + 1) * SN, i * SN:(i + 1) * SN]], axis=0)
            if i != PN - 1:
                negs = tf.concat([dis_mat[i * SN:(i + 1) * SN, 0:i * SN], dis_mat[i * SN:(i + 1) * SN, (i + 1) * SN:]],
                                 axis=1)
            else:
                negs = tf.concat(dis_mat[i * SN:(i + 1) * SN, 0:i * SN], axis=0)
            negative = tf.concat([negative, negs], axis=0)
        positive = K.max(positive, axis=1)
        negative = K.min(negative, axis=1)
        a1 = 0.6
        loss = K.mean(K.maximum(0.0, positive - negative + a1))
        return loss



