from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input, Embedding, Lambda
import config
from .base_model import BaseModel
from keras import backend as K
import tensorflow as tf
import functools

'''
def _center_loss_func(features, labels, alpha, num_classes,
                      centers, second_features, feature_dim):
    assert feature_dim == second_features.get_shape()[1]
    centers = Embedding(num_classes, feature_dim)(labels)  # Embedding层用来存放中心
    # loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')
    # ([second_features, centers])
    return centers


def get_center_loss(alpha, num_classes, features, feature_dim):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # features 是为了存储倒数第二层的值的
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([len(config.classes)], feature_dim)

    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha,
                                 num_classes, centers, features, feature_dim)
    return center_loss

'''


class VGG16(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(VGG16, self).__init__(*args, **kwargs)

    def _create(self):

        base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        #base_model = KerasVGG16(weights='imagenet', include_top=False, input_shape=(165, 100, 3))
        self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        print("the vgg16 layers is!!!!!!!!!!!!!!!!!!!!!!!")
        print(base_model.layers[3].output.shape)
        # print(base_model.layers[18].output.shape)
        # print(base_model.layers[19].output.shape)
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        feature = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(feature)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)
        # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
        # print(len(config.classes))
        # self.model = Model(input=base_model.input, output=predictions).save()

        if config.isCenterLoss:
            print(config.isCenterLoss)
            # input_2 chu
            input_target = Input(shape=(None,))
            #print(input_target)
            centers = Embedding(len(config.classes), 4096)(input_target)
            print('center:', centers)
            #print(centers.ndim)
            # print(centers.shape)
            # print(centers[:, 0])
            # print(centers[:, 1])
            center_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='center_loss')(
                [feature, centers])
            self.center_model = Model(inputs=[base_model.input, input_target], outputs=[predictions, center_loss])

        elif config.isTripletLoss:
            self.triplet_model = Model(input=base_model.input, output=[predictions, feature])

        else:
            print(base_model.input)  # tensor("input_1:0")
            self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return VGG16(*args, **kwargs)

