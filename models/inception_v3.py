from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Embedding, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
import config
from .base_model import BaseModel
from keras import backend as K

class InceptionV3(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 1024

    def __init__(self, *args, **kwargs):
        super(InceptionV3, self).__init__(*args, **kwargs)

        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 86

        # self.img_size = (299, 299)

    def _create(self):
        #print("sdfsdvews")
        base_model = KerasInceptionV3(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
       # print("base_model.layers:", len(base_model.layers))
        #self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        feature = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        #x = Dropout(0.6)(feature)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(feature)


        if config.isCenterLoss:
            print(config.isCenterLoss)
            input_target = Input(shape=(None,))
            centers = Embedding(len(config.classes), 4096)(input_target)
            print('center:', centers)
            center_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='center_loss')(
                [feature, centers])
            self.center_model = Model(inputs=[base_model.input, input_target], outputs=[predictions, center_loss])

        elif config.isTripletLoss:
            self.triplet_model = Model(input=base_model.input, output=[predictions, feature])

        else:
            print(base_model.input)
            self.model = Model(input=base_model.input, output=predictions)



    # @staticmethod
    # def apply_mean(image_data_generator):
    #     pass
    '''
    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

        self.model.fit_generator(
            self.get_train_datagen(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=self.preprocess_input),
            samples_per_epoch=config.nb_train_samples,
            nb_epoch=self.nb_epoch,
            validation_data=self.get_validation_datagen(preprocessing_function=self.preprocess_input),
            nb_val_samples=config.nb_validation_samples,
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience),
            class_weight=self.class_weight)

        self.model.save(config.get_model_path())
    '''

def inst_class(*args, **kwargs):
    return InceptionV3(*args, **kwargs)
