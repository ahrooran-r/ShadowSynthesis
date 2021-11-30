import glob
import os
import pathlib
import random

import cv2
import numpy as np
import tensorflow as tf

from networks import build_shadow_generator

tf.get_logger().setLevel('ERROR')


class Synthesizer:
    def __init__(self, model_path, original_path, mask_path, source_path, target_path, final_mask_path, width=1920,
                 height=1440, channel=64):
        self.dataset = []
        self.__width = width
        self.__height = height
        self.__target_path = target_path
        self.__source_path = source_path
        self.__final_mask_path = final_mask_path

        self.__setup_dataset(original_path, mask_path)
        self.__load_model(model_path, channel)

    def __load_model(self, model_path, channel):
        with tf.variable_scope(tf.get_variable_scope()):
            self.__input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.__mask = tf.placeholder(tf.float32, shape=[None, None, None, 1])

        # build the model
        # I_s = I_ns * I_sm
        self.__shadowed_image = build_shadow_generator(tf.concat([self.__input, self.__mask], axis=3),
                                                       channel) * self.__input

        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())
        idtd_ckpt = tf.train.get_checkpoint_state(model_path)
        saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
        print('loaded ' + idtd_ckpt.model_checkpoint_path)
        saver_restore.restore(self.__sess, idtd_ckpt.model_checkpoint_path)

    def __setup_dataset(self, original_path, mask_path, ext=["jpg", "png"]):
        original_files = []
        [original_files.extend(glob.glob(original_path + '*.' + e)) for e in ext]

        mask_files = random.sample(
            [os.path.join(mask_path, x) for x in os.listdir(mask_path)],
            len(original_files)
        )

        for img in range(len(original_files)):
            self.dataset.append((original_files[img], mask_files[img]))
        del original_files, mask_files

    def execute(self):

        for img_path, mask_path in self.dataset:
            iminput = cv2.resize(cv2.imread(img_path, 1), (self.__width, self.__height))
            immask = cv2.resize(cv2.imread(mask_path, 1), (self.__width, self.__height))

            imoutput = self.__sess.run(
                self.__shadowed_image,
                feed_dict={
                    self.__input: np.expand_dims(iminput / 255., axis=0),
                    self.__mask: np.expand_dims(immask[:, :, 0:1] / 255., axis=0)
                }
            )

            imshadow = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput, 0.0), 1.0)) * 255.0)

            # save cropped input to source folder
            imname = pathlib.Path(img_path).stem
            cv2.imwrite(os.path.join(self.__source_path, f'{imname}.jpg'), iminput)

            # save cropped mask into final_mask folder
            cv2.imwrite(os.path.join(self.__final_mask_path, f'{imname}.jpg'), immask)

            # save to target folder
            cv2.imwrite(os.path.join(self.__target_path, f'{imname}.jpg'), imshadow)
            cv2.waitKey(0)

            print(f"completed {imname}.jpg")
