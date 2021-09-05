import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from tkinter import *
from tkinter.colorchooser import askcolor

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import image
from PIL import ImageGrab


def make_model():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
      """Normalizes images: `uint8` -> `float32`."""
      return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(64)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(64)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=10,
        validation_data=ds_test,
    )

    model.save('mnist.model')

class Paint(object):

    DEFAULT_PEN_SIZE = 40.0
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()
        self.root.title("MNIST Tester")

        self.c = Canvas(self.root, bg='black', width=448, height=448)
        self.c.grid(row=0, column = 0)

        self.button_frame = Frame(self.root)
        self.button_frame.grid(row = 0, column = 1)

        self.eraser_button = Button(self.button_frame, 
                                    text='clear', 
                                    command=self.use_eraser,
                                    height = 5,
                                    width = 20)
        self.eraser_button.pack()

        self.choose_size_button = Scale(self.button_frame, 
                                        from_=20, 
                                        to=60, 
                                        orient=HORIZONTAL,
                                        length = 200)
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)
        self.choose_size_button.pack()


        self.num = StringVar()
        self.num.set("#")
        self.num_label = Label(self.button_frame, textvariable = self.num, font = ("Arial", 25))
        self.num_label.pack(ipadx = 100, ipady = 100)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.model = tf.keras.models.load_model('mnist.model')

        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def disp_num(self):
        x=self.root.winfo_rootx()+self.c.winfo_x()
        y=self.root.winfo_rooty()+self.c.winfo_y()
        x1=x+self.c.winfo_width()
        y1=y+self.c.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).resize((28, 28)).convert("L").save("/tmp/number.png",)
        k = image.imread('/tmp/number.png')
        k[:,[0, -1]] = k[[0,-1]] = 0
        k = np.array([k])

        preds = self.model.predict(k)
        self.num.set(np.argmax(preds))

    def use_eraser(self):
        self.c.delete("all")
        self.num.set("#")

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=100)


        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.disp_num()
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        make_model()
    else:
        Paint()