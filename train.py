from pixiv_grabber import PixivDataset
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import shutil
import datetime
from configs import *
from utils import *


if LOAD_MODEL:
    model = tf.keras.models.load_model(MODEL_LOAD_DIR)
else:
    model = tf.keras.applications.InceptionV3(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000,
        classifier_activation='softmax'
    )

    x = model.layers[-2].output
    output = Dense(len(all_characters), activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=model.input, outputs=output)

model.summary()
# quit()

if USE_TENSORBOARD:
    if DELETE_LOGS:
        try: shutil.rmtree('logs')
        except FileNotFoundError: pass
    train_log_dir = "logs/" + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_train"
    test_log_dir  = "logs/" + str(datetime.datetime.now().time().replace(microsecond=0)).replace(':', '_') + "_model_test"

    train_summary_writer = tf.summary.create_file_writer(logdir=train_log_dir)
    test_summary_writer  = tf.summary.create_file_writer(logdir=test_log_dir)

    def write_summary():
        tf.summary.scalar('loss', metrics_loss.result(), step=step)
        tf.summary.scalar('accuracy', metrics_accuracy.result(), step=step)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_obj  = tf.keras.losses.CategoricalCrossentropy()
metrics_loss      = tf.keras.metrics.Mean(name='train_loss')
metrics_accuracy  = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')

grabber = PixivDataset()

step = 0
for epoch_num in range(EPOCHS):

    # TRAINING #
    for X, y in grabber.train_ds:
        step += 1
        with tf.GradientTape() as tape:
            pred = model(X, training=True)
            loss = loss_obj(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metrics_loss.reset_states()
        metrics_accuracy.reset_states()

        metrics_loss(loss)
        metrics_accuracy(y, pred)

        show_pred, show_label = grabber.one_hot_decode(pred, top=3, round_num=3), \
                                grabber.one_hot_decode(y,    top=None, round_num=3)
        print(pd.DataFrame.from_dict(
            {'Prediction': list(show_pred),
             'Label': list(show_label)}
        ))
        train_result(epoch_num, EPOCHS, grabber.train_ds.batch_count, grabber.train_ds.num_batches,
                     metrics_loss.result(), metrics_accuracy.result(),
                     LEARNING_RATE, None, step)
        if USE_TENSORBOARD:
            with train_summary_writer.as_default(): write_summary()

    metrics_loss.reset_states()
    metrics_accuracy.reset_states()

    # TESTING #
    for X, y in grabber.test_ds:
        pred = model(X, training=False)
        loss = loss_obj(y, pred)

        metrics_loss(loss)
        metrics_accuracy(y, pred)

    test_result(epoch_num, EPOCHS,
                metrics_loss.result(), metrics_accuracy.result(), None)
    if USE_TENSORBOARD:
        with test_summary_writer.as_default(): write_summary()

    model.save(MODEL_SAVE_DIR.format(epoch_num+1,
            round(float(metrics_loss.result()), 3), round(float(metrics_accuracy.result()), 3)))