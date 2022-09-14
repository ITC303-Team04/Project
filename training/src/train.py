import argparse
import os
import tarfile
import boto3
import numpy as np
import tensorflow as tf
import cv2 as cv
import pandas as pd

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import backend as K
    
# //==================== create df ====================//
def createDataFrame(train_dir):
    sat_image = []
    mask_image = []

    for subdir, _, files in os.walk(train_dir):
        for file in files:
            fp = os.path.join(subdir, file)
            if(subdir[-3:] == 'img'):
                sat_image.append(fp)
            elif(subdir[-3:] == 'lbl'):
                mask_image.append(fp)
    
    return pd.DataFrame({
        'Image' : sat_image,
        'Mask' : mask_image
    })

# //==================== create dataset ====================//
def modify_mask(mask):
    mask = np.expand_dims(mask, axis = 2)
    t_mask = np.zeros(mask.shape)
    np.place(t_mask[:, :, 0], mask[:, :, 0] >=100, 1)
    return t_mask

def map_function(img, mask, img_shape):
    img_shape_tuple = tuple(img_shape[:2])
    
    img = cv.imread(img.decode())
    mask = cv.imread(mask.decode(), cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, img_shape_tuple)
    mask = modify_mask(cv.resize(mask, img_shape_tuple))
    img = img/255.0

    return img.astype(np.float64), mask.astype(np.float64)

def create_dataset(data, batch_size, img_shape):
    dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda img, mask : tf.numpy_function(
                    map_function, [img, mask, img_shape], [tf.float64, tf.float64]),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(batch_size)

    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return dataset

# //==================== create model ====================//
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0) 
 
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def unet(learning_rate, input_shape, num_classes):
  inp = Input(input_shape)
  # Block 1
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(inp)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
  x = BatchNormalization()(x)
  block_1_out = Activation('relu')(x)
  x = MaxPooling2D()(block_1_out)
  # Block 2
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
  x = BatchNormalization()(x)
  block_2_out = Activation('relu')(x)
  x = MaxPooling2D()(block_2_out)
  # Block 3
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
  x = BatchNormalization()(x)
  block_3_out = Activation('relu')(x)
  x = MaxPooling2D()(block_3_out)
  # Block 4
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
  x = BatchNormalization()(x)
  block_4_out = Activation('relu')(x)

  x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP2')(block_4_out)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_3_out])
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(256, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # UP 3
  x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP3')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_2_out])
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(128, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # UP 4
  x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP4')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Concatenate()([x, block_1_out])
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3), padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_classes, (3, 3), activation='sigmoid', padding='same')(x)
 
  model = Model(inputs=inp, outputs=x)
  adam = tf.keras.optimizers.Adam(lr=learning_rate)
  model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
  return model

# //==================== AWS functions ====================//
def log(msg, output=''):
    return f'{"="*20} [{msg.upper}] {"="*20} \n{output}'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--img-shape', type=tuple, default=(512, 512, 3))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--retrain-model', type=bool, default=False)
    parser.add_argument('--bucket', type=str)

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args, _ = parser.parse_known_args()
    return args

def retrain_existing_model(file, bucket):
    log('Not yet implemented')
    exit()
#     path = f'opt/ml/retrain/{file}'
#     download_from_s3(file, bucket, path)
    
#     with tarfile.open(path) as model_tar:
#         for entry in model_tar:
#             #
            
#     model = load_model(model_path, compile=True, custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef}) 
#     return model

def compress_model():
    with tarfile.open(os.path.join(args.model_dir, 'model.tar.gz'), "w:gz") as tar:
        tar.add(os.path.join(args.model_dir, 'model.h5'), arcname='model.h5')
        tar.add(os.path.join(args.model_dir, 'model.json'), arcname='model.json')
    return tar

def upload_to_s3(path, file, bucket_name):
    s3 = boto3.Session().resource('s3')
    bucket = s3.Bucket(bucket_name[5:])
    obj = bucket.Object(os.path.join('model', file))

    try:
        response = obj.upload_file(os.path.join(path, file))
        
    except Exception as e:
        log('s3 upload failed', e)
        
def download_from_s3(file_name, bucket_name, path):
    log('Not yet implemented')
    exit()
#     s3 = boto3.Session().resource('s3')
#     bucket = s3.Bucket(bucket_name[5:])
#     key = f'retrain/{file_name}'
#     for file in bucket.objects.all():
#         if file.key == key:
#             bucket.download_file(file.key, file.key)


if __name__ =='__main__':

    args = parse_args()

    if args.retrain_model:
        model = retrain_existing_model('model.tar.gz', args.bucket)
    else:
        model = unet(
            learning_rate=args.learning_rate,
            input_shape=args.img_shape,
            num_classes=args.num_classes,
        )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, 'model.h5'),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30),
    ]
    
    df = createDataFrame(args.train)   
    train_dataset = create_dataset(df, args.batch_size, args.img_shape)

    history = model.fit(train_dataset, callbacks=callbacks, epochs=args.epochs)

    model_json = model.to_json()
    with open(os.path.join(args.model_dir, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    
    compressed_model = compress_model()
    upload_to_s3(args.model_dir, 'model.tar.gz', args.bucket)