import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras import layers, models
from pathlib import Path
import argparse
from tensorflow.keras.utils import Sequence
from tensorflow.keras.mixed_precision import set_global_policy, Policy


from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset

def main(args):
    print("TF Version: ", tf.__version__)
    print("Has CUDA: ", tf.test.is_built_with_cuda())
    tf.keras.backend.clear_session()

    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Mixed Precision (if supported)
    policy = Policy('mixed_float16')
    set_global_policy(policy)
    ## Prepare the data

    # Specify the paths to the data
    data_path = args.dataroot
    images_path = data_path + '/image/'
    metadata_path = data_path + '/metadata/'
    json_path = data_path + '/metadata.json'
    labels = ('year', 'month', 'day', 'hour', 'grade')

    def image_filter(image):
        return image.grade() < 7 and image.year() >= 2020

    def transform_func(image_ray):
        image_ray = np.clip(image_ray, standardize_range[0], standardize_range[1])
        image_ray = (image_ray - standardize_range[0]) / (standardize_range[1] - standardize_range[0])
        
        if downsample_size != (512, 512):
            image_ray = tf.convert_to_tensor(image_ray, dtype=tf.float32)
            image_ray = tf.reshape(image_ray, [1, image_ray.shape[0], image_ray.shape[1], 1])
            image_ray = tf.image.resize(image_ray, size=downsample_size, method='bilinear')
            image_ray = tf.reshape(image_ray, [downsample_size[0], downsample_size[1]])
            image_ray = image_ray.numpy()
        
        return image_ray

    # Load Dataset
    dataset = DigitalTyphoonDataset(str(images_path),
                                    str(metadata_path),
                                    str(json_path),
                                    labels='grade',
                                    get_images_by_sequence=True,
                                    filter_func=image_filter,
                                    transform_func=transform_func)

    train_set, test_set, val_set = dataset.random_split([0.7, 0.2, 0.1], split_by='sequence')

    ## Prepare the model
    num_epochs = args.max_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    standardize_range = (150, 350)
    downsample_size = (112, 112)
    input_sequence_length = args.input_sequence_length
    output_sequence_length = args.output_sequence_length
    
    train_loader = ConvLSTMDataLoader(train_set, batch_size=batch_size, target_size=downsample_size, 
                                      input_sequence_length=input_sequence_length, 
                                      output_sequence_length=output_sequence_length)
    test_loader = ConvLSTMDataLoader(test_set, batch_size=batch_size, target_size=downsample_size, 
                                     input_sequence_length=input_sequence_length, 
                                     output_sequence_length=output_sequence_length)
    val_loader = ConvLSTMDataLoader(val_set, batch_size=batch_size, target_size=downsample_size, 
                                    input_sequence_length=input_sequence_length, 
                                    output_sequence_length=output_sequence_length)

    sample_images, sample_labels = train_loader[0]
    print(f"Input Shape: {sample_images.shape}")  # Should be (batch_size, time_steps, height, width, channels)
    print(f"Output Shape: {sample_labels.shape}")  # Should be (batch_size, output_sequence_length, height, width, channels)


    input_shape = sample_images.shape[1:]  # (time_steps, height, width, channels)

    model = create_convlstm_model(input_shape, output_sequence_length, batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=tf.keras.losses.MeanSquaredError(),
                    metrics=['mae'])
    print("Model Summary:")
    print(model.summary())

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.fit(train_loader, epochs=num_epochs, validation_data=val_loader, batch_size=batch_size)

        test_loss, _ = model.evaluate(test_loader)
        print(f"Test results - Loss: {test_loss}, MAE: {_}")
        
        predictions = model(sample_images)
    print(f"Predictions Shape: {predictions.shape}")


# Define the encoding and forecasting networks
def encoding_network(input_shape):
    x = layers.Input(shape=input_shape, name='encoder_input')
    convlstm = layers.ConvLSTM2D(1, (5, 5), activation='relu', padding='same', return_sequences=True)(x)
    batchnorm = layers.BatchNormalization()(convlstm)
    return models.Model(inputs=x, outputs=batchnorm)

# the initial states and cell outputs of the forecasting network are copied from the last state of the encoding network.
def forecasting_network(encoding_out, output_sequence_length):
    convlstm = layers.ConvLSTM2D(1, (1, 1), activation='relu', padding='same', return_sequences=True)(encoding_out)
    # batchnorm = layers.BatchNormalization()(convlstm)
    
    output_conv = layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(convlstm)
    
    return models.Model(inputs=encoding_out, outputs=output_conv)

        
def create_convlstm_model(input_shape, output_sequence_length, batch_size):
    encoding = encoding_network(input_shape)
    forecasting = forecasting_network(encoding.output, output_sequence_length)
    
    inp = layers.Input(shape=input_shape, name='initial_input')
    
    encoding_out = encoding(inp)
    forecasting_out = forecasting(encoding_out)
    
    return models.Model(inputs=inp, outputs=forecasting_out)


class ConvLSTMDataLoader(Sequence):
    def __init__(self, dataset, batch_size, target_size=(112, 112), input_sequence_length=10, output_sequence_length=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(self.dataset))
        self.target_size = target_size
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        
        input_sequences = []
        output_sequences = []
        
        for sequence, label in batch:
            if len(sequence) >= self.input_sequence_length + self.output_sequence_length:
                for i in range(len(sequence) - self.input_sequence_length - self.output_sequence_length + 1):
                    input_seq = sequence[i:i+self.input_sequence_length]
                    output_seq = sequence[i+self.input_sequence_length:i+self.input_sequence_length+self.output_sequence_length]
                    
                    input_sequences.append(self.process_sequence(input_seq))
                    output_sequences.append(self.process_sequence(output_seq))

        if not input_sequences:
            # Handle the case where no valid sequences were found
            return np.zeros((0, self.input_sequence_length, *self.target_size, 1)), np.zeros((0, self.output_sequence_length, *self.target_size, 1))

        return np.array(input_sequences), np.array(output_sequences)

    def process_sequence(self, seq):
        processed_seq = []
        for img in seq:
            if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                img = np.zeros(self.target_size + (1,))
            elif len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            elif len(img.shape) != 3:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            
            img = img.astype(np.float32)
            resized_img = tf.image.resize(img, self.target_size)
            processed_seq.append(resized_img)
        
        return np.array(processed_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ConvLSTM model with TensorFlow')
    parser.add_argument('--dataroot', required=True, type=str, help='Path to the root data directory')
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', '--lr', default=0.001, type=float)
    parser.add_argument('--input_sequence_length', default=12, type=int, help='Number of input frames')
    parser.add_argument('--output_sequence_length', default=12, type=int, help='Number of frames to predict')
    args = parser.parse_args()

    main(args)