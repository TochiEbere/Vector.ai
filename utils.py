import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

def parse_args():
    """Command line arguments to be used for various scripts

    Returns:
        args: All defined command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Command line arguments", 
        allow_abbrev=False)

    parser.add_argument('--data_source', type=str,
                        default='fashion_mnist', choices=['fashion_mnist','custom'],
                        help='Either fashion_mnist or custom')

    parser.add_argument('--data_dir', type=str,
                        default=None,
                        help='Directory path to data')

    parser.add_argument('--num_epoch', type=int,
                        default=10,
                        help='Number of epochs')

    parser.add_argument('--num_class', type=int,
                        default=10,
                        help='Number of classes') 

    parser.add_argument('--model_path', type=str,
                        default='model',
                        help='Path to save the CNN model in')

    parser.add_argument('--labels', type=str,
                        default=None, help='Json file of labels')

    parser.add_argument('--model', type=str,
                        help='Full path to CNN model')

    parser.add_argument('--target_size', type=int,
                        default=28,
                        help='Dimension to which input image will be resized')  

    parser.add_argument('--broker_type', type=str,
                        default='kafka', choices=['kafka', 'googelpubsub'],
                        help='Broker type to use. Options are kafka or googlepubsub')

    parser.add_argument('--topic_name', type=str,
                        help='Topic name')

    parser.add_argument('--image_path', type=str,
                        help='Full path to image file')

    parser.add_argument('--gcp_credentials', type=str,
                        default=None,help='Full path to GCP credentials')
    
    parser.add_argument('--project_id', type=str,
                        default=None, help='GCP service account project ID')

    parser.add_argument('--subscription_id', type=str,
                        default=None, help='Subscrition ID of Google Pubsub consumer')
    
    args = parser.parse_args()
    return args


def data_generator(source, data_dir=None, target_size=None):
    """Generates data either from the in-built fashion mnist datadet or a custome data from a directory
    
    Args:
        source (str): Specifies which data to tain on. Either of fashion_mnist or custom
        data_dir (str): Directory to call train data from
        target_size (str): Target size to reshape input image to

    Returns:
        train: A train data generator
        test: A test data generator
    """

    if source=='fashion_mnist':
        # Load in-built fashion mnist dataset
        print("""Loading fashion mnist data from tensorflow""")
        train, test = tfds.load(
            "fashion_mnist", 
            split=["train", "test"],
            shuffle_files=True,
            as_supervised=True,)
        
        # Normalizing the dataset
        def normalize_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        train = train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        test = test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        train = train.batch(32)
        test = test.batch(32)
        
        return train, test

    else:

        # Generate custom train data
        datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
        train_data_generator = datagen.flow_from_directory(
            data_dir,
            target_size = (target_size, target_size),
            class_mode = 'sparse',
            color_mode='grayscale',
            shuffle = True,
            subset='training'
            )

        # Generate custom test data        
        test_data_generator = datagen.flow_from_directory(
            data_dir,
            target_size = (target_size, target_size),
            class_mode = 'sparse',
            color_mode='grayscale',
            shuffle = True,
            subset='validation'
            )
    
        return train_data_generator, test_data_generator
