import json
import sys
import os

import tensorflow as tf

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from cnn_pipeline import CnnPipeline
from utils import parse_args, data_generator

def main(model_path, data_source, num_classes, data_dir, target_size, epoch):
    """A function that runs the entire CNN pipeline

    Args:
        model_path (int): Path to save the CNN model
        data_source (str): Specifies which data to train on. Either fashion mist or a custom data
        num_classes (int): NUmber of classes in train data
        data_dir (str): Path to train data
        target_size (int): Size to reshape train image to
        epoch (int): Number of epochs
    """

    LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    OPTIMIZER = "adam"

    train_data, test_data = data_generator(
        source=data_source,
        data_dir=data_dir,
        target_size=target_size
        )
    
    # Save class labels of custom trian data in a json file
    if data_source != 'fashion_mnist':
        labels = train_data.class_indices
        with open("label.json", "w") as outfile:
            json.dump(labels, outfile)

    model_pipeline = CnnPipeline(train_data, test_data)
    model_pipeline.model_architecture(target_size, num_classes, OPTIMIZER, LOSS)
    cnn_model = model_pipeline.train_CNNmodel(num_epoch=epoch)
    model_pipeline.model_accuracy()
    cnn_model.save("{}\model.h5".format(model_path))

if __name__ == "__main__":

    args = parse_args()

    MODEL_PATH = args.model_path
    DATA_SOURCE = args.data_source
    TARGET_SIZE = args.target_size
    NUM_CLASSES = args.num_class
    DATA_DIRECTORY = args.data_dir
    EPOCH = args.num_epoch

    main(
        model_path=MODEL_PATH, 
        data_source=DATA_SOURCE,  
        num_classes=NUM_CLASSES, 
        data_dir=DATA_DIRECTORY,
        target_size=TARGET_SIZE,
        epoch=EPOCH
        )