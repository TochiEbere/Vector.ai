from keras import layers, models

class CnnPipeline:

    def __init__(self, train_data, test_data):
        """Initializes a CnnPipeline

        Args:
            train_data (tf data generator): train data
            test_data (tf data generator): test data
        """
        self.train_data = train_data
        self.test_data = test_data
        
    def model_architecture(self, target_size, num_class, optimizer, loss):            
        """Compiles a CNN architechure

        Args:
            target_size (int): target size to reshape input image to
            num_class (int): number of classes in train data
            optimizer (str): Optimization function
            loss (tf object): Loss function
        """
        INPUT_SHAPE = (target_size, target_size, 1)
        METRICS = ['accuracy']

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_class))
        model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)
        self.model_arch = model

    def train_CNNmodel(self, num_epoch):
        """Trains a CNN classifier

        Args:
            num_epoch (int): Number of epochs

        Returns:
            cnn_model: A CNN classifier
        """
        cnn_model = self.model_arch
        history = cnn_model.fit(self.train_data, epochs=num_epoch, 
                    validation_data=self.test_data)
        self.model_history = history
        self.model = cnn_model
        return cnn_model      

    def model_accuracy(self):
        """Evaluates accuracy of the CNN model"""
        history = self.model_history
        print('Train accuracy score is:', history.history['accuracy'][-1])    
        print('Test accuracy score is:', history.history['val_accuracy'][-1])    
