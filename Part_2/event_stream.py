import os
from io import BytesIO

import cv2
from concurrent.futures import TimeoutError
from PIL import Image
from kafka import KafkaProducer, KafkaConsumer
from google.cloud import pubsub_v1


class StreamRequest():

    def __init__(self, topic_name):
        self.topic = topic_name

    def data_encoder(self, image_path):
        """Data serializer

        Args:
            image_path (str): Path to image data to be produced

        Returns:
            data: encoded image data
        """
        image = cv2.imread(image_path)
        ret, data = cv2.imencode('.jpg', image)
        self.encoded_data = data
        return data

    def consume(self,):
        pass

    def produce(self,):
        pass

class KafkaStream(StreamRequest):

    def __init__(self, topic_name):
        super().__init__(topic_name)

    def consume(self, bootstrap_server=['localhost:9092']):
        """Consume message from kafka broker

        Args:
            bootstrap_server (list, optional): bootstrap server. Defaults to ['localhost:9092'].

        Returns:
            image: consumed image data
        """
        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=bootstrap_server,
            auto_offset_reset='latest',
            enable_auto_commit=True
            )

        print('----Starting the consumer----')

        for message in consumer:
            stream = BytesIO(message.value)
            image = Image.open(stream).convert("L")
            stream.close()
            break
        return image

    def produce(self, bootstrap_server=['localhost:9092']):
        """Produce message to a kafka broker

        Args:
            bootstrap_server (list, optional): Bootstrap server. Defaults to ['localhost:9092'].
        """
        producer=KafkaProducer(bootstrap_servers=bootstrap_server)
        producer.send(self.topic, self.encoded_data.tobytes())
        producer.flush()
        print('---Sending data to Kafka broker---')
    

class GooglePubsub(StreamRequest):

    def __init__(self, topic_name, path_to_credentials):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= path_to_credentials
        super().__init__(topic_name)
    
    def consume(self, subscription_id, project_id):
        """Cosume message from a pup sub producer

        Args:
            subscription_id (str): Subcriber ID
            project_id (str): Google project ID

        Returns:
            result: Consumed message
        """
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project_id, subscription_id)

        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            print(f"Received {message}.")
            message.ack()

        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        print(f"Listening for messages on {subscription_path}..\n")

        with subscriber:
            try:
                result = streaming_pull_future.result(timeout=15)
            except TimeoutError:
                streaming_pull_future.cancel()  # Trigger the shutdown.
                streaming_pull_future.result()  # Block until the shutdown is complete.
        
        return result

    def produce(self, project_id):
        """Write data to a pub sub producer

        Args:
            project_id (str): Google project ID
        """
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, self.topic)
        future = publisher.publish(topic_path, self.encoded_data)
        print(f"Published messages to {topic_path}.")
