import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import parse_args
from event_stream import KafkaStream, GooglePubsub

args = parse_args()

def producer(broker_type, topic, image, gcp_creds=None, project_id=None):
    """Write data either to a kafka or pub sub producer

    Args:
        broker_type (str): Type of broker
        topic (_type_): Topic name
        image (_type_): Image data
        gcp_creds (str, optional): GCP credentials. Defaults to None.
        project_id (str, optional): GCP project ID. Defaults to None.
    """
    if broker_type=='kafka':
        kafka_broker = KafkaStream(topic)
        data = kafka_broker.data_encoder(image)
        kafka_broker.produce()
    else:
        pub_sub = GooglePubsub(topic, gcp_creds)
        pub_sub.produce(project_id)

BROKER_TYPE = args.broker_type
TOPIC = args.topic_name
IMAGE = args.image_path
GCP_CREDS = args.gcp_credentials
PROJECT_ID = args.project_id

# Run the producer function
producer(broker_type=BROKER_TYPE, topic=TOPIC, image=IMAGE, gcp_creds=GCP_CREDS, project_id=PROJECT_ID)
