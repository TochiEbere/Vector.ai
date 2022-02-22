import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import parse_args
from part_2.event_stream import KafkaStream, GooglePubsub


def consumer(broker_type, topic, gcp_creds=None, sub_id=None, project_id=None):
    """A consumer function that cosumes data from either a kafka or pubsub broker

    Args:
        broker_type (str): Type of broker to consume message from. Either of Kafka or GooglePubsub
        topic (str): Topic name
        gcp_creds (str, optional): GCP credentials. Defaults to None.
        sub_id (str, optional): Pub sub Subscrition ID. Defaults to None.
        project_id (str, optional): Google project ID. Defaults to None.

    Returns:
        result: consumed message
    """

    if broker_type=='kafka':
        kafka_broker = KafkaStream(topic)
        image = kafka_broker.consume()
        return image

    else:
        pub_sub = GooglePubsub(topic, gcp_creds)
        result = pub_sub.consume(sub_id, project_id)
        return result

args = parse_args()
BROKER_TYPE = args.broker_type
GCP_CREDS = args.gcp_credentials
PROJECT_ID = args.project_id
SUB_ID = args.subscription_id
TOPIC = args.topic_name

# Run the consumer function
result = consumer(broker_type=BROKER_TYPE, topic=TOPIC, gcp_creds=GCP_CREDS, sub_id=SUB_ID, project_id=PROJECT_ID)