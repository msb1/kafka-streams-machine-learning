import io
import json
import csv
import time
from kafka import KafkaProducer

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def main():

    """
        Test script for Kafka Streams (Java) program kafka-streams-tester
        Classification data generated with scikits learn make_classification
    """

    # kafkaProducer = KafkaProducer(bootstrap_servers='localhost:9092')
    kafkaProducer = KafkaProducer(bootstrap_servers='192.168.5.4:9092', api_version=(0, 10, 1))

    # generate classification data
    X, y = datasets.make_classification(n_samples=101000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=42)
    # uncommment block of code to generate test data to train model
    # with open('skTestData.csv', "w") as csv_file:
    #     writer = csv.writer(csv_file, delimiter=',')
    #     for i in range(1000):
    #         rlist = X[i, :].tolist()
    #         rlist.append(y[i])
    #         writer.writerow(rlist)


    # Kafka topic
    kafkaTopic = "data1"
    # send data to Kafka
    for row in range(1000, 101000):
        key = str(row)
        # create JSON string
        jsonDict = {}
        jsonDict['feature']= X[row, :].tolist()
        jsonDict['result'] = int(y[row])
        jsonString = json.dumps(jsonDict)
        print(jsonString)
        kafkaProducer.send(kafkaTopic, key=key.encode(), value=jsonString.encode())
        time.sleep(1)


if __name__ == '__main__':
    main()
    print("data-gen-1 script is complete...")