### kafka-streams-machine-learning
* Spring Boot application including Spring Cloud Stream with Kafka Streams
* Logistic Regression (LR) and Neural Net (NN) Model included
* LR is written in Java for classification; the model is trained in another app and the model parameters are included in a JSON file
* A NN model is developed in Keras for a similar purpose; the model is trained and save to file. The files is imported with DL4J (Java) and executed
* The classification is performed real-time on the Kafka Stream as data arrive arrives at the stream; the output classification is then grouped and counted in a KTable and converted back to a KStream which is output and read by a standard Spring KafkaListener (consumer)
* Input data to the Kafka Stream is generated with a Python script using Kafka-Python and SciKits Learn
* Another Python script is also included that was used to train the NN model
