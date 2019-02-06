package com.barnwaldo.kafkastreamstester;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.barnwaldo.kafkastreamstester.model.Analytics;

/**
 * Spring Boot Spring Cloud Kafka Streams test application to demonstrate the application of machine and deep
 * learning with Kafka Streams
 * 
 * @author barnwaldo
 * @version 1.0
 * @since Jan 11, 2019
 */
@SpringBootApplication
public class KafkaStreamsTesterApplication {

	public static void main(String[] args) {
       // Analytics.getInstance().initLRModel();
        Analytics.getInstance().initNNModel();
		SpringApplication.run(KafkaStreamsTesterApplication.class, args);
	}

}

