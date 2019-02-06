package com.barnwaldo.kafkastreamstester.utils;

import java.util.Arrays;

import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.KTable;
import org.springframework.cloud.stream.annotation.EnableBinding;
import org.springframework.cloud.stream.annotation.StreamListener;
import org.springframework.cloud.stream.binder.kafka.streams.annotations.KafkaStreamsProcessor;
import org.springframework.messaging.handler.annotation.SendTo;
import org.springframework.stereotype.Component;

import com.barnwaldo.kafkastreamstester.model.Analytics;
import com.barnwaldo.kafkastreamstester.model.Continuous;

/**
 * Spring Cloud Kafka Streams implementation of ML and DL models
 * 
 * @author barnwaldo
 * @version 1.0
 * @since Jan 11, 2019
 */
@Component
public class DataStreamListener {

    @EnableBinding(KafkaStreamsProcessor.class)
    public class DataAnalyticsProcessorApplication {

        /**
         *
         * @param input
         * @return
         */
        @StreamListener("input")
        @SendTo("output")
        public KStream<?, String> process(KStream<Object, Continuous> input) {


            KStream<String, Continuous> ostream = input
            		// inspect input data
                    .peek((key, value) -> {
                        System.out.println("\nKAFKA STREAM input -- " + key + "\nfeatures: " + Arrays.toString(value.getFeature()));
                    })
                    // perform classification 
                    .map((key, value) -> KeyValue.pair(String.valueOf(Analytics.getInstance().predictNN(value)), value))
                    // inspect classification results
                    .peek((key, value) -> {
                        System.out.println("Mapped Stream -- Deep Learning Classification: " + key);
                    });
      
            KTable<String, Long> mtable = ostream
                    .map((key, value) -> new KeyValue<>(key, "0"))
                    .groupByKey()
                    // count number of records classified in each class
                    .count();

            // return stream with updates when a class event is incremented
            return mtable.toStream()
                    .map((key, value) -> KeyValue.pair(key, "Class " + key + " counts: " + value));
        }

    }
}
