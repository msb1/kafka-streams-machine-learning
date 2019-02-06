package com.barnwaldo.kafkastreamstester.model;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import lombok.AllArgsConstructor;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class LRModel {

    private int numFeature;
    private int numClass;
    private double alpha;				// learning rate for gradient descent
    private double regL1;				// L1 regularization 
    private String[] headers;
    private double[][] w;				// weights for transfer function

    public LRModel(int numFeature, int numClass, double alpha, double regL1) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.alpha = alpha;
        this.regL1 = regL1;
        w = new double[numFeature][numClass];
    }

    @Override
    public String toString() {
        ObjectMapper mapper = new ObjectMapper();

        String jsonString = "";
        try {
            mapper.enable(SerializationFeature.INDENT_OUTPUT);
            jsonString = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            System.out.println(e.getMessage());
        }
        return jsonString;
    }

    public void jsonToFile(String filename) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            mapper.writeValue(new File(filename), this);
        } catch (IOException ex) {
            Logger.getLogger(LRModel.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
