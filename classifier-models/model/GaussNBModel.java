package com.barnwaldo.classifiers.model;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

@Getter
@Setter
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class GaussNBModel {

    private int numFeature;
    private int numClass;
    private String[] headers;
    private double[] priors;		// prior probabilities from training set (per class)
    private double[][] means;		// means from training set (per feature per class)
    private double[][] stdDevs;		// stdDevs from training set (per feature per class)

    public GaussNBModel(int numFeature, int numClass) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.priors = new double[numClass];
        this.means = new double[numFeature][numClass];
        this.stdDevs = new double[numFeature][numClass];
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
}
