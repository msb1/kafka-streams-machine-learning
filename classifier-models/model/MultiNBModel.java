package com.barnwaldo.classifiers.model;

import com.barnwaldo.classifiers.data.Continuous;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class MultiNBModel {

    private int numFeature;
    private int numClass;
    private String[] headers;
    private int[] numLevel; // number of levels for each feature
    private double[] priors; // prior probabilities from training set (per class)
    private double[][][] frequency; // frequency (probability) for each feature for each class for each level
    private double[][] threshold; // thresholds for each level for each feature for continuous data

    public MultiNBModel(int numFeature, int numClass, int[] numLevel) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.numLevel = numLevel;
        this.priors = new double[numClass];
        this.frequency = new double[numFeature][numClass][];
        this.threshold = new double[numFeature][];
        for (int i = 0; i < numFeature; i++) {
            for (int j = 0; j < numClass; j++) {
                this.frequency[i][j] = new double[numLevel[i]];
            }
            this.threshold[i] = new double[numLevel[i] + 1];
        }
    }

    /**
     * Helper method to find thresholds from String[] data read from CSV file
     *
     * @param rawData
     * @param featureIndex
     */
    public void findThresholdsFromStringData(List<String[]> rawData, int featureIndex) {
        double minVal = 1.0e10;
        double maxVal = -1.0e10;
        for (String[] s : rawData) {
            double val = Double.parseDouble(s[featureIndex]);
            if (val > maxVal) {
                maxVal = val;
            }
            if (val < minVal) {
                minVal = val;
            }
        }
        double delta = (maxVal - minVal) / numLevel[featureIndex];
        threshold[featureIndex][0] = minVal;
        for (int i = 0; i < numLevel[featureIndex]; i++) {
            threshold[featureIndex][i + 1] = minVal + (i + 1) * delta;
        }
    }

    /**
     * Helper method to find thresholds from Continuous data objects
     *
     * @param data
     * @param featureIndex
     */
    public void findThresholdsFromContinuousData(List<Continuous> data, int featureIndex) {
        double minVal = 1.0e10;
        double maxVal = -1.0e10;
        for (Continuous c : data) {
            double val = c.getFeature()[featureIndex];
            if (val > maxVal) {
                maxVal = val;
            }
            if (val < minVal) {
                minVal = val;
            }
        }
        double delta = (maxVal - minVal) / numLevel[featureIndex];
        threshold[0][featureIndex] = minVal;
        for (int i = 0; i < numLevel[featureIndex]; i++) {
            threshold[i + 1][featureIndex] = minVal + (i + 1) * delta;
        }
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
