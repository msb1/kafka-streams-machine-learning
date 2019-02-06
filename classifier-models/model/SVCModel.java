package com.barnwaldo.classifiers.model;

import libsvm.svm_model;
import libsvm.svm_parameter;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

@Getter
@Setter
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SVCModel {

    private int numFeature;
    private int numClass;
    private String[] headers;
    private svm_parameter svmParameter;			// parameters defined in libSVM
    private svm_model svmModel;					// complete model from libSVM

    public SVCModel(int numFeature, int numClass) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.svmParameter = new svm_parameter();
    }

    /**
     * Helper method to set libSVM parameters
     *
     * Note that Map must contain Double for each value... 
     * svm_type: C_SVC = 0; NU_SVC = 1; ONE_CLASS = 2; EPSILON_SVR = 3; NU_SVR = 4; 
     * kernel_type: LINEAR = 0; POLY = 1; RBF = 2; SIGMOID = 3; PRECOMPUTED = 4;
     *
     * @param params
     */
    public void setModelParameters(Map<String, Double> params) {
        // Define svm model and run fit to training data
        svmParameter.svm_type = params.get("svm_type").intValue();
        svmParameter.kernel_type = params.get("kernel_type").intValue(); // or POLY
        svmParameter.degree = params.get("degree").intValue(); // for POLY kernel only
        svmParameter.gamma = params.get("gamma"); // default to 1/numFeature
        svmParameter.coef0 = params.get("coef0"); // default to 0 for POLY only

        // these are for training only
        svmParameter.cache_size = params.get("cache_size"); // in MB
        svmParameter.eps = params.get("eps"); // stopping criteria
        svmParameter.C = params.get("C"); // for C_SVC, EPSILON_SVR and NU_SVR
        svmParameter.nr_weight = params.get("nr_weight").intValue(); // for C_SVC
        svmParameter.weight_label = new int[0]; // for C_SVC
        svmParameter.weight = new double[0]; // for C_SVC
        svmParameter.nu = params.get("nu"); // for NU_SVC, ONE_CLASS, and NU_SVR
        svmParameter.p = params.get("p"); // for EPSILON_SVR
        svmParameter.shrinking = params.get("shrinking").intValue(); // use the shrinking heuristics
        svmParameter.probability = params.get("probability").intValue(); // do probability estimates
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
