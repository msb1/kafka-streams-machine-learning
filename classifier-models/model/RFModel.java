package com.barnwaldo.classifiers.model;

import lombok.Getter;
import lombok.Setter;
import lombok.NoArgsConstructor;

import com.barnwaldo.classifiers.data.Tree;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

@Getter
@Setter
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class RFModel {

    private int numFeature;
    private int numClass;
    private int numTree;
    private String[] headers;
    private int[] rootTreeId;
    private Tree[] trees;

    public RFModel(int numFeature, int numClass, int numTree) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.numTree = numTree;
        this.rootTreeId = new int[numTree];
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
