package com.barnwaldo.kafkastreamstester.model;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Singleton class to initialize and apply analytics models
 * 
 * For test purposes, have included 
 * 
 *      (1) Logistic Regression model written in Java with model parameter fit from another application
 * 		(2) Neural Net model trained in Keras and save to 'h5' file which is read in by DL4J methods and then used
 * 			for streaming classification 
 * 
 *
 * @author barnwaldo
 * @version
 * @since Jan 11, 2019
 */
@Getter
@Setter
public class Analytics {

    private LogisticRegression lr;
    private List<Continuous> test;
    private final String jsonModelFilename = "src/main/java/com/barnwaldo/kafkastreamstester/lrModel.json";
    private final String kerasModelFilename = "src/main/java/com/barnwaldo/kafkastreamstester/test_model_gen1.h5";
    private MultiLayerNetwork modelNN;
    private INDArray indArray;

    private static class AnalyticsStateHelper {

        private static final Analytics INSTANCE = new Analytics();
    }

    public static Analytics getInstance() {
        return AnalyticsStateHelper.INSTANCE;
    }

    public Analytics() {
        test = new ArrayList<>();
        test.add(new Continuous());
    }

    /**
     * Initialize Logistic regression model with parameters fit from another app
     */
    public void initLRModel() {
        try {
            ObjectMapper mapper = new ObjectMapper();
            LRModel lrm = mapper.readValue(new File(jsonModelFilename), LRModel.class);
            lr = new LogisticRegression(lrm.getNumFeature(), lrm.getNumClass(), lrm.getAlpha(), lrm.getRegL1());
            lr.setModel(lrm);
        } catch (IOException ex) {
            Logger.getLogger(Analytics.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println("\nDESERIALIZED & PARSED:\n" + lr.getModel().toString());
    }

    
    /**
     * Initialize NN model with trained weights from Keras 
     */
    public void initNNModel() {
        // read keras model
        long start = System.currentTimeMillis();
        try {
            modelNN = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelFilename);
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException ex) {
            Logger.getLogger(Analytics.class.getName()).log(Level.SEVERE, null, ex);
        }
        long stop = System.currentTimeMillis();
        Logger.getLogger("Time to read Keras Model File: " + (stop - start) + " (ms)");
        modelNN.printConfiguration();
        // initialize DL4J input INDarray
    }

    /**
     *  Predict classification on record with Logistic Regression
     * @param c
     * @return
     */
    public int predictLR(Continuous c) {
        test.set(0, c);
        int[] p = lr.predict(test);
        return p[0];
    }

    /**
     * Predict classification on record with NN model
     * @param c
     * @return
     */
    public int predictNN(Continuous c) {
        int numFeature = c.getFeature().length;
        indArray = Nd4j.zeros(numFeature);
        for (int j = 0; j < numFeature; j++) {
            indArray.putScalar(new int[]{j}, c.getFeature()[j]);
        }
        return (int) Math.rint(modelNN.output(indArray).getDouble(0));
    }
}
