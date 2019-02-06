package com.barnwaldo.classifiers.programs;

import java.util.ArrayList;
import java.util.List;

import com.barnwaldo.classifiers.data.Continuous;
import com.barnwaldo.classifiers.model.GaussNBModel;
import com.barnwaldo.classifiers.utils.TrainTestData;

import lombok.Getter;
import lombok.Setter;

/**
 * Gaussian Naive Bayes
 *
 * (1) Train/Test/Predict data must be transferred to Continuous (Data) objects
 *
 * (2) fitModel is used to calculate prior probabilities, means, stdDevs from
 * training data set
 *
 * (3) predict is used to determine class based on input features (only)
 *
 * (4) model can be saved by using getModel().toString() which provides a JSON
 * string with all model parameters
 *
 * (5) model can be used rather than training by GaussNBModel model =
 * mapper.readValue(jsonModelText, GaussNBModel.class);
 *
 * @author barnwaldo
 *
 */
@Getter
@Setter
@SuppressWarnings({"Duplicates", "JavaDoc"})
public class GaussianNaiveBayes {

    private int numClass;
    private int numFeature;
    private GaussNBModel model;
    private double[] posteriors;
    private List<List<Continuous>> classData;

    public GaussianNaiveBayes(int numFeature, int numClass) {
        this.numClass = numClass;
        this.numFeature = numFeature;
        posteriors = new double[numClass];
        model = new GaussNBModel(numFeature, numClass);
        classData = new ArrayList<>();
    }

    /**
     * Method fits Gaussian Naive Bayes to training data... Fit is determined by
     * finding means, stdDevs per feature per class
     *
     * @param data
     */
    public void fitModel(List<Continuous> data) {

        // get data by classes
        for (int classId = 0; classId < numClass; classId++) {
            classData.add(TrainTestData.getContinuousByClass(data, classId));
        }

        // calculate priors
        int total = data.size();
        System.out.println("Train data samples: " + total);
        for (int classId = 0; classId < numClass; classId++) {
            List<Continuous> tempData = classData.get(classId);
            model.getPriors()[classId] = (double) tempData.size() / total;
            calculateStats(tempData, classId);
            System.out.println("ClassId: " + classId + ", Class Sample Size: "
                    + tempData.size() + ", Prior = " + model.getPriors()[classId]);
//			System.out.println("     Means: ");
//			for(int i = 0; i < numFeature; i++) {
//				System.out.print("   " + model.getMeans()[i][classId]);
//			}
//			System.out.println("     StdDevs: ");
//			for(int i = 0; i < numFeature; i++) {
//				System.out.print("   " + model.getStdDevs()[i][classId]);
//			}
        }
    }

    /**
     * Method calculates means and stdDevs from training data set
     *
     * @param con
     * @param id
     */
    private void calculateStats(List<Continuous> con, int id) {
        int numSamples = con.size();

        // calculate means
        con.forEach((c) -> {
            for (int i = 0; i < numFeature; i++) {
                model.getMeans()[i][id] += c.getFeature()[i];
            }
        });
        for (int i = 0; i < numFeature; i++) {
            model.getMeans()[i][id] = model.getMeans()[i][id] / numSamples;
        }

        // calculate std dev
        con.forEach((c) -> {
            for (int i = 0; i < numFeature; i++) {
                model.getStdDevs()[i][id] += (c.getFeature()[i] - model.getMeans()[i][id])
                        * (c.getFeature()[i] - model.getMeans()[i][id]);
            }
        });
        for (int i = 0; i < numFeature; i++) {
            model.getStdDevs()[i][id] = Math.sqrt(model.getStdDevs()[i][id] / (numSamples - 1));
        }
    }

    /**
     * Method gets predictions from features based on model fit
     *
     * @param data
     * @return
     */
    public int[] predict(List<Continuous> data) {
        int numSamples = data.size();
        int[] predictions = new int[numSamples];
        for (int n = 0; n < numSamples; n++) {
            double[] feature = data.get(n).getFeature();
            for (int id = 0; id < numClass; id++) {
                double[] probabilities = new double[numFeature];
                for (int i = 0; i < numFeature; i++) {
                    double mean = model.getMeans()[i][id];
                    double stdDev = model.getStdDevs()[i][id];
                    probabilities[i] = Math.exp((feature[i] - mean) * (feature[i] - mean) / (2.0 * stdDev * stdDev));
                    probabilities[i] = probabilities[i] / (stdDev * Math.sqrt(2.0 * Math.PI));
                }

                double likelihood = 1.0;
                for (double p : probabilities) {
                    likelihood *= p;
                }
                posteriors[id] = model.getPriors()[id] * likelihood;
            }
            int maxClassId = 0;
            for (int id = 1; id < numClass; id++) {
                if (posteriors[id] > posteriors[maxClassId]) {
                    maxClassId = id;
                }
            }
            predictions[n] = maxClassId;
        }
        return predictions;
    }

}
