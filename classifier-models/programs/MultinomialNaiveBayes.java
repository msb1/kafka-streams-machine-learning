package com.barnwaldo.classifiers.programs;

import java.util.ArrayList;
import java.util.List;

import com.barnwaldo.classifiers.data.Category;
import com.barnwaldo.classifiers.model.MultiNBModel;
import com.barnwaldo.classifiers.utils.TrainTestData;

import lombok.Getter;
import lombok.Setter;

/**
 * Multinomial Naive Bayes
 *
 * (1) Train/Test/Predict data must be transferred to Category (Data) objects
 * where features are tracked as integer levels 0 thru numLevel - 1
 *
 * (2) fitModel is used to calculate prior probabilities, likelihoods from
 * training data set
 *
 * (3) predict is used to determine class based on input features (only)
 *
 * (4) model can be saved by using getModel().toString() which provides a JSON
 * string with all model parameters
 *
 * (5) model can be used rather than training by MultiNBModel model =
 * mapper.readValue(jsonModelText, MultiNBModel.class);
 *
 * @author barnwaldo
 *
 */
@Getter
@Setter
public class MultinomialNaiveBayes {

    private int numClass;
    private int numFeature;
    private MultiNBModel model;
    private double[] posteriors;
    private List<List<Category>> classData;

    public MultinomialNaiveBayes(int numFeature, int numClass, int[] numLevel) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.model = new MultiNBModel(numFeature, numClass, numLevel);
        posteriors = new double[numClass];
        classData = new ArrayList<>();
    }

    /**
     * Method fits Multinomial Naive Bayes to training data...
     *
     * Fit is determined by finding frequencies (probabilities)
     *
     * @param data
     */
    public void fitModel(List<Category> data) {
        // get data by classes
        for (int classId = 0; classId < numClass; classId++) {
            classData.add(TrainTestData.getCategoryByClass(data, classId));
        }
        // calculate priors
        int total = data.size();
        System.out.println("Train data samples: " + total);
        for (int classId = 0; classId < numClass; classId++) {
            List<Category> tempData = classData.get(classId);
            model.getPriors()[classId] = (double) tempData.size() / total;
            calculateFreqs(tempData, classId);
            System.out.println("ClassId: " + classId + ", Class Sample Size: " + tempData.size()
                    + ", Prior = " + model.getPriors()[classId]);
//			for (int i = 0; i < numFeature; i++) {
//				System.out.println(model.getHeaders()[i] + ": ");
//				for(int j = 0; j < model.getNumLevel()[i]; j++) {
//					System.out.print("   " + model.getFrequency()[i][classId][j]);
//				}
//			}
        }
    }

    /**
     * Calculate likelihoods from training data
     *
     * @param cat
     * @param classId
     */
    private void calculateFreqs(List<Category> cat, int classId) {
        int numSamples = cat.size();
        for (int i = 0; i < numFeature; i++) {
            int[] ctr = new int[model.getNumLevel()[i]];
            for (Category c : cat) {
                int feature = c.getFeature()[i];
                if (feature < model.getNumLevel()[i]) {
                    ctr[feature] += 1;
                }
            }
            // calculate frequency with Laplace smoothing
            for (int j = 0; j < model.getNumLevel()[i]; j++) {
                model.getFrequency()[i][classId][j] = (double) (ctr[j] + 1) / (numSamples + numFeature);
            }
        }
    }

    /**
     * Method gets predictions from features based on model fit
     *
     * @param data
     * @return
     */
    public int[] predict(List<Category> data) {
        int numSamples = data.size();
        int[] predictions = new int[numSamples];
        for (int n = 0; n < numSamples; n++) {
            int[] feature = data.get(n).getFeature();
            for (int classId = 0; classId < numClass; classId++) {
                double likelihood = 1.0;
                for (int index = 0; index < numFeature; index++) {
                    likelihood *= model.getFrequency()[index][classId][feature[index]];
                }
                posteriors[classId] = model.getPriors()[classId] * likelihood;
            }
            int maxClassId = 0;
            for (int classId = 1; classId < numClass; classId++) {
                if (posteriors[classId] > posteriors[maxClassId]) {
                    maxClassId = classId;
                }
            }
            predictions[n] = maxClassId;
        }
        return predictions;
    }

}
