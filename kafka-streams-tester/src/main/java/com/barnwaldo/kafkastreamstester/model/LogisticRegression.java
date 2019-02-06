package com.barnwaldo.kafkastreamstester.model;

import java.util.List;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Logistic Regression
 *
 * (1) Train/Test/Predict data must be transferred to Continuous (Data) objects
 *
 * (2) fitModel is used to calculate weights from cost function using batch
 * gradient descent and Softmax function for probabilities
 *
 * (3) predict is used to determine class based on input features (only) using
 * weights and Softmax
 *
 * (4) model can be saved by using getModel().toString() which provides a JSON
 * string with all model parameters
 *
 * (5) model can be used rather than training by LRModel model =
 * mapper.readValue(jsonModelText, LRModel.class);
 *
 * @author barnwaldo
 *
 */
@Getter
@Setter
@NoArgsConstructor
public class LogisticRegression {

    private int numFeature;
    private int numClass;
    private int numEpoch;
    private double alpha;
    private double regL1;
    private double[] loss;
    private LRModel model;
    private double[][] grad;

    public LogisticRegression(int numFeature, int numClass, double alpha, double regL1) {
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.alpha = alpha;
        this.regL1 = regL1;
        model = new LRModel(numFeature, numClass, alpha, regL1);
        grad = new double[numFeature][numClass];
    }

    /**
     * Softmax probability helper function uses input vector x, model weights w
     * and returns p for input class
     *
     * @param x
     * @param classId
     * @return
     */
    private double softMax(double[] x, int classId) {
        double denominator = 0.0;
        double[] p = new double[numClass];
        for (int id = 0; id < numClass; id++) {
            double exponent = 0.0;
            for (int feature = 0; feature < numFeature; feature++) {
                exponent += x[feature] * model.getW()[feature][id];
            }
            p[id] = Math.exp(exponent);
            denominator += p[id];
        }
        return p[classId] / denominator;
    }

    /**
     * Fits model to data - weights numEpochs must be called prior to using
     * fitModel for LR
     *
     * @param data
     */
    public void fitModel(List<Continuous> data) {
        int numSamples = data.size();
        loss = new double[numEpoch];
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            // zero gradient column matrix
            for (double[] g : grad) {
                for (int i = 0; i < g.length; i++) {
                    g[i] = 0;
                }
            }
            double currentLoss = 0.0;
            // loop over each training data entry - add to grad and loss for each entry
            for (Continuous d : data) {
                // cycle through classes (columns)
                for (int clid = 0; clid < numClass; clid++) {
                    double[] x = d.getFeature();
                    double prob = softMax(x, clid);
                    for (int n = 0; n < numFeature; n++) {
                        if (clid == d.getResult()) {
                            grad[n][clid] += x[n] * (1 - prob);
                            currentLoss += Math.log(prob);
                        } else {
                            grad[n][clid] -= x[n] * (prob);
                        }
                    }
                }
            }
            // scale gradient by number of samples (batch) and update weights
            double regLossSum = 0.0;
            for (int j = 0; j < numClass; j++) {
                for (int i = 0; i < numFeature; i++) {
                    regLossSum += model.getW()[i][j] * model.getW()[i][j];
                    grad[i][j] = grad[i][j] / numSamples + regL1 * model.getW()[i][j];
                    model.getW()[i][j] += alpha * grad[i][j];
                }
            }
            // update loss function
            loss[epoch] = regL1 * regLossSum / 2.0 - currentLoss / numSamples;
            // System.out.println("Epoch: " + epoch + ", loss = " + loss[epoch]);
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
            Continuous d = data.get(n);
            double[] prob = new double[numClass];
            for (int clid = 0; clid < numClass; clid++) {
                prob[clid] = softMax(d.getFeature(), clid);
            }
            double maxProb = prob[0];
            predictions[n] = 0;
            for (int i = 1; i < numClass; i++) {
                if (prob[i] > maxProb) {
                    maxProb = prob[i];
                    predictions[n] = i;
                }
            }
        }
        return predictions;
    }

}
