package com.barnwaldo.classifiers.programs;

import java.util.Arrays;
import java.util.List;

import com.barnwaldo.classifiers.data.Continuous;
import com.barnwaldo.classifiers.model.SVCModel;

import libsvm.svm;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import lombok.Getter;
import lombok.Setter;


/**
 * Support Vector Machine
 * 
 * (1) Train/Test/Predict data must be transferred to Continuous (Data) objects and normalized [-1, 1] by calling normalize on
 * each Continuous data object
 * 
 * (2) fitModel is used to determine libSVM model from training data
 * 
 * (3) predict is used to determine class based on input features (only) and saved libSVM model
 * 
 * (4) model can be saved by using getModel().toString() which provides a JSON string with all model parameters
 * 
 * (5) model can be used rather than training by SVCModel model = mapper.readValue(jsonModelText, SVCModel.class);
 * 
 * @author barnwaldo
 *
 */
@Getter
@Setter
public class SupportVector {
	private int numFeature;
	private int numClass;
	private int cv;
	private String errorMessage;
	private svm_problem svmProblem;
	private SVCModel model;

	public SupportVector(int numFeature, int numClass) {
		this.numFeature = numFeature;
		this.numClass = numClass;
		this.svmProblem = new svm_problem();
		this.model = new SVCModel(numFeature, numClass);
	}

	/**
	 * Method fits SVM Classifier to training data
	 * 
	 * @param data
	 */
	public void fitModel(List<Continuous> data) {
		int numSample = data.size();
		svmProblem.l = numSample;
		svmProblem.x = new svm_node[numSample][numFeature];
		svmProblem.y = new double[numSample];
		// convert Continuous class data to svm_problem (and node) class data
		for (int i = 0; i < numSample; i++) {
			Continuous d = data.get(i);
			double[] attr = d.getFeature();
			svm_node s = new svm_node();
			for (int j = 0; j < numFeature; j++) {
				s.index = j;
				s.value = attr[j];
				svmProblem.x[i][j] = s;
			}
			svmProblem.y[i] = (double) d.getResult();
		}

		errorMessage = svm.svm_check_parameter(svmProblem, model.getSvmParameter());

		if (errorMessage != null) {
			System.err.print("ERROR: " + errorMessage + "\n");
			return;
		}

		if (cv >= 2) {
			crossValidation();
		} else {
			model.setSvmModel(svm.svm_train(svmProblem, model.getSvmParameter()));
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
		int[] labels = new int[numClass];
		double[] probEstimates = new double[numClass];
		svm.svm_get_labels(model.getSvmModel(), labels);

		for (int n = 0; n < numSamples; n++) {
			Continuous d = data.get(n);
			double[] attr = d.getFeature();
			svm_node[] x = new svm_node[numFeature];
			for(int i = 0; i < numFeature; i++) {
				x[i] = new svm_node();
				x[i].index = i;
				x[i].value =  attr[i];
			}
			svm_parameter svmp = model.getSvmParameter();
			if (svmp.probability == 1 && (svmp.svm_type == svm_parameter.C_SVC
					|| svmp.svm_type == svm_parameter.NU_SVC)) {
				predictions[n] = (int)svm.svm_predict_probability(model.getSvmModel(), x, probEstimates);
				System.out.println("ClassId: " + predictions[n] + ", Probabilities: " + Arrays.toString(probEstimates));
			} else {
				predictions[n] = (int)svm.svm_predict(model.getSvmModel(), x);
			}
		}
		return predictions;
	}

	/**
	 * Helper method to perform cross validation if option is enabled -- cv > 2
	 */
	private void crossValidation() {
		int i;
		int total_correct = 0;
		double[] target = new double[svmProblem.l];

		// perform cross validation
		svm.svm_cross_validation(svmProblem, model.getSvmParameter(), cv, target);
		// regression is not implemented here - only classification
		for (i = 0; i < svmProblem.l; i++)
			if (target[i] == svmProblem.y[i])
				++total_correct;
		System.out.print("Cross Validation Accuracy = " + 100.0 * total_correct / svmProblem.l + "%\n");
	}
}
