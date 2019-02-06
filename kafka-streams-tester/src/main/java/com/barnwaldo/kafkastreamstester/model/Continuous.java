package com.barnwaldo.kafkastreamstester.model;

import lombok.*;

/**
 * Suggested usage for category data: (1) read in data (2) find max and min vals
 * for training set (3) use normalize method to scale data between [-1,1]
 * 
 * @author barnwaldo
 *
 */
@Getter
@Setter
@NoArgsConstructor
public class Continuous {

	private double[] feature;
	private int result;

	/**
	 * Method will normalize feature data in sample
	 * 
	 * @param minVals
	 * @param maxVals
	 */
	public void normalize(double[] minVals, double[] maxVals) {
		for (int i = 0; i < feature.length; i++) {
			feature[i] = 2.0 * (feature[i] - minVals[i]) / (maxVals[i] - minVals[i]) - 1.0;
		}
	}
}
