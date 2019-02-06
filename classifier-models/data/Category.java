package com.barnwaldo.classifiers.data;

import lombok.*;

/**
 * Suggested usage for category data: (1) read in data (2) translate categories
 * into levels beginning with 0 thru numLevels - 1
 * 
 * @author barnwaldo
 *
 */
@Getter
@Setter
@NoArgsConstructor
public class Category {

	private int[] feature;
	private int result;

}
