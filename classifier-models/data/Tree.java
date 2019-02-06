package com.barnwaldo.classifiers.data;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Tree {
	private int id;					// tree id 
	private boolean terminal;			// True = terminal or leaf node
	private int depth;				// depth of this node
	private int leftTreeId;				// left child node
	private int rightTreeId;			// right child node
	private int classifier;				// class if current node is a leaf (or terminal) node
	private int rowSplit;				// row split index
	private int colSplit;				// column split index
	private double splitValue;			// value at split
	private Continuous[] data;
	
	public Tree(int id) {
		this.id = id;
	}
}
