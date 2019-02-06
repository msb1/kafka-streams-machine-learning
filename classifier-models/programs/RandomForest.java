package com.barnwaldo.classifiers.programs;

import java.util.ArrayList;
import java.util.List;

import com.barnwaldo.classifiers.data.Continuous;
import com.barnwaldo.classifiers.data.Tree;
import com.barnwaldo.classifiers.model.RFModel;
import com.barnwaldo.classifiers.utils.TrainTestData;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

/**
 * Random Forest
 *
 * (1) Train/Test/Predict data must be transferred to Continuous (Data) objects
 *
 * (2) fitModel is used to calculate forest of trees where splits and left/right
 * trees are saved for predictions
 *
 * (3) predict is used to determine class based on forest of trees
 *
 * (4) model can be saved by using getModel().toString() which provides a JSON
 * string with all model parameters
 *
 * (5) model can be used rather than training by RFModel model =
 * mapper.readValue(jsonModelText, RFModel.class);
 *
 * @author barnwaldo
 *
 */
@Getter
@Setter
public class RandomForest {

    private int numTree; // sklearn default = 100
    private int maxDepth; // sklearn default = None (here use -1)
    private int minSize; // sklearn default = 1 (min samples leaf)
    private int numFeature;
    private int numClass;
    private double sampleRate; // suggest default to 0.8 or 0.9
    private RFModel model;
    @Getter(AccessLevel.NONE)
    @Setter(AccessLevel.NONE)
    private final List<Tree> treeList;

    public RandomForest(int numFeature, int numClass, int numTree) {
        this.numTree = numTree;
        this.numFeature = numFeature;
        this.numClass = numClass;
        this.model = new RFModel(numFeature, numClass, numTree);
        treeList = new ArrayList<>();
    }

    public void fitModel(List<Continuous> data) {

        for (int treeIndex = 0; treeIndex < numTree; treeIndex++) {
            // get random shuffled split from training data
            List<Continuous> trainData = TrainTestData.sampleContinuousData(data, sampleRate);
            // System.out.println("-----> Starting new tree -- index: " + treeIndex + " DataSize: " + trainData.size());
            Tree root = new Tree(treeList.size());
            root.setDepth(1);
            root.setData(TrainTestData.copyListToArray(trainData));
            model.getRootTreeId()[treeIndex] = root.getId();
            treeList.add(root);
            buildTree(root);
        }
        // save treeList to model
        Tree[] trees = new Tree[treeList.size()];
        for (int i = 0; i < treeList.size(); i++) {
            Tree tree = treeList.get(i);
            tree.setData(null);
            trees[i] = tree;
        }
        model.setTrees(trees);
    }

    /**
     * Method build tree starting from root (where data sample is passed with
     * root tree)
     *
     * @param root
     */
    private void buildTree(Tree root) {
        // System.out.println("Enter buildTree...");
        List<Integer> subTrees = new ArrayList<>();
        subTrees.add(root.getId());

        // loop through all subtrees
        while (subTrees.size() > 0) {
            int treeId = subTrees.remove(0);
            Tree tree = treeList.get(treeId);
            getSplit(tree);
            // split data into right/left groups based on best split
            List<Continuous> leftList = new ArrayList<>();
            List<Continuous> rightList = new ArrayList<>();
            Continuous[] data = tree.getData();
            for (Continuous data1 : data) {
                if (data1.getFeature()[tree.getColSplit()] < tree.getSplitValue()) {
                    leftList.add(data1);
                } else {
                    rightList.add(data1);
                }
            }
            Continuous[] leftData = TrainTestData.copyListToArray(leftList);
            Continuous[] rightData = TrainTestData.copyListToArray(rightList);
            int leftSize = leftData.length;
            int rightSize = rightData.length;

            // System.out.println("SplitPoint: " + tree.getRowSplit() + " " + tree.getColSplit() + "  treeId: "
            //		+ tree.getId() + "   'depth: " + tree.getDepth());
            // System.out.println("Left Tree Size: " + leftSize + "  Right Tree Size: " + rightSize);
            tree.setTerminal(false);
            // add new left and trees
            Tree leftTree = new Tree(treeList.size());
            leftTree.setDepth(tree.getDepth() + 1);
            treeList.add(leftTree);
            tree.setLeftTreeId(leftTree.getId());

            Tree rightTree = new Tree(treeList.size());
            rightTree.setDepth(tree.getDepth() + 1);
            treeList.add(rightTree);
            tree.setRightTreeId(rightTree.getId());

            if (leftSize == 0 || rightSize == 0) {
                int classifier = getClassWithMostSamples(tree);
                leftTree.setData(tree.getData());
                rightTree.setData(tree.getData());
                leftTree.setClassifier(classifier);
                rightTree.setClassifier(classifier);
                leftTree.setTerminal(true);
                rightTree.setTerminal(true);
                // System.out.println("--->> Terminal node reached... Group size criterion...");
                continue;
            }

            leftTree.setData(leftData);
            rightTree.setData(rightData);
            if (tree.getDepth() > maxDepth) {
                int leftClassifier = getClassWithMostSamples(leftTree);
                leftTree.setClassifier(leftClassifier);
                leftTree.setTerminal(true);
                int rightClassifier = getClassWithMostSamples(rightTree);
                rightTree.setClassifier(rightClassifier);
                rightTree.setTerminal(true);
                // System.out.println("--->> Terminal node reached... Max depth criterion...");
                continue;
            }

            if (leftSize < minSize) {
                int leftClassifier = getClassWithMostSamples(leftTree);
                leftTree.setClassifier(leftClassifier);
                leftTree.setTerminal(true);
                // System.out.println("--->> Left Terminal node reached... Min size criterion...");
            } else {
                subTrees.add(leftTree.getId());
                // System.out.println("Add left child subtree...");
            }
            if (rightSize < minSize) {
                int rightClassifier = getClassWithMostSamples(rightTree);
                rightTree.setClassifier(rightClassifier);
                rightTree.setTerminal(true);
                // System.out.println("--->> Right Terminal node reached... Min size criterion...");
            } else {
                subTrees.add(rightTree.getId());
                // System.out.println("Add right child subtree...");
            }
        }
    }

    /**
     * Helper method finds optimal tree split based on lowest GINI index
     *
     * @param tree
     */
    private void getSplit(Tree tree) {
        double gini = 1.0;
        Continuous[] data = tree.getData();
        List<Continuous> leftData = new ArrayList<>();
        List<Continuous> rightData = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < numFeature; j++) {
                double splitValue = data[i].getFeature()[j];
                // split data into left - right for test
                leftData.clear();
                rightData.clear();
                for (Continuous data1 : data) {
                    if (data1.getFeature()[j] < splitValue) {
                        leftData.add(data1);
                    } else {
                        rightData.add(data1);
                    }
                }
                Continuous[] left = TrainTestData.copyListToArray(leftData);
                Continuous[] right = TrainTestData.copyListToArray(rightData);
                // determine GINI index for test split
                double testGini = giniIndex(left, right);
                // save split to node if testGini < nodeGini
                if (testGini < gini) {
                    gini = testGini;
                    tree.setRowSplit(i);
                    tree.setColSplit(j);
                    tree.setSplitValue(splitValue);
                }
            }
        }
    }

    /**
     * Helper method calculates GINI index for a left/right data split at
     * current tree level
     *
     * @param left
     * @param right
     * @return
     */
    private double giniIndex(Continuous[] left, Continuous[] right) {
        int numLeft = left.length;
        int numRight = right.length;
        int numTotal = numRight + numLeft;
        int[] leftClassCtr = new int[numClass];
        int[] rightClassCtr = new int[numClass];

        for (Continuous left1 : left) {
            leftClassCtr[left1.getResult()] += 1;
        }
        for (Continuous right1 : right) {
            rightClassCtr[right1.getResult()] += 1;
        }
        double leftTerm = 1.0;
        if (numLeft != 0) {
            for (int i = 0; i < numClass; i++) {
                leftTerm -= (leftClassCtr[i] / numLeft) * (leftClassCtr[i] / numLeft);
            }
        }
        double rightTerm = 1.0;
        if (numRight != 0) {
            for (int i = 0; i < numClass; i++) {
                rightTerm -= (rightClassCtr[i] / numRight) * (rightClassCtr[i] / numRight);
            }
        }
        // return GINI index for split
        return (numLeft * leftTerm + numRight * rightTerm) / numTotal;
    }

    /**
     * Helper method to get classId for class with most samples at terminal node
     * of tree
     *
     * @param tree
     * @return
     */
    private int getClassWithMostSamples(Tree tree) {
        Continuous[] data = tree.getData();
        int[] ctr = new int[numClass];
        for (Continuous c : data) {
            if (c.getResult() < numClass) {
                ctr[c.getResult()]++;
            }
        }
        int maxCtrIndex = 0;
        for (int i = 1; i < numClass; i++) {
            if (ctr[i] > ctr[maxCtrIndex]) {
                maxCtrIndex = i;
            }
        }
        return maxCtrIndex;
    }

    /**
     * Method gets predictions from features based on model fit
     *
     * @param data
     * @return
     */
    public int[] predict(List<Continuous> data) {
        // System.out.println("--- Predictions ---");
        int numSamples = data.size();
        int[] predictions = new int[numSamples];
        for (int n = 0; n < numSamples; n++) {
            Continuous c = data.get(n);
            int[] ctr = new int[numClass];
            for (int i = 0; i < numTree; i++) {
                int rootTreeId = model.getRootTreeId()[i];
                int classId = predictClass(model.getTrees()[rootTreeId], c);
                ctr[classId]++;
            }
            int maxCtrIndex = 0;
            for (int i = 1; i < numClass; i++) {
                if (ctr[i] > ctr[maxCtrIndex]) {
                    maxCtrIndex = i;
                }
            }
            predictions[n] = maxCtrIndex;
            // System.out.println("Sample " + n + "   classId: " + maxCtrIndex);
        }
        return predictions;
    }

    /**
     * Helper method to get predicted classId for a given data entry and tree
     *
     * @param root
     * @param row
     * @return
     */
    private int predictClass(Tree root, Continuous row) {
        List<Integer> subTrees = new ArrayList<>();
        subTrees.add(root.getId());
        Tree tree = root;
        // loop through all subtrees
        while (subTrees.size() > 0) {
            int treeId = subTrees.remove(0);
            tree = model.getTrees()[treeId];
            if (tree.isTerminal()) {
                break;
            }
            if (row.getFeature()[tree.getColSplit()] < tree.getSplitValue()) {
                subTrees.add(tree.getLeftTreeId());
            } else {
                subTrees.add(tree.getRightTreeId());
            }
        }
        return tree.getClassifier();
    }

}
