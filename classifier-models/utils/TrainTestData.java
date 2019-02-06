package com.barnwaldo.classifiers.utils;

import com.barnwaldo.classifiers.data.Category;
import com.barnwaldo.classifiers.data.Continuous;
import com.opencsv.CSVReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class TrainTestData {

    /**
     * Method to read data from CSV (text) file
     *
     * @param filename
     * @return
     */
    public static List<String[]> readFromCsvFile(String filename) {
        // read in data file with openCSV
        List<String[]> rows;
        try (CSVReader reader = new CSVReader(new FileReader(filename))){
            rows = reader.readAll();
        } catch (IOException e) {
            rows = new ArrayList<>();
            System.out.println(e.getMessage());
        }
        return rows;
    }

    /**
     * Method to split training data into train/test data for model training and
     * validation
     *
     * @param train
     * @param trainSplitFraction
     * @return
     */
    public static List<String[]> splitStringData(List<String[]> train, double trainSplitFraction) {
        int trainSize = train.size();
        int trainSplitSize = (int) (trainSplitFraction * trainSize);
        List<String[]> test = new ArrayList<>();
        Random random = new Random();
        while (trainSize > trainSplitSize) {
            int randomIndex = random.nextInt(trainSize);
            test.add(train.get(randomIndex));
            train.remove(randomIndex);
            trainSize = train.size();
        }
        return test;
    }

    /**
     * Method to split training data into train/test data for model training and
     * validation
     *
     * @param train
     * @param trainSplitFraction
     * @return
     */
    public static List<Continuous> splitContinuousData(List<Continuous> train, double trainSplitFraction) {
        int trainSize = train.size();
        int trainSplitSize = (int) (trainSplitFraction * trainSize);
        List<Continuous> test = new ArrayList<>();
        Random random = new Random();
        while (trainSize > trainSplitSize) {
            int randomIndex = random.nextInt(trainSize);
            test.add(train.get(randomIndex));
            train.remove(randomIndex);
            trainSize = train.size();
        }
        return test;
    }

    /**
     * Method to split training data into train/test data for model training and
     * validation
     *
     * @param train
     * @param trainSplitFraction
     * @return
     */
    public static List<Category> splitCategoryData(List<Category> train, double trainSplitFraction) {
        int trainSize = train.size();
        int trainSplitSize = (int) (trainSplitFraction * trainSize);
        List<Category> test = new ArrayList<>();
        Random random = new Random();
        while (trainSize > trainSplitSize) {
            int randomIndex = random.nextInt(trainSize);
            test.add(train.get(randomIndex));
            train.remove(randomIndex);
            trainSize = train.size();
        }
        return test;
    }

    /**
     * Method to sample training data for random forest model training
     *
     * @param train
     * @param sampleRate
     * @return
     */
    public static List<Continuous> sampleContinuousData(List<Continuous> train, double sampleRate) {
        int trainSize = train.size();
        int trainSplitSize = (int) ((1.0 - sampleRate) * trainSize);
        List<Continuous> test = new ArrayList<>();
        Random random = new Random();
        while (trainSize > trainSplitSize) {
            int randomIndex = random.nextInt(trainSize);
            test.add(train.get(randomIndex));
            train.remove(randomIndex);
            trainSize = train.size();
        }
        // add test data back to original sample (for random forest sampling)
        train.addAll(test);
        return test;
    }

    /**
     * Method to get data entries by ClassId ClassId assumed to be last item in
     * String data row
     *
     * @param train
     * @param classId
     * @return
     */
    public static List<Continuous> getContinuousByClass(List<Continuous> train, int classId) {
        List<Continuous> continuous = new ArrayList<>();
        train.stream().filter((c) -> (c.getResult() == classId)).forEachOrdered(continuous::add);
        return continuous;
    }

    /**
     * Method to get data entries by ClassId ClassId assumed to be last item in
     * String data row
     *
     * @param train
     * @param classId
     * @return
     */
    public static List<Category> getCategoryByClass(List<Category> train, int classId) {
        List<Category> category = new ArrayList<>();
        train.stream().filter((c) -> (c.getResult() == classId)).forEachOrdered(category::add);
        return category;
    }

    /**
     * Method calculates accuracy
     *
     * @param predict
     * @param result
     * @return
     */
    public static double accuracy(int[] predict, int[] result) {
        int len = predict.length;
        double success = 0.0;
        for (int i = 0; i < len; i++) {
            if (result[i] == predict[i]) {
                success += 1.0;
            }
        }
        return 100.0 * success / len;
    }

    /**
     * Method to normalize continuous data
     *
     * @param data
     */
    public static void normalizeContinuous(List<Continuous> data) {
        // normalize data with max and min vals
        int numAttr = data.get(0).getFeature().length;
        int numData = data.size();
        double[] minVals = new double[numAttr];
        double[] maxVals = new double[numAttr];
        Arrays.fill(minVals, 1.0e010);
        Arrays.fill(maxVals, -1.0e10);

        for (Continuous datum : data) {
            double[] attr = datum.getFeature();
            for (int i = 0; i < numAttr; i++) {
                if (attr[i] < minVals[i]) {
                    minVals[i] = attr[i];
                }
                if (attr[i] > maxVals[i]) {
                    maxVals[i] = attr[i];
                }
            }
        }

        System.out.println("--> Mins: " + Arrays.toString(minVals));
        System.out.println("--> Maxs: " + Arrays.toString(maxVals));

        // perform normalization of data
        IntStream.range(0, numData).forEachOrdered(index -> data.get(index).normalize(minVals, maxVals));
    }

    /**
     * Helper method to copy continuous data lists to arrays for random forest
     * model
     *
     * @param list
     * @return
     */
    public static Continuous[] copyListToArray(List<Continuous> list) {
        Continuous[] array = new Continuous[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

}
