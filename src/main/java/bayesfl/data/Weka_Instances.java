/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2024 Universidad de Castilla-La Mancha, España
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

/**
 * Package containing datasets related with federated Bayesian networks.
 */
package bayesfl.data;

/**
 * Third-party imports.
 */
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import java.util.Random;

/**
 * A class representing a dataset in Weka format.
 */
public class Weka_Instances implements Data {

    /**
     * The name of the dataset.
     */
    private String name;

    /**
     * The training data.
     */
    private Instances train;

    /**
     * The testing data.
     */
    private Instances test;

    /**
     * Constructs a dataset.
     *
     * @param name The name of the dataset.
     * @param path The path to the dataset.
     */
    public Weka_Instances(String name, String path) {
        this.name = name;

        try {
            // Assume the last attribute is the class variable
            DataSource source = new DataSource(path);
            this.train = source.getDataSet();
            this.train.setClassIndex(this.train.numAttributes() - 1);
        }

        catch (Exception exception) {
            exception.printStackTrace();
        }
    }

    /**
     * Constructs a dataset.
     *
     * @param name The name of the dataset.
     * @param train The training data.
     * @param test The testing data.
     */
    public Weka_Instances(String name, Instances train, Instances test) {
        this.name = name;
        this.train = train;
        this.test = test;
    }

    /**
     * Divide the data for the experiment for the given number of folds and clients.
     *
     * @param name The name of the dataset.
     * @param path The path of the dataset.
     * @param nFolds The number of folds.
     * @param nClients The number of clients.
     * @param seed The seed.
     * @return The divided data.
     */
    public static Instances[][][] divide(String name, String path, int nFolds, int nClients, int seed) {
        Data data = new Weka_Instances(name, path);
        Instances instances = (Instances) data.getData();

        // Stratify the data for the clients
        Random random = new Random(seed);
        instances.randomize(random);
        instances.stratify(nClients);

        Instances[][][] splits = new Instances[nFolds][nClients][2];

        for (int i = 0; i < nClients; i++) {
            // Get the data for the client, which corresponds
            // to the testing data for the first level split
            Instances all = instances.testCV(nClients, i);

            // Stratify the data for the folds
            random = new Random(seed + i);
            all.randomize(random);
            all.stratify(nFolds);

            for (int j = 0; j < nFolds; j++) {
                // Get the training and testing data for the fold
                Instances train = all.trainCV(nFolds, j, random);
                Instances test = all.testCV(nFolds, j);
                splits[j][i][0] = train;
                splits[j][i][1] = test;
            }
        }

        return splits;
    }

    /**
     * Retrieves the data.
     * 
     * @return The data.
    */
    @Override
    public Object getData() {
        return this.train;
    }

    /**
     * Sets the data.
     * 
     * @param data The data.
     */
    @Override
    public void setData(Object data) {
        if (!(data instanceof Instances)) {
            throw new IllegalArgumentException("The data must be object of the Instances class");
        }

        this.train = (Instances) data;
    }

    /**
     * Retrieves the training data.
     * 
     * @return The training data.
     */
    public Instances getTrain() {
        return this.train;
    }

    /**
     * Retrieves the testing data.
     * 
     * @return The testing data.
     */
    public Instances getTest() {
        return this.test;
    }

    /**
     * Retrieves the name of the dataset.
     * 
     * @return The name of the dataset.
     */
    @Override
    public String getName() {
        return this.name;
    }

    /**
     * Retrieves the number of instances in the dataset.
     * 
     * @return The number of instances in the dataset.
     */
    @Override
    public int getNInstances() {
        return this.train.numInstances();
    }
}
