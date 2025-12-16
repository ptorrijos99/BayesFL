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
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
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
     * Unified data partitioning method.
     * <p>
     * This method ensures a fair comparison between IID and Non-IID settings by always
     * using a <b>Global Test Set</b> strategy.
     * </p>
     * <ul>
     * <li><b>Step 1:</b> The dataset is split into K folds (Cross-Validation).</li>
     * <li><b>Step 2:</b> For each fold, the Global Test set is reserved.</li>
     * <li><b>Step 3:</b> The Global Training set is partitioned among clients:
     * <ul>
     * <li>If <code>alpha > 0</code>: Uses <b>Dirichlet</b> distribution (Non-IID).</li>
     * <li>If <code>alpha <= 0</code>: Uses <b>Stratified</b> splitting (IID).</li>
     * </ul>
     * </li>
     * </ul>
     *
     * @param name The name of the dataset.
     * @param path The path of the dataset.
     * @param nFolds The number of folds.
     * @param nClients The number of clients.
     * @param seed The seed.
     * @param alpha The Dirichlet concentration parameter. If <= 0, standard IID partitioning is used.
     * @return The divided data in format [Fold][Client][0=Train, 1=Test].
     */
    public static Instances[][][] divide(String name, String path, int nFolds, int nClients, int seed, double alpha) {
        Data data = new Weka_Instances(name, path);
        Instances instances = (Instances) data.getData();
        Random random = new Random(seed);

        // Structure: [Fold][Client][0=Train, 1=Test]
        Instances[][][] splits = new Instances[nFolds][nClients][2];

        // 1. Global Stratified Shuffle for Cross-Validation
        // This ensures the Test Set represents the global distribution.
        instances.randomize(random);
        instances.stratify(nFolds);

        for (int fold = 0; fold < nFolds; fold++) {
            // A. Extract Global Train and Global Test for this fold
            Instances globalTrain = instances.trainCV(nFolds, fold, random);
            Instances globalTest = instances.testCV(nFolds, fold);

            // B. Partition the Global Train among clients
            // We use a specific seed per fold to ensure variation between folds but reproducibility
            Random foldRand = new Random(seed + fold);
            List<Instances> clientPartitions;

            if (alpha > 0) {
                // Non-IID: Partition using Dirichlet Distribution
                clientPartitions = dirichletSplit(globalTrain, nClients, alpha, foldRand);
            } else {
                // IID: Partition using Stratified Split (Equal distribution)
                clientPartitions = stratifiedSplit(globalTrain, nClients, foldRand);
            }

            // C. Assign partitions to the data structure
            for (int c = 0; c < nClients; c++) {
                splits[fold][c][0] = clientPartitions.get(c); // Local Client Train

                // IMPORTANT: We assign the Global Test set to all clients.
                // This ensures we measure accuracy against the true global objective.
                splits[fold][c][1] = new Instances(globalTest);
            }
        }
        return splits;
    }

    /**
     * Unified data partitioning method with default alpha (-1.0, IID).
     *
     * @param name The name of the dataset.
     * @param path The path of the dataset.
     * @param nFolds The number of folds.
     * @param nClients The number of clients.
     * @param seed The seed.
     * @return The divided data in format [Fold][Client][0=Train, 1=Test].
     */
    public static Instances[][][] divide(String name, String path, int nFolds, int nClients, int seed) {
        return divide(name, path, nFolds, nClients, seed, -1.0);
    }

    /**
     * Standard IID Stratified split of a training set among N clients.
     * This ensures every client gets a representative subset of the training data.
     *
     * @param data The global training data to split.
     * @param nClients The number of clients.
     * @param rand Random number generator.
     * @return A list of Instances partitions (IID).
     */
    private static List<Instances> stratifiedSplit(Instances data, int nClients, Random rand) {
        Instances copy = new Instances(data);
        copy.randomize(rand);
        copy.stratify(nClients);

        List<Instances> partitions = new ArrayList<>();
        for (int c = 0; c < nClients; c++) {
            // We treat the "folds" logic of Weka as "client buckets" here
            partitions.add(copy.testCV(nClients, c));
        }
        return partitions;
    }

    /**
     * Core logic to split a dataset into N clients using Dirichlet distribution (Non-IID).
     * Includes "Robin Hood" logic to prevent empty clients.
     *
     * @param data The dataset to split.
     * @param nClients The number of clients.
     * @param alpha The heterogeneity parameter.
     * @param rand The random number generator.
     * @return A list of Instances, one for each client.
     */
    private static List<Instances> dirichletSplit(Instances data, int nClients, double alpha, Random rand) {
        int numClasses = data.numClasses();

        // 1. Organize instances by class
        List<List<Instance>> classInstances = new ArrayList<>(numClasses);
        for (int k = 0; k < numClasses; k++) {
            classInstances.add(new ArrayList<>());
        }
        for (Instance inst : data) {
            classInstances.get((int) inst.classValue()).add(inst);
        }

        // 2. Create empty buckets for clients
        List<Instances> clients = new ArrayList<>(nClients);
        for (int c = 0; c < nClients; c++) {
            clients.add(new Instances(data, 0));
        }

        // 3. Distribute each class according to Dirichlet proportions
        for (int k = 0; k < numClasses; k++) {
            List<Instance> clazz = classInstances.get(k);
            if (clazz.isEmpty()) continue;

            Collections.shuffle(clazz, rand);

            // Sample proportions from Dirichlet via Gamma distribution
            double[] proportions = new double[nClients];
            double sum = 0.0;
            for (int c = 0; c < nClients; c++) {
                proportions[c] = sampleGamma(alpha, 1.0, rand);
                sum += proportions[c];
            }

            // Normalize and calculate counts
            int nClass = clazz.size();
            int[] counts = new int[nClients];
            int countSum = 0;

            for (int c = 0; c < nClients; c++) {
                if (sum == 0) proportions[c] = 1.0 / nClients;
                else proportions[c] /= sum;

                counts[c] = (int) (proportions[c] * nClass);
                countSum += counts[c];
            }

            // Distribute remainder (due to rounding)
            int diff = nClass - countSum;
            for (int i = 0; i < diff; i++) {
                counts[rand.nextInt(nClients)]++;
            }

            // Assign instances to clients
            int start = 0;
            for (int c = 0; c < nClients; c++) {
                for (int j = 0; j < counts[c]; j++) {
                    if (start < nClass) {
                        clients.get(c).add(clazz.get(start++));
                    }
                }
            }
        }

        // --- STEP 4: FIX EMPTY CLIENTS ("Robin Hood" Strategy) ---
        // This prevents Weka crashes by ensuring every client has at least 1 instance.
        for (int c = 0; c < nClients; c++) {
            if (clients.get(c).isEmpty()) {
                // Find the richest client (donor)
                int donorIndex = -1;
                int maxInst = 0;

                for (int d = 0; d < nClients; d++) {
                    if (clients.get(d).numInstances() > maxInst) {
                        maxInst = clients.get(d).numInstances();
                        donorIndex = d;
                    }
                }

                // If a valid donor exists with at least 2 instances (so we don't empty them)
                if (donorIndex != -1 && maxInst > 1) {
                    Instances donor = clients.get(donorIndex);

                    // Steal the last instance from the donor
                    Instance stolen = donor.instance(donor.numInstances() - 1);
                    donor.delete(donor.numInstances() - 1);

                    // Give it to the empty client
                    clients.get(c).add(stolen);
                } else {
                    // Critical failure case: Dataset is smaller than number of clients
                    System.err.println("CRITICAL WARNING: Client " + c + " remains empty. " +
                            "The dataset is too small or partitioned too sparsely (Max instances in a client: " + maxInst + "). " +
                            "This may cause a crash in local training.");
                }
            }
        }

        return clients;
    }

    /**
     * Generates a sample from Gamma(k, theta) using Marsaglia and Tsang's method.
     * Necessary to simulate Dirichlet distribution in Java.
     *
     * @param k The shape parameter.
     * @param theta The scale parameter.
     * @param rand The random number generator.
     * @return A random sample from the Gamma distribution.
     */
    private static double sampleGamma(double k, double theta, Random rand) {
        boolean accept = false;
        if (k < 1) {
            // Weibull algorithm for small k
            return sampleGamma(1 + k, theta, rand) * Math.pow(rand.nextDouble(), 1.0 / k);
        }

        double d = k - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        double v = 0;
        double u = 0;
        double x = 0;

        while (!accept) {
            do {
                x = rand.nextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            u = rand.nextDouble();

            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
                accept = true;
            } else if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
                accept = true;
            }
        }
        return d * v * theta;
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
