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
 * Package containing experiments for federated Bayesian networks.
 */
package bayesfl.experiments;

/**
 * Standard imports.
 */
import java.util.ArrayList;
import java.util.Random;

/**
 * Third-party imports.
 */
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.algorithms.Bins_MDLP;
import bayesfl.algorithms.CCBN;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Data;
import bayesfl.data.Weka;
import bayesfl.fusion.Bins_Fusion;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.FusionPosition;
import bayesfl.fusion.PT_Fusion_Client;
import bayesfl.fusion.PT_Fusion_Server;
import bayesfl.Client;
import bayesfl.Server;

/**
 * A class representing an experiment with class-conditional Bayesian networks.
 */
public class CCBNExperiment {

    /**
     * The base path for the datasets.
     */
    private static String baseDatasetPath = "res/classification/";

    /**
     * The base path for the results.
     */
    private static String baseOutputPath = "res/experiments/";

    /**
     * Run the experiment.
     *
     * @param datasetName The name of the dataset.
     * @param discretizerOptions The options for the discretizer.
     * @param algorithmOptions The options for the algorithm.
     * @param nClients The number of clients.
     * @param nIterations The number of iterations.
     * @param nFolds The number of folds.
     * @param seed The seed.
     * @param suffix The suffix for the output file.
     */
    public static void run(String datasetName, String[] discretizerOptions, String[] algorithmOptions, int nClients, int nIterations, int nFolds, int seed, String suffix) {
        // Get the cross-validation splits for each client
        String datasetPath = baseDatasetPath + datasetName + ".arff";
        Instances[][][] splits = divide(datasetName, datasetPath, nFolds, nClients, seed);

        // Initialize the variables for running the federated learning
        String algorithmName;
        boolean buildStats;
        boolean fusionStats;
        boolean stats;
        Fusion fusionClient;
        Fusion fusionServer;
        Convergence convergence;
        String outputPath;
        Object[] models = new Object[nFolds];

        // First step, federate the discretization
        // to get the cut points for the algorithm
        algorithmName = "Bins_MDLP";
        buildStats = false;
        fusionStats = false;
        stats = false;
        fusionClient = new FusionPosition(-1);
        fusionServer = new Bins_Fusion();
        convergence = new NoneConvergence();
        outputPath = "";
        models = validate(datasetName, splits, seed, models, algorithmName, discretizerOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, 1, outputPath);

        // Second step, federate the class-conditional Bayesian network classifier
        algorithmName = "CCBN";
        buildStats = true;
        fusionStats = true;
        stats = false;
        fusionClient = new PT_Fusion_Client();
        fusionServer = new PT_Fusion_Server();
        convergence = new NoneConvergence();
        outputPath = baseOutputPath + datasetName + "/" + algorithmName + "_" + suffix + "_" + seed + ".csv";

        models = validate(datasetName, splits, seed, models, algorithmName, algorithmOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, nIterations, outputPath);
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
        Data data = new Weka(name, path);
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
     * Get the algorithm for the experiment.
     *
     * @param name The name of the algorithm.
     * @param options The options for the algorithm.
     * @param model The previous model in case it is required.
     * @return The algorithm.
     */
    private static LocalAlgorithm getAlgorithm(String name, String[] options, Object model) {
        // The algorithm depends on its name
        switch (name) {
            // Supervised discretization method
            case "Bins_MDLP":
                return new Bins_MDLP(options);
            // Class-conditional Bayesian network
            case "CCBN":
                double[][] cutPoints = (double[][]) model;
                return new CCBN(options, cutPoints);
            // Add more algorithms here
            default:
                return null;
        }
    }

    /**
     * Get the operation for the experiment.
     *
     * @param fold The fold number.
     * @param algorithmName The name of the algorithm.
     * @param seed The seed.
     * @param nClients The number of clients.
     * @return The operation.
     */
    private static String getOperation(int fold, String algorithmName, int seed, int nClients) {
        // The operation depends on the algorithm
        switch (algorithmName) {
            // Supervised discretization method
            case "Bins_MDLP":
                return "";
            // Class-conditional Bayesian network
            case "CCBN":
                return fold + "," + algorithmName + "," + seed + "," + nClients;
            // Add more algorithms here
            default:
                return "";
        }
    }

    /**
     * Validate the algorithm for the experiment.
     *
     * @param datasetName The name of the dataset.
     * @param splits The data splits.
     * @param models The previous models in case they are required.
     * @param algorithmName The name of the algorithm.
     * @param algorithmOptions The options for the algorithm.
     * @param buildStats Whether to build statistics.
     * @param fusionStats Whether to build fusion statistics.
     * @param stats Whether to build statistics.
     * @param fusionClient The client fusion method.
     * @param fusionServer The server fusion method.
     * @param convergence The convergence method.
     * @param nIterations The number of iterations.
     * @param operation The operation.
     * @param outputPath The output path.
     * @return The models.
     */
    private static Object[] validate(String datasetName, Instances[][][] splits, int seed, Object[] models, String algorithmName, String[] algorithmOptions, boolean buildStats, boolean fusionStats, boolean stats, Fusion fusionClient, Fusion fusionServer, Convergence convergence, int nIterations, String outputPath) {
        // The first level of the splits corresponds to the folds
        int nFolds = splits.length;

        for (int i = 0; i < nFolds; i++) {
            // Get the partitions for the clients in the current fold 
            Instances[][] partitions = splits[i];
            int nClients = partitions.length;

            Object model = models[i];
            String operation = getOperation(i, algorithmName, seed, nClients);

            models[i] = run(datasetName, partitions, algorithmName, algorithmOptions, model, buildStats, fusionStats, fusionClient, stats, fusionServer, convergence, nIterations, operation, outputPath);
        }

        return models;
    }

    /**
     * Run the algorithm for the experiment.
     *
     * @param datasetName The name of the dataset.
     * @param partitions The data partitions.
     * @param algorithmName The name of the algorithm.
     * @param algorithmOptions The options for the algorithm.
     * @param model The previous model in case it is required.
     * @param buildStats Whether to build statistics.
     * @param fusionStats Whether to build fusion statistics.
     * @param fusionClient The client fusion method.
     * @param stats Whether to build statistics.
     * @param fusionServer The server fusion method.
     * @param convergence The convergence method.
     * @param nIterations The number of iterations.
     * @param operation The operation.
     * @param outputPath The output path.
     * @return The model.
     */
    private static Object run(String datasetName, Instances[][] partitions, String algorithmName, String[] algorithmOptions, Object model, boolean buildStats, boolean fusionStats, Fusion fusionClient, boolean stats, Fusion fusionServer, Convergence convergence, int nIterations, String operation, String outputPath) {
        int nClients = partitions.length;
        ArrayList<Client> clients = new ArrayList<Client>(nClients);

        for (int i = 0; i < nClients; i++) {
            Instances train = partitions[i][0];
            Instances test = partitions[i][1];
            Data data = new Weka(datasetName, train, test);

            LocalAlgorithm algorithm = getAlgorithm(algorithmName, algorithmOptions, model);
            Client client = new Client(fusionClient, algorithm, data);
            client.setStats(buildStats, fusionStats, outputPath);
            client.setID(i);
            client.setExperimentName(operation);
            clients.add(client);
        }

        Server server = new Server(fusionServer, convergence, clients);
        server.setStats(stats, outputPath);
        server.setExperimentName(operation);
        server.setnIterations(nIterations);

        server.run();

        return server.getGlobalModel().getModel();
    }

    /**
     * The main method.
     * @param args The arguments.
     */
    public static void main(String[] args) {
        String datasetName = args[0];
        String structure = args[1];  // Possibles values: "NB"
        String parameterLearning = args[2];  // Possibles values: "dCCBN", "wCCBN", "eCCBN"
        int nClients = Integer.parseInt(args[3]);
        int nIterations = Integer.parseInt(args[4]);
        int nFolds = Integer.parseInt(args[5]);
        int seed = Integer.parseInt(args[6]);

        String[] discretizerOptions = new String[] {""};
        String[] algorithmOptions = new String[] {"-S", structure, "-P", parameterLearning};

        String suffix = structure + "_" + parameterLearning;

        CCBNExperiment.run(datasetName, discretizerOptions, algorithmOptions, nClients, nIterations, nFolds, seed, suffix);
    }
}