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

/**
 * Third-party imports.
 */
import bayesfl.algorithms.PT_NB;
import bayesfl.data.Weka_Instances;
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.algorithms.Bins_MDLP;
import bayesfl.algorithms.PT_CCBN;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Data;
import bayesfl.fusion.Bins_Fusion;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.FusionPosition;
import bayesfl.fusion.PT_Fusion_Client;
import bayesfl.fusion.PT_Fusion_Server;
import bayesfl.Client;
import bayesfl.Server;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;

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
    private static String baseOutputPath = "";

    /**
     * Run the experiment.
     *
     * @param datasetName The name of the dataset.
     * @param nBins The number of bins.
     * @param algorithmOptions The options for the algorithm.
     * @param nClients The number of clients.
     * @param nIterations The number of iterations.
     * @param nFolds The number of folds.
     * @param seed The seed.
     * @param suffix The suffix for the output file.
     */
    public static void run(String folder, String datasetName, int nBins, String[] algorithmOptions, int nClients, int nIterations, int nFolds, int seed, String suffix) {
        // Get the cross-validation splits for each client
        String datasetPath = baseDatasetPath + folder + "/" + datasetName + ".arff";
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

        String[] discretizerOptions;
        if (nBins == -1) {
            discretizerOptions = new String[] {""};
        } else {
            discretizerOptions = new String[] {"-F", "-B", ""+nBins};
        }

        models = validate(datasetName, splits, seed, models, nBins, algorithmName, discretizerOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, 1, outputPath);

        // Second step, federate the class-conditional Bayesian network classifier

        // WEKA NaiveBayes
        if (algorithmOptions[3].equals("WEKA")) {
            algorithmName = "PT_NB";
            fusionClient = new FusionPosition(-1);
        }
        // wdBayes
        else {
            algorithmName = "PT_CCBN";
            fusionClient = new PT_Fusion_Client();
        }
        buildStats = true;
        fusionStats = true;
        stats = false;
        fusionServer = new PT_Fusion_Server();
        convergence = new NoneConvergence();
        outputPath = baseOutputPath + algorithmName + "_" + suffix;

        models = validate(datasetName, splits, seed, models, nBins, algorithmName, algorithmOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, nIterations, outputPath);
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
            case "Bins_MDLP" -> {
                // If options is empty, supervised, else unsupervised
                if (options.length == 0) {
                    return new Bins_MDLP(true, options);
                } else {
                    return new Bins_MDLP(false, options);
                }
            }
            // Class-conditional Bayesian network
            case "PT_CCBN" -> {
                double[][] cutPoints = (double[][]) model;
                return new PT_CCBN(options, cutPoints);
            }
            case "PT_NB" -> {
                double[][] cutPoints = (double[][]) model;
                return new PT_NB(cutPoints);
            }
            // Add more algorithms here
            default -> {
                return null;
            }
        }
    }

    /**
     * Get the operation for the experiment.
     *
     * @param fold The fold number.
     * @param algorithmName The name of the algorithm.
     * @param seed The seed.
     * @param nClients The number of clients.
     * @param algorithmOptions The options for the algorithm.
     * @return The operation.
     */
    private static String getOperation(int fold, String algorithmName, int nBins, int seed, int nClients, String[] algorithmOptions) {
        // The operation depends on the algorithm
        switch (algorithmName) {
            // Supervised discretization method
            case "Bins_MDLP":
                return "";
            // Class-conditional Bayesian network
            case "PT_CCBN":
                String algName = algorithmOptions[1] + '-' + algorithmOptions[3];
                return fold + "," + algName + "," + nBins + "," + seed + "," + nClients;
            // WEKA Naive Bayes
            case "PT_NB":
                return fold + ",NB-FED," + nBins + "," + seed + "," + nClients;
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
     * @param outputPath The output path.
     * @return The models.
     */
    protected static Object[] validate(String datasetName, Instances[][][] splits, int seed, Object[] models, int nBins, String algorithmName, String[] algorithmOptions, boolean buildStats, boolean fusionStats, boolean stats, Fusion fusionClient, Fusion fusionServer, Convergence convergence, int nIterations, String outputPath) {
        // The first level of the splits corresponds to the folds
        int nFolds = splits.length;

        for (int i = 0; i < nFolds; i++) {
            // Get the partitions for the clients in the current fold 
            Instances[][] partitions = splits[i];
            int nClients = partitions.length;

            Object model = models[i];
            String operation = getOperation(i, algorithmName, nBins, seed, nClients, algorithmOptions);

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
        ArrayList<Client> clients = new ArrayList<>(nClients);

        for (int i = 0; i < nClients; i++) {
            Instances train = partitions[i][0];
            Instances test = partitions[i][1];
            Data data = new Weka_Instances(datasetName, train, test);

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
        args = readParametersFromArgs(args);

        /*String folder = "AnDE";
        String datasetName = "Adult";
        //int nClients = 500;
        int seed = 42;
        int nFolds = 5;
        int nIterations = 1;
        String structure = "NB";  // Possibles values: "NB"
        String parameterLearning = "WEKA";  // Possibles values: "dCCBN", "wCCBN", "eCCBN". "WEKA" for WEKA NB.
        String maxIterations = "10";
        int nBins = -1;*/

        String folder = args[0];
        String datasetName = args[1];
        int nClients = Integer.parseInt(args[2]);
        int seed = Integer.parseInt(args[3]);
        int nFolds = Integer.parseInt(args[4]);
        int nIterations = Integer.parseInt(args[5]);
        String structure = args[6];  // Possibles values: "NB"
        String parameterLearning = args[7];  // Possibles values: "dCCBN", "wCCBN", "eCCBN". "WEKA" for WEKA NB.
        String maxIterations = args[8];
        int nBins = Integer.parseInt(args[9]);


        String[] algorithmOptions = new String[] {"-S", structure, "-P", parameterLearning, "-I", maxIterations};



        /*int[] nClientss = {5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
        for (int nClients : nClientss) {
            System.out.println("Running with " + nClients + " clients");
            String suffix = datasetName + "_" + nBins + "_" + structure + "_" + parameterLearning + "_" + maxIterations + "_" + nClients + "_" + seed + ".csv";
            CCBNExperiment.run(folder, datasetName, nBins, algorithmOptions, nClients, nIterations, nFolds, seed, suffix);
        }*/

        String suffix = datasetName + "_" + nBins + "_" + structure + "_" + parameterLearning + "_" + maxIterations + "_" + nClients + "_" + seed + "_" + nIterations + "_" + nFolds + ".csv";
        CCBNExperiment.run(folder, datasetName, nBins, algorithmOptions, nClients, nIterations, nFolds, seed, suffix);
    }
}
