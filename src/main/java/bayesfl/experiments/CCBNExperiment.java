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
import java.util.Arrays;
import java.util.List;

/**
 * Third-party imports.
 */
import weka.core.Instances;
import weka.core.Utils;

/**
 * Local application imports.
 */
import bayesfl.Client;
import bayesfl.Server;
import bayesfl.algorithms.Bins_Supervised;
import bayesfl.algorithms.Bins_Unsupervised;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.algorithms.PT_NB;
import bayesfl.algorithms.WDPT_CCBN;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import static bayesfl.data.Weka_Instances.divide;
import bayesfl.fusion.Bins_Fusion;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.FusionPosition;
import bayesfl.fusion.PT_Fusion_Client;
import bayesfl.fusion.PT_Fusion_Server;
import bayesfl.fusion.WDPT_Fusion_Client;
import bayesfl.fusion.WDPT_Fusion_Server;

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
     * Runs a federated learning experiment using class-conditional Bayesian networks.
     * <p>
     * This method performs multi-fold, multi-client training and fusion over a given dataset,
     * applying a specified discretization strategy, algorithm configurations, and output labeling.
     * </p>
     *
     * @param folder The path to the folder containing the dataset files.
     * @param datasetName The name of the dataset (used to locate and label results).
     * @param discretizerOptions The options used to configure the discretization algorithm.
     * @param algorithmOptions The options for configuring the local learning algorithm.
     * @param clientOptions The options applied specifically at the client level.
     * @param serverOptions The options applied specifically at the server (fusion) level.
     * @param nClients The number of clients (partitions) used in the federated setting.
     * @param nIterations The number of training iterations (e.g., communication rounds).
     * @param nFolds The number of cross-validation folds to use in the experiment.
     * @param seed The random seed for reproducibility of data splits.
     * @param suffix The suffix used to distinguish the output file or experiment version.
     */
    public static void run(String folder, String datasetName, String[] discretizerOptions, String[] algorithmOptions, String[] clientOptions, String[] serverOptions, int nClients, int nIterations, int nFolds, int seed, String suffix) {
        // Get the cross-validation splits for each client
        String datasetPath = baseDatasetPath + folder + "/" + datasetName + ".arff";
        Instances[][][] splits = divide(datasetName, datasetPath, nFolds, nClients, seed);

        // Initialize the variables for running the federated learning
        String discretizerName = getDiscretizerName(discretizerOptions);
        String algorithmName = getAlgorithmName(algorithmOptions);
        boolean buildStats;
        boolean fusionStats;
        boolean stats;
        Fusion fusionClient;
        Fusion fusionServer;
        Convergence convergence;
        String outputPath;
        Object[] models = new Object[nFolds];

        // First step, federate the discretization to use the same cut points for all algorithms
        buildStats = false;
        fusionStats = false;
        stats = false;
        fusionClient = new FusionPosition(-1);
        fusionServer = new Bins_Fusion();
        convergence = new NoneConvergence();
        outputPath = "";

        models = validate(datasetName, splits, seed, models, discretizerName, discretizerOptions, discretizerName, discretizerOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, 1, outputPath);

        // Second step, federate the algorithms
        buildStats = true;
        fusionStats = true;
        stats = false;
        fusionClient = getClientFusion(algorithmName, clientOptions);
        fusionServer = getServerFusion(algorithmName, serverOptions);
        convergence = new NoneConvergence();
        outputPath = baseOutputPath + algorithmName + "_" + suffix;

        models = validate(datasetName, splits, seed, models, discretizerName, discretizerOptions, algorithmName, algorithmOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, nIterations, outputPath);
    }

    /**
     * Returns the name of the discretization method based on the provided options.
     *
     * @param options The discretization options passed to the method. If the array is empty, it indicates that supervised discretization is being used.
     * @return The name of the discretization method.
     */
    private static String getDiscretizerName(String[] options) {
        return options.length == 0 ? "Bins_Supervised" : "Bins_Unsupervised";
    }

    /**
     * Returns the name of the algorithm based on the provided options.
     *
     * @param options The options passed to the method, typically containing algorithm-related flags.
     * @return The name of the algorithm as a string.
     */
    private static String getAlgorithmName(String[] options) {
        // Copy the options array to avoid modifying the original
        options = Arrays.copyOf(options, options.length);

        try {
            // Extract the parameter learning method using the corresponding flag
            String parameterLearning = Utils.getOption("P", options);

            // Return appropriate internal algorithm name based on parameter learning type
            return parameterLearning.contains("Weka") ? "PT_NB" : "WDPT_CCBN";

        } catch (Exception e) {
            // If option parsing fails, return null
            return null;
        }
    }

    /**
     * Returns the local algorithm instance based on the algorithm name and configuration.
     *
     * @param name The name of the algorithm.
     * @param options The options used to configure the algorithm.
     * @param model A model object passed if required by the algorithm.
     * @return An instance of the selected local learning algorithm.
     */
    private static LocalAlgorithm getAlgorithm(String name, String[] options, Object model) {
        switch (name) {
            // Supervised discretization wrapper
            case "Bins_Supervised" -> {
                return new Bins_Supervised(options);
            }

            // Unsupervised discretization wrapper
            case "Bins_Unsupervised" -> {
                return new Bins_Unsupervised(options);
            }

            // Naive Bayes using Weka's implementation and external cut points
            case "Weka_NB" -> {
                double[][] cutPoints = (double[][]) model;
                return new PT_NB(cutPoints);
            }

            // Federated class-conditional Bayesian network with cut point info
            case "WDPT_CCBN" -> {
                double[][] cutPoints = (double[][]) model;
                return new WDPT_CCBN(options, cutPoints);
            }

            // Handle unknown algorithm names gracefully
            default -> {
                return null;
            }
        }
    }

    /**
     * Returns the fusion strategy used at the client side based on the algorithm name and provided options.
     *
     * @param algorithmName The name of the learning algorithm.
     * @param options The command-line-style flags passed to configure fusion behavior.
     * @return An instance of the corresponding {@link Fusion} strategy for the client.
     */
    private static Fusion getClientFusion(String algorithmName, String[] options) {
        switch (algorithmName) {
            // Case for Weka's Naive Bayes implementation
            case "Weka_NB" -> {
                return new PT_Fusion_Client();
            }

            // Case for class-conditional Bayesian networks
            case "WDPT_CCBN" -> {
                try {
                    // Parse boolean flags for fusion behavior from provided options
                    // These flags control whether to fuse parameters and/or probabilities
                    boolean fuseParameters = Utils.getFlag("FP", options);
                    boolean fuseProbabilities = Utils.getFlag("FPR", options);

                    return new WDPT_Fusion_Client(fuseParameters, fuseProbabilities);

                } catch (Exception e) {
                    // In case of any parsing or casting errors, return null
                    return null;
                }
            }

            // Catch-all for unsupported or unknown algorithm names
            default -> {
                return null;
            }
        }
    }

    /**
     * Returns the fusion strategy used at the server side based on the algorithm name and provided options.
     *
     * @param algorithmName The name of the learning algorithm (e.g., "Weka_NB", "dCCBN").
     * @param options The command-line-style flags passed to configure fusion behavior.
     * @return An instance of the corresponding {@link Fusion} strategy for the server.
     */
    private static Fusion getServerFusion(String algorithmName, String[] options) {
        switch (algorithmName) {
            // Case for Weka's Naive Bayes implementation
            case "Weka_NB" -> {
                return new PT_Fusion_Server();
            }

            // Case for class-conditional Bayesian networks
            case "WDPT_CCBN" -> {
                try {
                    // Parse boolean flags for fusion behavior from provided options
                    // These flags control whether to fuse parameters and/or probabilities
                    boolean fuseParameters = Utils.getFlag("FP", options);
                    boolean fuseProbabilities = Utils.getFlag("FPR", options);

                    return new WDPT_Fusion_Server(fuseParameters, fuseProbabilities);

                } catch (Exception e) {
                    // In case of any parsing or casting errors, return null
                    return null;
                }
            }

            // Catch-all for unsupported or unknown algorithm names
            default -> {
                return null;
            }
        }
    }

    /**
     * Constructs a CSV-like string that represents the current operation configuration.
     *
     * @param fold The current fold index of cross-validation.
     * @param discretizerOptions The options passed to the discretizer.
     * @param algorithmName The resolved internal name of the algorithm.
     * @param algorithmOptions The options passed to the learning algorithm.
     * @param seed The random seed for reproducibility.
     * @param nClients The number of clients participating in federated learning.
     * @return A string encoding the current configuration for logging or output files.
     */
    private static String getOperation(int fold, String[] discretizerOptions, String algorithmName, String[] algorithmOptions, int seed, int nClients) {
        // Default values if options were not found
        String structure = null;
        String parameterLearning = null;
        String nBins = null;

        try {
            // Copy arrays to avoid destructive modifications as it clears matched flags
            discretizerOptions = Arrays.copyOf(discretizerOptions, discretizerOptions.length);
            algorithmOptions = Arrays.copyOf(algorithmOptions, algorithmOptions.length);

            // Retrieve discretization and algorithm options
            nBins = Utils.getOption("B", discretizerOptions);
            structure = Utils.getOption("S", algorithmOptions);
            parameterLearning = Utils.getOption("P", algorithmOptions);

        } catch (Exception e) {
            e.printStackTrace();
        }
    
        // The operation depends on the algorithm
        switch (algorithmName) {
            //
            case "Bins_Supervised", "Bins_Unsupervised" -> {
                return "";
            }
            case "Weka_NB" -> {
                return fold + "," + algorithmName + "," + nBins + "," + seed + "," + nClients;
            }
            case "WDPT_CCBN" -> {
                // Construct a composite name
                String combinedName = structure + "-" + parameterLearning;
                return fold + "," + combinedName + "," + nBins + "," + seed + "," + nClients;
            }
            // Add more algorithms here
            default -> {
                return "";
            }
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
    protected static Object[] validate(String datasetName, Instances[][][] splits, int seed, Object[] models, String discretizerName, String[] discretizerOptions, String algorithmName, String[] algorithmOptions, boolean buildStats, boolean fusionStats, boolean stats, Fusion fusionClient, Fusion fusionServer, Convergence convergence, int nIterations, String outputPath) {
        // The first level of the splits corresponds to the folds
        int nFolds = splits.length;

        for (int i = 0; i < nFolds; i++) {
            // Get the partitions for the clients in the current fold 
            Instances[][] partitions = splits[i];
            int nClients = partitions.length;

            Object model = models[i];
            String operation = getOperation(i, discretizerOptions, algorithmName, algorithmOptions, seed, nClients);

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
     * Entry point for running a federated learning experiment using class-conditional Bayesian networks.
     * <p>
     * This method sets up the configuration for discretization, learning structure, parameter estimation,
     * number of clients, fusion method, and folds for cross-validation. It then launches the experiment
     * using {@link #run}.
     * </p>
     * @param Command-line arguments (not used in this setup).
     */
    public static void main(String[] args) {
        // Dataset and experimental configuration
        String folder = "Discretas";
        String datasetName = "King-rook-vs-King";
        int nClients = 50;
        int seed = 42;
        int nFolds = 5;
        int nIterations = 10;
    
        // Structure and parameter learning configurations
        String structure = "NB";  // Possibles values: "NB"
        String parameterLearning = "wCCBN";  // Possibles values: "dCCBN", "wCCBN", "eCCBN", and "Weka"
        String maxIterations = "10";

        // Fusion behaviour
        boolean fuseParameters = true;
        boolean fuseProbabilities = false;
        int nBins = -1;

        // Use supervised discretization in case the number of bins is not provided and equal-frequency otherwise
        String[] discretizerOptions = nBins == -1 ? new String[] {} : new String[] {"-F", "-B", "" + nBins};

        // Options for local learning algorithm
        String[] algorithmOptions = new String[] {"-S", structure, "-P", parameterLearning, "-I", maxIterations};

        // Build shared fusion flags based on configuration
        List<String> flags = new ArrayList<String>();
        if (fuseParameters) flags.add("-FP"); // Fuse parameter vectors
        if (fuseProbabilities) flags.add("-FPR"); // Fuse class-conditional probabilities

        // Convert flags to arrays for client and server options
        String[] type = new String[0];
        String[] clientOptions = flags.toArray(type);
        String[] serverOptions = flags.toArray(type);

        // Create output suffix for result identification
        String suffix = datasetName + "_" + nBins + "_" + structure + "_" + parameterLearning + "_" + maxIterations + "_" + fuseParameters + "_" + fuseProbabilities + "_" + nClients + "_" + seed + "_" + nIterations + "_" + nFolds + ".csv";

        // Run the experiment
        CCBNExperiment.run(folder, datasetName, discretizerOptions, algorithmOptions, clientOptions, serverOptions, nClients, nIterations, nFolds, seed, suffix);
    }
}
