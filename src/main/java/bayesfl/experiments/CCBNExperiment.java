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
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Third-party imports.
 */
import bayesfl.algorithms.*;
import bayesfl.fusion.*;
import bayesfl.model.Classes;
import bayesfl.privacy.Gaussian_Noise;
import bayesfl.privacy.Laplace_Noise;
import bayesfl.privacy.NumericNoiseGenerator;
import bayesfl.privacy.ZCDP_Noise;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Local application imports.
 */
import bayesfl.Client;
import bayesfl.Server;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.computeSensitivity;

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
    public static void run(String folder, String datasetName, String[] discretizerOptions, String[] algorithmOptions, String[] clientOptions, String[] serverOptions, String[] dpOptions, int nClients, int nIterations, int nFolds, int seed, String suffix) {
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
        Object[] modelsAnDE = new Object[nFolds];

        // STEP 1 — Federate discretization
        buildStats = false;
        fusionStats = false;
        stats = false;
        fusionClient = new FusionPosition(-1);
        fusionServer = new Bins_Fusion();
        convergence = new NoneConvergence();
        outputPath = "";

        models = validate(datasetName, splits, seed, models, null, discretizerName, discretizerOptions, clientOptions, discretizerName, discretizerOptions, dpOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, 1, outputPath);

        // STEP 2 — Federate AnDE synthetic classes
        buildStats = false;
        fusionStats = false;
        stats = false;
        fusionClient = new FusionPosition(-1);
        fusionServer = new Classes_Fusion();  // new federation logic
        convergence = new NoneConvergence();
        outputPath = "";

        modelsAnDE = validate(datasetName, splits, seed, modelsAnDE, null, discretizerName, discretizerOptions, clientOptions, "Classes_AnDE", algorithmOptions,  dpOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence,1, outputPath);


        // STEP 3 — Real training and fusion of the algorithms
        buildStats = true;
        fusionStats = true;
        stats = false;
        // copy the client options to avoid modifying the original array
        fusionClient = getClientFusion(algorithmName, Arrays.copyOf(clientOptions, clientOptions.length));
        fusionServer = getServerFusion(algorithmName, Arrays.copyOf(serverOptions, serverOptions.length));
        convergence = new NoneConvergence();
        outputPath = baseOutputPath + algorithmName + "_" + suffix;

        validate(datasetName, splits, seed, models, modelsAnDE, discretizerName, discretizerOptions, clientOptions, algorithmName, algorithmOptions, dpOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, nIterations, outputPath);
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
    private static LocalAlgorithm getAlgorithm(String name, String[] options, Object model, Object modelAnDE) {
        switch (name) {
            // Supervised discretization wrapper
            case "Bins_Supervised" -> {
                return new Bins_Supervised(options);
            }

            // Unsupervised discretization wrapper
            case "Bins_Unsupervised" -> {
                return new Bins_Unsupervised(options);
            }

            // AnDE synthetic class discovery (structure only)
            case "Classes_AnDE" -> {
                return new Classes_AnDE(options);
            }

            // Naive Bayes using Weka's implementation and external cut points
            case "PT_NB" -> {
                double[][] cutPoints = (double[][]) model;

                // Check for AnDE option
                int nAnDE = 0;
                try {
                    options = Arrays.copyOf(options, options.length);
                    String structure = Utils.getOption("S", options);
                    if (structure.startsWith("A") && structure.endsWith("DE")) {
                        nAnDE = Integer.parseInt(structure.substring(1, structure.length() - 2));
                    }
                } catch (Exception ignored) {}

                Classes structure = (Classes) modelAnDE;
                return new PT_AnDE(cutPoints, nAnDE, structure.getSyntheticClassMaps());
            }

            // Federated class-conditional Bayesian network with cut point info
            case "WDPT_CCBN" -> {
                double[][] cutPoints = (double[][]) model;

                // Retrieve combinations and class maps from modelsAnDE
                Classes structure = (Classes) modelAnDE;
                return new WDPT_CCBN(options, cutPoints, structure.getSyntheticClassMaps());
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
            // Case for Weka's Naive Bayes implementation (always maintain the model obtained from the server)
            case "PT_NB" -> {
                return new FusionPosition(-1);
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
     * @param algorithmName The name of the learning algorithm (e.g., "Weka", "dCCBN").
     * @param options The command-line-style flags passed to configure fusion behavior.
     * @return An instance of the corresponding {@link Fusion} strategy for the server.
     */
    private static Fusion getServerFusion(String algorithmName, String[] options) {
        switch (algorithmName) {
            // Case for Weka's Naive Bayes implementation
            case "PT_NB" -> {
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
    private static String getOperation(int fold, String[] discretizerOptions, String algorithmName, String[] algorithmOptions, String[] clientOptions, String[] dpOptions, int seed, int nClients) {
        // Default values if options were not found
        String structure = null;
        String parameterLearning = null;
        String nBins = null;
        boolean fuseParameters = false;
        boolean fuseProbabilities = false;

        String dpType = null;
        double epsilon = 0;
        double delta = 0;
        double rho = 0;
        double sensitivity = 0;
        boolean autoSensitivity = false;

        try {
            // Copy arrays to avoid destructive modifications as it clears matched flags
            discretizerOptions = Arrays.copyOf(discretizerOptions, discretizerOptions.length);
            algorithmOptions = Arrays.copyOf(algorithmOptions, algorithmOptions.length);
            dpOptions = Arrays.copyOf(dpOptions, dpOptions.length);

            // Retrieve discretization and algorithm options
            nBins = Utils.getOption("B", discretizerOptions);
            structure = Utils.getOption("S", algorithmOptions);
            parameterLearning = Utils.getOption("P", algorithmOptions);

            // Retrieve the FP and FPR flags from the client options
            fuseParameters = Utils.getFlag("FP", clientOptions);
            fuseProbabilities = Utils.getFlag("FPR", clientOptions);

            // Retrieve the DP options
            dpType = Utils.getOption("DP", dpOptions);
            if (dpType.equalsIgnoreCase("Laplace")) {
                epsilon = Double.parseDouble(Utils.getOption("E", dpOptions));
                sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
            } else if (dpType.equalsIgnoreCase("Gaussian")) {
                epsilon = Double.parseDouble(Utils.getOption("E", dpOptions));
                delta = Double.parseDouble(Utils.getOption("D", dpOptions));
                sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
            } else if (dpType.equalsIgnoreCase("ZCDP")) {
                rho = Double.parseDouble(Utils.getOption("R", dpOptions));
                sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
            }
            autoSensitivity = Utils.getFlag("AUTO", dpOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // The operation depends on the algorithm
        switch (algorithmName) {
            //
            case "Bins_Supervised", "Bins_Unsupervised", "Classes_Fusion" -> {
                return "";
            }
            case "PT_NB" -> {
                String combinedName = structure + "-" + parameterLearning;
                return fold + "," + combinedName + "," + nBins + "," + seed + "," + nClients + ",false,false" + "," + dpType + "," + epsilon + "," + delta + "," + rho + "," + sensitivity + "," + autoSensitivity;
            }
            case "WDPT_CCBN" -> {
                // Construct a composite name
                String combinedName = structure + "-" + parameterLearning;
                return fold + "," + combinedName + "," + nBins + "," + seed + "," + nClients + "," + fuseParameters + "," + fuseProbabilities + "," + dpType + "," + epsilon + "," + delta + "," + rho + "," + sensitivity + "," + autoSensitivity;
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
     * @param modelsAnDE The AnDE class cuts in case they are required.
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
    protected static Object[] validate(String datasetName, Instances[][][] splits, int seed, Object[] models, Object[] modelsAnDE, String discretizerName, String[] discretizerOptions, String[] clientOptions, String algorithmName, String[] algorithmOptions, String[] dpOptions, boolean buildStats, boolean fusionStats, boolean stats, Fusion fusionClient, Fusion fusionServer, Convergence convergence, int nIterations, String outputPath) {
        // The first level of the splits corresponds to the folds
        int nFolds = splits.length;

        if (!outputPath.isEmpty() && done(outputPath)) {
            System.out.println("Skip: " + outputPath);
            return models;
        }

        for (int i = 0; i < nFolds; i++) {
            // Copy the options to avoid modifying the original array
            String[] discretizerOptionsTemp = Arrays.copyOf(discretizerOptions, discretizerOptions.length);
            String[] algorithmOptionsTemp = Arrays.copyOf(algorithmOptions, algorithmOptions.length);
            String[] clientOptionsTemp = Arrays.copyOf(clientOptions, clientOptions.length);
            String[] dpOptionsTemp = Arrays.copyOf(dpOptions, dpOptions.length);

            // Get the partitions for the clients in the current fold 
            Instances[][] partitions = splits[i];
            int nClients = partitions.length;

            Object model = models[i];
            Object modelAnDE = null;
            if (modelsAnDE != null) {
                modelAnDE = modelsAnDE[i];
            }
            String operation = getOperation(i, discretizerOptionsTemp, algorithmName, algorithmOptionsTemp, clientOptionsTemp, dpOptionsTemp, seed, nClients);

            models[i] = run(datasetName, partitions, algorithmName, algorithmOptionsTemp, dpOptionsTemp, model, modelAnDE, buildStats, fusionStats, fusionClient, stats, fusionServer, convergence, nIterations, operation, outputPath);
        }

        return models;
    }

    private static boolean done(String path) {
        String dir = "results/Client/";
        return new java.io.File(dir + "Build/" + path).exists() ||
                new java.io.File(dir + "Fusion/" + path).exists();
    }

    /**
     * Creates a {@link NumericNoiseGenerator} based on Weka-style differential privacy options.
     * <p>
     * Accepted formats:
     * <ul>
     *   <li><b>Laplace:</b> {"-DP", "Laplace", "-E", "epsilon", "-S", "sensitivity"}</li>
     *   <li><b>Gaussian:</b> {"-DP", "Gaussian", "-E", "epsilon", "-D", "delta", "-S", "sensitivity"}</li>
     *   <li><b>ZCDP:</b> {"-DP", "ZCDP", "-R", "rho", "-S", "sensitivity"}</li>
     * </ul>
     * If the input is empty or invalid, the method returns {@code null}.
     *
     * @param dpOptions array of command-line style options
     * @return a configured {@link NumericNoiseGenerator} instance or {@code null} if disabled or invalid
     */
    private static NumericNoiseGenerator getNoiseGenerator(String[] dpOptions) {
        if (dpOptions == null || dpOptions.length == 0) return null;

        // Copy the options to avoid modifying the original array
        dpOptions = Arrays.copyOf(dpOptions, dpOptions.length);

        try {
            String type = Utils.getOption("DP", dpOptions);

            switch (type.toLowerCase()) {
                case "laplace" -> {
                    double epsilon = Double.parseDouble(Utils.getOption("E", dpOptions));
                    double sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
                    return new Laplace_Noise(epsilon, sensitivity);
                }
                case "gaussian" -> {
                    double epsilon = Double.parseDouble(Utils.getOption("E", dpOptions));
                    double delta = Double.parseDouble(Utils.getOption("D", dpOptions));
                    double sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
                    return new Gaussian_Noise(epsilon, delta, sensitivity);
                }
                case "zcdp" -> {
                    double rho = Double.parseDouble(Utils.getOption("R", dpOptions));
                    double sensitivity = Double.parseDouble(Utils.getOption("S", dpOptions));
                    return new ZCDP_Noise(rho, sensitivity);
                }
                default -> {
                    return null;
                }
            }
        } catch (Exception e) {
            System.err.println("Invalid DP configuration: " + Arrays.toString(dpOptions));
            return null;
        }
    }

    /**
     * Run the algorithm for the experiment.
     *
     * @param datasetName The name of the dataset.
     * @param partitions The data partitions.
     * @param algorithmName The name of the algorithm.
     * @param algorithmOptions The options for the algorithm.
     * @param model The previous model in case it is required.
     * @param modelAnDE The AnDE class cuts in case it is required.
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
    private static Object run(String datasetName, Instances[][] partitions, String algorithmName, String[] algorithmOptions, String[] dpOptions, Object model, Object modelAnDE, boolean buildStats, boolean fusionStats, Fusion fusionClient, boolean stats, Fusion fusionServer, Convergence convergence, int nIterations, String operation, String outputPath) {
        int nClients = partitions.length;
        ArrayList<Client> clients = new ArrayList<>(nClients);

        for (int i = 0; i < nClients; i++) {
            Instances train = partitions[i][0];
            Instances test = partitions[i][1];
            Data data = new Weka_Instances(datasetName, train, test);

            LocalAlgorithm algorithm = getAlgorithm(algorithmName, algorithmOptions, model, modelAnDE);

            NumericNoiseGenerator dp = getNoiseGenerator(dpOptions);
            try {
                // Copy the dpOptions to avoid modifying the original array
                dpOptions = Arrays.copyOf(dpOptions, dpOptions.length);
                if (Utils.getFlag("AUTO", dpOptions)) {
                    dp.setSensitivity(computeSensitivity(data, algorithmOptions));
                }
            } catch (Exception e) {throw new RuntimeException(e);}

            Client client = new Client(fusionClient, algorithm, data, dp);
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
     * @param args Command-line arguments (not used in this setup).
     */
    public static void main(String[] args) {
        // Default dataset and experimental configuration
        String folder = "Discretas";
        String datasetName = "Soybean";
        int nClients = 100;
        int seed = 42;
        int nFolds = 5;
        int nIterations = 10;

        // Structure and parameter learning configurations
        String structure = "A2DE";  // Possible values: "NB", "A1DE", "A2DE", ..., "AnDE"
        String parameterLearning = "wCCBN";  // Possible values: "dCCBN", "wCCBN", "eCCBN", and "Weka"
        String maxIterations = "5";

        // Fusion behaviour
        boolean fuseParameters = true;
        boolean fuseProbabilities = true;
        int nBins = -1;

        // Differential privacy parameters
        String dpType = "Laplace";  // "Laplace", "Gaussian", "ZCDP", "None"
        double epsilon = 1;      // for Laplace and Gaussian
        double delta = 1e-5;     // only for Gaussian
        double rho = 0.1;        // only for ZCDP
        double sensitivity = 1.0;  // default for PT; smaller for WDPT
        boolean autoSensitivity  = true; // Calculate sensitivity automatically based on the data columns

        // Check if arguments are provided
        // If arguments are provided, read the parameters from the file and override the default values
        if (args.length > 0) {
            int index = Integer.parseInt(args[0]);
            String paramsFileName = args[1];
            //int threads = Integer.parseInt(args[2]);

            // Read the parameters from args
            String[] parameters = null;
            try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
                String line;
                for (int i = 0; i < index; i++)
                    br.readLine();
                line = br.readLine();
                parameters = line.split(" ");
            }
            catch(Exception e){ System.out.println(e); }

            System.out.println("Number of hyperparams: " + parameters.length);
            int i=0;
            for (String string : parameters) {
                System.out.println("Param[" + i + "]: " + string);
                i++;
            }

            // Read the parameters from file
            folder = parameters[0];
            datasetName = parameters[1];
            nClients = Integer.parseInt(parameters[2]);
            seed = Integer.parseInt(parameters[3]);
            nFolds = Integer.parseInt(parameters[4]);
            nIterations = Integer.parseInt(parameters[5]);
            structure = parameters[6];
            parameterLearning = parameters[7];
            maxIterations = parameters[8];
            fuseParameters = Boolean.parseBoolean(parameters[9]);
            fuseProbabilities = Boolean.parseBoolean(parameters[10]);
            nBins = Integer.parseInt(parameters[11]);

            // Add DP parameters if passed
            if (parameters.length >= 14) {
                dpType = parameters[12];
                sensitivity = Double.parseDouble(parameters[13]);
                autoSensitivity = Boolean.parseBoolean(parameters[14]);

                if (dpType.equalsIgnoreCase("Laplace")) {
                    epsilon = Double.parseDouble(parameters[15]);
                } else if (dpType.equalsIgnoreCase("Gaussian")) {
                    epsilon = Double.parseDouble(parameters[15]);
                } else if (dpType.equalsIgnoreCase("ZCDP")) {
                    rho = Double.parseDouble(parameters[15]);
                }

                if (dpType.equalsIgnoreCase("Gaussian")) {
                    delta = Double.parseDouble(parameters[16]);
                }
            } else {
                dpType = "None";
            }
        }

        // Use supervised discretization in case the number of bins is not provided and equal-frequency otherwise
        String[] discretizerOptions = nBins == -1 ? new String[] {} : new String[] {"-F", "-B", "" + nBins};

        // Options for local learning algorithm
        String[] algorithmOptions = new String[] {"-S", structure, "-P", parameterLearning, "-I", maxIterations};

        // Build shared fusion flags based on configuration
        List<String> flags = new ArrayList<>();
        if (fuseParameters) flags.add("-FP"); // Fuse parameter vectors
        if (fuseProbabilities) flags.add("-FPR"); // Fuse class-conditional probabilities

        // Convert flags to arrays for client and server options
        String[] type = new String[0];
        String[] clientOptions = flags.toArray(type);
        String[] serverOptions = flags.toArray(type);

        // Create DP options
        List<String> dpList = new ArrayList<>();
        switch (dpType.toLowerCase()) {
            case "laplace" -> dpList.addAll(List.of("-DP", "Laplace", "-E", "" + epsilon, "-S", "" + sensitivity));
            case "gaussian" -> dpList.addAll(List.of("-DP", "Gaussian", "-E", "" + epsilon, "-D", "" + delta, "-S", "" + sensitivity));
            case "zcdp" -> dpList.addAll(List.of("-DP", "ZCDP", "-R", "" + rho, "-S", "" + sensitivity));
        }
        if (autoSensitivity) dpList.add("-AUTO");
        String[] dpOptions = dpList.toArray(new String[0]);

        // Create output suffix for result identification
        String suffix = datasetName + "_" + nBins + "_" + structure + "_" + parameterLearning + "_" + maxIterations + "_" + fuseParameters + "_" + fuseProbabilities
                + "_" + dpType + "_" + epsilon + "_" + delta + "_" + rho + "_" + sensitivity + "_" + autoSensitivity
                + "_" + nClients + "_" + seed + "_" + nIterations + "_" + nFolds + ".csv";

        // Run the experiment
        CCBNExperiment.run(folder, datasetName, discretizerOptions, algorithmOptions, clientOptions, serverOptions, dpOptions, nClients, nIterations, nFolds, seed, suffix);
    }
}
