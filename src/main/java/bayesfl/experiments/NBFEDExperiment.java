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

import bayesfl.Client;
import bayesfl.Server;
import bayesfl.algorithms.Bins_MDLP;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.algorithms.PT_CCBN;
import bayesfl.algorithms.PT_NB;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.fusion.*;
import weka.core.Instances;

import java.util.ArrayList;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.CCBNExperiment.validate;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;

/**
 * A class representing an experiment with class-conditional Bayesian networks.
 */
public class NBFEDExperiment {

    /**
     * The base path for the datasets.
     */
    private static String baseDatasetPath = "res/classification/";

    /**
     * The base path for the results.
     */
    private static String baseOutputPath = "";


    public static void runWEKANB(String folder, String datasetName, String[] discretizerOptions, String[] algorithmOptions, int nClients, int nIterations, int nFolds, int seed, String suffix) {
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
        models = validate(datasetName, splits, seed, models, algorithmName, discretizerOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, 1, outputPath);

        // Second step, federate the class-conditional Bayesian network classifier
        algorithmName = "PT_NB";
        buildStats = true;
        fusionStats = true;
        stats = false;
        fusionClient = new FusionPosition(-1);
        fusionServer = new PT_Fusion_Server();
        convergence = new NoneConvergence();
        outputPath = baseOutputPath + datasetName + "_" + suffix + "_" + seed +  ".csv";

        models = validate(datasetName, splits, seed, models, algorithmName, algorithmOptions, buildStats, fusionStats, stats, fusionClient, fusionServer, convergence, nIterations, outputPath);
    }


    /**
     * The main method.
     * @param args The arguments.
     */
    public static void main(String[] args) {
        //args = readParametersFromArgs(args);

        String folder = "AnDE";
        String datasetName = "Iris_Classification";
        int nClients = 2;
        int seed = 42;
        int nFolds = 2;

        /*String folder = args[0];
        String datasetName = args[1];
        int nClients = Integer.parseInt(args[2]);
        int seed = Integer.parseInt(args[3]);
        int nFolds = Integer.parseInt(args[4]);*/

        String suffix = "NB-FED_" + nClients;
        String[] discretizerOptions = new String[] {""};
        String[] algorithmOptions = new String[] {""};

        NBFEDExperiment.runWEKANB(folder, datasetName, discretizerOptions, algorithmOptions, nClients, 1, nFolds, seed, suffix);
    }
}
