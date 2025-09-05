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
 * Package containing models related with federated Bayesian networks.
*/
package bayesfl.model;

/**
 * Third-party imports.
 */
import java.util.List;
import java.util.Map;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;

import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import EBNC.wdBayes;

import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.privacy.DenoisableModel;
import bayesfl.privacy.NoiseGenerator;


import static bayesfl.experiments.utils.ExperimentUtils.*;

/**
 * A class representing a federated AnDE-style discriminative model.
 * This model stores a list of parameter trees and classifiers,
 * one for each combination used in the AnDE structure (n=0 is Naive Bayes).
 */
public class WDPT implements DenoisableModel {

    /**
     * The list of tree-based parameter storages (one per combination).
     */
    private final List<wdBayesParametersTree> trees;

    /**
     * The list of classifiers (AbstractClassifier wrapping wdBayes).
     */
    private final List<AbstractClassifier> classifiers;

    /**
     * The list of minimizers used for local parameter optimization.
     */
    private final List<Minimizer> minimizers;

    /**
     * The list of attribute index combinations used to generate synthetic classes.
     */
    private final List<int[]> combinations;

    /**
     * The list of synthetic class mappings, one per combination.
     */
    private final List<Map<String, Integer>> syntheticClassMaps;

    /**
     * The list of objective functions used for local parameter optimization.
     */
    private final List<ObjectiveFunction> functions;

    /**
     * The header for the file.
     */
    private final String header = "bbdd,id,cv,algorithm,bins,seed,nClients,fusParams,fusProbs,dptype,epsilon,delta,rho,sensitivity,autoSens,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

    /**
     * Constructor.
     *
     * @param trees The list of parameter trees.
     * @param classifiers The list of classifiers.
     * @param minimizers The list of minimizers.
     * @param combinations The list of attribute combinations.
     * @param syntheticClassMaps The synthetic class mappings.
     * @param functions The list of objective functions.
     */
    public WDPT(List<wdBayesParametersTree> trees, List<AbstractClassifier> classifiers, List<Minimizer> minimizers,
                List<int[]> combinations, List<Map<String, Integer>> syntheticClassMaps, List<ObjectiveFunction> functions) {
        this.trees = trees;
        this.classifiers = classifiers;
        this.minimizers = minimizers;
        this.combinations = combinations;
        this.syntheticClassMaps = syntheticClassMaps;
        this.functions = functions;
    }

    /**
     * Applies noise to the internal probabilistic parameters of the model using a {@link NoiseGenerator}.
     * <p>
     * This method perturbs the class prior probabilities and conditional distributions in each
     * {@link wdBayesParametersTree}. Since the model stores these values in probability space
     * (including log-probabilities), noise is added in linear space, then the results are renormalized
     * and converted back to log-space when needed.
     * </p>
     * <p>This method does not guarantee formal (ε, δ)-differential privacy,
     * as the original counts are not accessible and the sensitivity of probabilities is not bounded.
     * Instead, this noise injection serves as a heuristic privacy-preserving mechanism that introduces
     * uncertainty and impairs exact model reconstruction.</p>
     *
     * @param noise the {@link NoiseGenerator} used to sample perturbation noise
     */
    @Override
    public void applyNoise(NoiseGenerator noise) {
        for (wdBayesParametersTree tree : trees) {
            // Privatize classCounts[] (linear scale)
            double[] original = tree.getClassCounts();
            double[] noisy = noise.privatize(original);
            double[] renorm = normalize(noisy);
            System.arraycopy(renorm, 0, tree.classCounts, 0, renorm.length);

            // Privatize xyCount[] in log-space
            if (tree.wdBayesNode_ != null) {
                for (int i = 0; i < tree.wdBayesNode_.length; i++) {
                    applyNoiseToNode(tree.wdBayesNode_[i], noise);
                }
            }
        }
    }

    /**
     * Applies noise to a {@link wdBayesNode}'s conditional distribution stored in {@code xyCount[]}.
     * <p>
     * The {@code xyCount[]} array contains log-probabilities of P(x | parents). This method:
     * <ul>
     *     <li>Exponentiates the values to retrieve probabilities</li>
     *     <li>Adds noise using the given {@link NoiseGenerator}</li>
     *     <li>Renormalizes the result to form a valid probability distribution</li>
     *     <li>Converts the values back to log-space and overwrites {@code xyCount[]}</li>
     * </ul>
     * This process is applied recursively to all children in the trie.
     * </p>
     * <p>Since noise is applied to normalized probabilities, the resulting privacy
     * protection is heuristic and does not satisfy formal differential privacy without
     * additional assumptions on sensitivity.</p>
     *
     * @param node  the node whose conditional probabilities will be perturbed
     * @param noise the {@link NoiseGenerator} to apply
     */
    private void applyNoiseToNode(wdBayesNode node, NoiseGenerator noise) {
        if (node == null || node.xyCount == null) return;

        int len = node.xyCount.length;
        double[] probs = new double[len];

        for (int i = 0; i < len; i++) {
            probs[i] = Math.exp(node.xyCount[i]); // Transform from log-space to probability
        }

        double[] noisy = noise.privatize(probs);
        for (int i = 0; i < len; i++) {
            noisy[i] = Math.max(noisy[i], 1e-9); // Avoid negative probabilities
        }
        
        double[] renormalized = normalize(noisy);

        for (int i = 0; i < len; i++) {
            node.xyCount[i] = Math.log(renormalized[i]); // Back to log-space
        }

        if (node.children != null) {
            for (DataStructure.wdBayesNode child : node.children) {
                applyNoiseToNode(child, noise);
            }
        }
    }

    /**
     * Renormalizes a probability vector to ensure it sums to 1.
     * <p>
     * This method adds lower bounds to avoid numerical instability from near-zero or negative values
     * introduced by the noise. It then performs standard normalization.
     * </p>
     *
     * @param values the noisy probability values (non-negative, not necessarily summing to 1)
     * @return a new normalized array representing a valid probability distribution
     */
    private double[] normalize(double[] values) {
        double sum = 0.0;
        for (double v : values) {
            sum += Math.max(v, 1e-12); // Avoid division by zero
        }

        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = Math.max(values[i], 1e-12) / sum;
        }
        return result;
    }


    /**
     * Gets the list of parameter trees.
     *
     * @return The list of parameter trees.
     */
    public List<wdBayesParametersTree> getTrees() {
        return trees;
    }

    /**
     * Gets the list of classifiers.
     *
     * @return The list of classifiers.
     */
    public List<AbstractClassifier> getClassifiers() {
        return classifiers;
    }

    /**
     * Gets the list of minimizers.
     *
     * @return The list of minimizers.
     */
    public List<Minimizer> getMinimizers() {
        return minimizers;
    }

    /**
     * Gets the list of attribute combinations.
     *
     * @return The list of combinations.
     */
    public List<int[]> getCombinations() {
        return combinations;
    }

    /**
     * Gets the list of synthetic class maps.
     *
     * @return The list of class maps.
     */
    public List<Map<String, Integer>> getSyntheticClassMaps() {
        return syntheticClassMaps;
    }

    /**
     * Gets the list of objective functions.
     *
     * @return The list of functions.
     */
    public List<ObjectiveFunction> getFunctions() {
        return functions;
    }

    /**
     * Gets the model. In this case, the model is the list of classifiers.
     *
     * @return The model.
     */
    @Override
    public List<wdBayesParametersTree> getModel() {
        return trees;
    }

    /**
     * Sets the model. This method is unused and throws an exception if called.
     * 
     * @param model The model.
     * @throws UnsupportedOperationException If the method is called.
     */
    public void setModel(Object model) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Saves the statistics of the model.
     *
     * @param operation The operation (e.g. "Client/Build", "Server").
     * @param epoch The epoch.
     * @param path The output path.
     * @param nClients The number of clients.
     * @param id The client ID (-1 if server).
     * @param data The training and test data.
     * @param iteration The current iteration.
     * @param time The time taken (seconds).
     */
    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        Weka_Instances weka = (Weka_Instances) data;
        Instances train = weka.getTrain();
        Instances test = weka.getTest();

        String statistics = "";
        String metrics;

        String bbdd = data.getName();
        int instances = train.numInstances();
        statistics += bbdd + "," + id + "," + operation + "," + epoch + "," + iteration + "," + instances + ",";

        FilteredClassifier classifier = (FilteredClassifier) this.classifiers.get(0);
        wdBayes algorithm = (wdBayes) classifier.getClassifier();
        int maxIterations = algorithm.getM_MaxIterations();
        statistics += maxIterations + ",";

        metrics = getClassificationMetricsEnsemble(classifiers, syntheticClassMaps, train);
        statistics += metrics;

        metrics = getClassificationMetricsEnsemble(classifiers, syntheticClassMaps, test);
        statistics += metrics;

        statistics += time + "\n";

        saveExperiment("results/" + epoch + "/" + path, header, statistics);
    }

    /**
     * Gets the score of the model. This method is unused and throws an exception if called.
     *
     * @return The score of the model.
     */
    public double getScore() {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Computes the score of the model. This method is unused and throws an exception if called.
     * 
     * @param data The data.
     * @return The score of the model.
     */
    public double getScore(Data data) {
        throw new UnsupportedOperationException("Method not implemented");
    }
}
