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
import DataStructure.wdBayesParametersTree;
import EBNC.wdBayes;
import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;

import java.util.List;
import java.util.Map;
import java.util.logging.Filter;

import static bayesfl.experiments.utils.ExperimentUtils.*;

/**
 * A class representing a federated AnDE-style discriminative model.
 * This model stores a list of parameter trees and classifiers,
 * one for each combination used in the AnDE structure (n=0 is Naive Bayes).
 */
public class WDPT implements Model {

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
    private final String header = "bbdd,id,cv,algorithm,bins,seed,nClients,fusParams,fusProbs,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

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
