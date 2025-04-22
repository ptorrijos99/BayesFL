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
 * Package containing algorithms related with federated Bayesian networks.
 */
package bayesfl.algorithms;

/**
 * Third-party imports.
 */
import DataStructure.wdBayesParametersTree;
import EBNC.wdBayes;
import objectiveFunction.ObjectiveFunction;
import optimize.Minimizer;
import optimize.StopConditions;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.model.Model;
import bayesfl.model.WDPT;

/**
 * A class representing a class-conditional Bayesian network algorithm.
 */
public class WDPT_CCBN implements LocalAlgorithm {

    /**
     * The filter method.
     */
    private Filter filter;

    /**
     * The algorithm to use.
     */
    private wdBayes algorithm;

    /**
     * The classifier.
     */
    private FilteredClassifier classifier;

    /**
     * The tree-based parameter storage.
     */
    private wdBayesParametersTree tree;

    /**
     * The maximum number of iterations.
     */
	private int maxIterations;

    /**
     * The minimization algorithm.
     */
    private Minimizer minimizer;

    /**
     * The objective function.
     */
    private ObjectiveFunction function;

    /**
     * The maximum gradient norm.
     */
    private double maxGradientNorm = 0.00000000000000000000000000000001;

    /**
     * The name of the algorithm.
     */
    private String algorithmName = "WDBN_CCBN";

    /**
     * The name of the refinement method.
     */
    private String refinementName = "None";

    /**
     * Constructor.
     *
     * @param cutPoints The cut points of the discretization filter.
     * @param options The options to set the parameters of the algorithm.
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints) {
        // Copy the options to avoid modifying the original array
        options = options.clone();

        try {
            // Set the algorithm's hyperparameters using options
            // since the constructor doesn't permit setting them
            this.algorithm = new wdBayes();
            this.algorithm.setOptions(options);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get the number of iterations from the algorithm
        this.maxIterations = this.algorithm.getM_MaxIterations();

        // Initialize the minimizer for the posterior parameter estimation
        this.minimizer = new Minimizer();
        StopConditions sc = minimizer.getStopConditions();
        sc.setMaxGradientNorm(this.maxGradientNorm);
        sc.setMaxIterations(this.maxIterations);

        if (cutPoints != null) {
            // Set the discretization filter if cut points are provided
            this.filter = new Dummy();
            Dummy discretizer = (Dummy) this.filter;
            discretizer.setCutPoints(cutPoints);
        } else {
            // Set a bypass filter to skip the discretization
            this.filter = new AllFilter();
        }

        // Set the algorithm and filter to the classifier
        this.classifier = new FilteredClassifier();
        this.classifier.setFilter(this.filter);
        this.classifier.setClassifier(this.algorithm);
    }

    /**
     * Builds a local model using the provided data.
     *
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        // Get the instances from the data
        Instances instances = (Instances) data.getData();

        try {
            // Build the classifier using the instances
            this.classifier.buildClassifier(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Save the tree-based storage and the objective function because
        // they are needed to build the local model from the global model
        this.tree = this.algorithm.getdParameters_();
        this.function = this.algorithm.getObjectiveFunction();

        return new WDPT(this.tree, this.classifier);
    }

    /**
     * Builds a local model using the provided data and existing local model.
     *
     * @param localModel The existing local model.
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Model localModel, Data data) {
        // If there is no local model, build a new one
        if (localModel == null) {
            return this.buildLocalModel(data);
        }

        WDPT model = (WDPT) localModel;
        wdBayesParametersTree tree = model.getModel();
        double[] parameters = tree.getParameters();

        try {
            // Note that the parameters provided to the optimization function are not modified,
            // they are internally copied to the tree and these are the ones that are optimized
            this.minimizer.run(this.function, parameters);
        } catch (Exception exception) {
            exception.printStackTrace();
        }

        return new WDPT(this.tree, this.classifier);
    }

    /**
     * Refines the existing local model using the provided data. In this case, the model is not refined.
     *
     * @param oldModel The existing local model.
     * @param localModel The current local model.
     * @param data The data to refine the model with.
     * @return The refined local model.
     */
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return localModel;
    }

    /**
     * Retrieves the classifier.
     */
    public FilteredClassifier getClassifier() {
        return this.classifier;
    }

    /** 
     * Retrieves the name of the algorithm.
     *
     * @return The name of the algorithm.
     */
    public String getAlgorithmName() {
        return this.algorithmName;
    }

    /** 
     * Retrieves the name of the refinement.
     *
     * @return The name of the refinement.
     */
    public String getRefinementName() {
        return this.refinementName;
    }
}
