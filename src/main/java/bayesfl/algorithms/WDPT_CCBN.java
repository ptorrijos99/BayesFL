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
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.*;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.model.Model;
import bayesfl.model.WDPT;

import static org.albacete.simd.utils.Utils.*;


/**
 * A class representing a class-conditional Bayesian network algorithm.
 */
public class WDPT_CCBN implements LocalAlgorithm {

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
     * The n of AnDE. 0 means Naive Bayes, 1 means A1DE, 2 means A2DE, etc.
     */
    private int nAnDE = 0;

    /**
     * The maximum number of iterations.
     */
    private int maxIterations;

    /**
     * The cut points of the discretization filter.
     */
    private final double[][] cutPoints;

    /**
     * The options to set the parameters of the algorithm.
     */
    private final String[] options;

    /**
     * Global class maps for synthetic classes.
     */
    private final List<Map<String, Integer>> globalClassMaps;

    /**
     * Constructor.
     *
     * @param options      The options to set the parameters of the algorithm.
     * @param cutPoints    The cut points of the discretization filter.
     */
    public WDPT_CCBN(String[] options, double[][] cutPoints, List<Map<String, Integer>> globalClassMaps) {
        this.cutPoints = cutPoints;
        this.options = Arrays.copyOf(options, options.length);
        this.globalClassMaps = globalClassMaps;

        // Copy the options to avoid modifying the original array
        try {
            String structure = Utils.getOption("S", this.options);
            if (structure.startsWith("A") && structure.endsWith("DE")) {
                nAnDE = Integer.parseInt(structure.substring(1, structure.length() - 2));
            }
            // The internal wdBayes structure is always NB
            setInternalStructureToNB();
        } catch (Exception ignored) {}
    }

    /**
     * Builds a local model using the provided data.
     *
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        // Get the instances from the data
        Instances originalData = (Instances) data.getData();
        int nAttributes = originalData.numAttributes() - 1; // excluding class

        // Generate combinations of attributes (n-AnDE structure)
        List<int[]> combinations = generateCombinations(nAttributes, nAnDE);
        List<wdBayesParametersTree> trees = new ArrayList<>();
        List<AbstractClassifier> classifiers = new ArrayList<>();
        List<Minimizer> minimizers = new ArrayList<>();
        List<ObjectiveFunction> functions = new ArrayList<>();
        List<Map<String, Integer>> syntheticClassMaps = new ArrayList<>();

        for (int i = 0; i < combinations.size(); i++) {
            int[] indices = combinations.get(i);
            Map<String, Integer> classMap = globalClassMaps.get(i);
            Instances modified = redefineClassAttribute(originalData, indices, classMap);
            //Instances completed = ensureAllClassValuesPresent(modified, classMap);

            // Build the classifier using the instances
            wdBayes algorithm = new wdBayes();
            try {
                algorithm.setOptions(Arrays.copyOf(options, options.length));
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Get the number of iterations from the algorithm
            this.maxIterations = algorithm.getM_MaxIterations();

            // Initialize the minimizer for the posterior parameter estimation
            Minimizer minimizer = new Minimizer();
            StopConditions sc = minimizer.getStopConditions();
            sc.setMaxGradientNorm(this.maxGradientNorm);
            sc.setMaxIterations(this.maxIterations);
            minimizers.add(minimizer);

            FilteredClassifier classifier = new FilteredClassifier();
            classifier.setClassifier(algorithm);

            // Set the discretization filter
            Filter filter;
            if (cutPoints != null) {
                filter = new Dummy();
                ((Dummy) filter).setCutPoints(cutPoints);
            } else {
                filter = new AllFilter(); // Bypass discretization
            }
            classifier.setFilter(filter);

            try {
                classifier.buildClassifier(modified);
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Save the tree-based storage and the objective function because
            // they are needed to build the local model from the global model
            trees.add(algorithm.getdParameters_());
            functions.add(algorithm.getObjectiveFunction());
            classifiers.add(classifier);
            syntheticClassMaps.add(classMap);
        }

        return new WDPT(trees, classifiers, minimizers, combinations, syntheticClassMaps, functions);
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

        // Retrieve trees, classifiers, and objective functions from the previous model
        WDPT previous = (WDPT) localModel;
        List<wdBayesParametersTree> oldTrees = previous.getTrees();
        List<AbstractClassifier> classifiers = previous.getClassifiers();
        List<Minimizer> minimizers = previous.getMinimizers();
        List<ObjectiveFunction> functions = previous.getFunctions();
        List<int[]> combinations = previous.getCombinations();
        List<Map<String, Integer>> classMaps = previous.getSyntheticClassMaps();

        List<wdBayesParametersTree> newTrees = new ArrayList<>();

        // Optimize each tree individually using its objective function
        for (int i = 0; i < oldTrees.size(); i++) {
            wdBayesParametersTree tree = oldTrees.get(i);
            double[] parameters = tree.getParameters();
            try {
                minimizers.get(i).run(functions.get(i), parameters);
            } catch (Exception e) {
                e.printStackTrace();
            }

            // Reuse the tree reference, since it has been updated internally
            newTrees.add(tree);
        }

        // Return a new model instance with updated parameters
        return new WDPT(newTrees, classifiers, minimizers, combinations, classMaps, functions);
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
     * Ensures that the structure option (-S) is set to "NB" in the options array.
     * <p>
     * If the -S flag is already present, replaces its associated value with "NB".
     * If not present, inserts "-S" and "NB" into the first pair of empty positions.
     * </p>
     */
    private void setInternalStructureToNB() {
        int pos = -1;
        for (int i = 0; i < this.options.length; i++) {
            if ("-S".equals(this.options[i])) {
                pos = i;
                break;
            }
        }

        if (pos != -1 && pos + 1 < this.options.length) {
            this.options[pos + 1] = "NB";
        } else {
            for (int i = 0; i < this.options.length - 1; i++) {
                if (this.options[i].isEmpty() && this.options[i + 1].isEmpty()) {
                    this.options[i] = "-S";
                    this.options[i + 1] = "NB";
                    break;
                }
            }
        }
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
