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

import bayesfl.data.Data;
import bayesfl.model.Model;
import bayesfl.model.PT;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

import java.util.*;

import static org.albacete.simd.utils.Utils.*;

/**
 * A class representing a WEKA AnDE algorithm (Naive Bayes is included as n=0).
 */
public class PT_AnDE implements LocalAlgorithm {

    /**
     * The classifier.
     */
    private ArrayList<AbstractClassifier> ensemble;

    /**
     * The cut points of the discretization filter.
     */
    private final double[][] cutPoints;

    /**
     * The n of AnDE. 0 means Naive Bayes, 1 means A1DE, 2 means A2DE, etc
     */
    private final int nAnDE;

    /**
     * Global class maps for synthetic classes.
     */
    private final List<Map<String, Integer>> globalClassMaps;

    /**
     * Constructor.
     *
     * @param cutPoints The cut points of the discretization filter.
     * @param nAnDE The n of AnDE. 0 means Naive Bayes, 1 means A1DE, 2 means A2DE, etc.
     */
    public PT_AnDE(double[][] cutPoints, int nAnDE, List<Map<String, Integer>> globalClassMaps) {
        this.nAnDE = nAnDE;
        this.cutPoints = cutPoints;
        this.globalClassMaps = globalClassMaps;
    }

    /**
     * Builds a local model using the provided data.
     *
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        Instances originalData = (Instances) data.getData();
        int nAttributes = originalData.numAttributes() - 1; // excluding class

        // Generate combinations of attributes
        List<int[]> combinations = generateCombinations(nAttributes, nAnDE);
        List<Map<String, Integer>> syntheticClassMaps = new ArrayList<>();
        this.ensemble = new ArrayList<>();

        for (int i = 0; i < combinations.size(); i++) {
            int[] indices = combinations.get(i);
            Map<String, Integer> classMap = globalClassMaps.get(i);
            Instances modified = redefineClassAttribute(originalData, indices, classMap);

            // Create a new classifier for the modified data
            NaiveBayes nb = new NaiveBayes();
            FilteredClassifier fc = new FilteredClassifier();
            fc.setClassifier(nb);

            Filter filter;
            if (cutPoints != null) {
                // Set the discretization filter if cut points are provided
                filter = new Dummy();
                ((Dummy) filter).setCutPoints(cutPoints);
            } else {
                // Set a bypass filter to skip the discretization
                filter = new AllFilter();
            }
            fc.setFilter(filter);

            try {
                fc.buildClassifier(modified);
            } catch (Exception e) {
                e.printStackTrace();
            }

            this.ensemble.add(fc);
            syntheticClassMaps.add(classMap);
        }

        return new PT(this.ensemble, combinations, syntheticClassMaps);
    }

    /**
     * Builds a local model using the provided data and existing local model. Not used in this case (one-shot model).
     *
     * @param localModel The previous local model.
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Model localModel, Data data) {
        return this.buildLocalModel(data);
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
    public ArrayList<AbstractClassifier> getClassifier() {
        return this.ensemble;
    }

    /** 
     * Retrieves the name of the algorithm.
     *
     * @return The name of the algorithm.
     */
    public String getAlgorithmName() {
        return "None";
    }

    /** 
     * Retrieves the name of the refinement.
     *
     * @return The name of the refinement.
     */
    public String getRefinementName() {
        return "None";
    }
}
