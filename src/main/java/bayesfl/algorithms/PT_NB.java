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
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.AllFilter;
import weka.filters.Filter;

/**
 * A class representing a WEKA Naive Bayes algorithm.
 */
public class PT_NB implements LocalAlgorithm {

    /**
     * The classifier.
     */
    private final FilteredClassifier classifier;

    /**
     * Constructor.
     */
    public PT_NB() {
        this(null);
    }

    /**
     * Constructor.
     *
     * @param cutPoints The cut points of the discretization filter.
     */
    public PT_NB(double[][] cutPoints) {
        NaiveBayes algorithm = new NaiveBayes();

        Filter filter;
        if (cutPoints != null) {
            // Set the discretization filter if cut points are provided
            filter = new PT_CCBN.Dummy();
            PT_CCBN.Dummy discretizer = (PT_CCBN.Dummy) filter;
            discretizer.setCutPoints(cutPoints);
        }

        else {
            // Set a bypass filter to skip the discretization
            filter = new AllFilter();
        }

        this.classifier = new FilteredClassifier();
        this.classifier.setFilter(filter);
        this.classifier.setClassifier(algorithm);
    }

    /**
     * Builds a local model using the provided data.
     *
     * @param data The data to build the model from.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        Instances instances = (Instances) data.getData();

        try {
            this.classifier.buildClassifier(instances);
        }
        catch (Exception exception) {
            exception.printStackTrace();
        }

        NaiveBayes clas = ((NaiveBayes)this.classifier.getClassifier());

        return new PT(clas.getClassEstimator(), clas.getConditionalEstimators(), this.classifier);
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
    public FilteredClassifier getClassifier() {
        return this.classifier;
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
