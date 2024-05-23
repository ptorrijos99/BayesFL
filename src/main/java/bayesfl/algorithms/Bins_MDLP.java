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
import weka.core.Instances;
import weka.filters.Filter;

/**
 * Local application imports.
 */
import bayesfl.model.Bins;
import bayesfl.data.Data;
import bayesfl.model.Model;

/**
 * A class representing a discrezation algorithm.
 */
public class Bins_MDLP implements LocalAlgorithm {

    /**
     * The options for the discretization method.
     */
    private String[] options;

    /**
     * The discretization method.
     */
    private Filter discretizer;

    /**
     * The name of the algorithm.
     */
    private String algorithmName = "Bins_MDLP";

    /**
     * The name of the refinement method.
     */
    private String refinement = "None";

    /**
     * Whether the algorithm is supervised or not.
     */
    private boolean supervised;

    /**
     * Constructor.
     */
    public Bins_MDLP(boolean supervised, String[] options) {
        this.options = options.clone();
        this.supervised = supervised;

        if (supervised) {
            this.discretizer = new weka.filters.supervised.attribute.Discretize();
        } else {
            this.discretizer = new weka.filters.unsupervised.attribute.Discretize();
        }

        try {
            this.discretizer.setOptions(this.options);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Builds a local model using the provided data.
     * 
     * @param data The data to use.
     * @return The built local model.
     */
    public Model buildLocalModel(Data data) {
        Instances instances = (Instances) data.getData();

        try {
            this.discretizer.setInputFormat(instances);
            Filter.useFilter(instances, this.discretizer);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Substract the class attribute from the number of attributes
        int numAttributes = instances.numAttributes() - 1;
        double[][] cutPoints = new double[numAttributes][];
    
        for (int i = 0; i < numAttributes; i++) {
            if (this.supervised) {
                weka.filters.supervised.attribute.Discretize discretizer = (weka.filters.supervised.attribute.Discretize) this.discretizer;
                cutPoints[i] = discretizer.getCutPoints(i);
            } else {
                weka.filters.unsupervised.attribute.Discretize discretizer = (weka.filters.unsupervised.attribute.Discretize) this.discretizer;
                cutPoints[i] = discretizer.getCutPoints(i);
            }
        }
        return new Bins(cutPoints);
    }

    /**
     * Builds a local model using the provided data and existing local model.
     * 
     * @param localModel The existing local model.
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
        return this.refinement;
    }
}
