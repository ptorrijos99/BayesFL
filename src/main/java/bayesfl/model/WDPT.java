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
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import static bayesfl.experiments.utils.ExperimentUtils.getClassificationMetrics;
import static bayesfl.experiments.utils.ExperimentUtils.saveExperiment;

/**
 * A class representing a parameters tree model.
 */
public class WDPT implements Model {

    /**
     * The tree-based parameter storage.
     */
    private wdBayesParametersTree tree;

    /**
     * The classifier.
     */
    private FilteredClassifier classifier;

    /**
     * The header for the file.
     */
    private String header = "bbdd,id,cv,algorithm,bins,seed,nClients,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

    /**
     * Constructor.
     *
     * @param tree The tree-based parameter storage.
     * @param classifier The classifier.
     */
    public WDPT(wdBayesParametersTree tree, FilteredClassifier classifier) {
        this.tree = tree;
        this.classifier = classifier;
    }

    /**
     * Gets the model. In this case, the model is the tree-based parameter storage.
     * 
     * @return The model.
     */
    public wdBayesParametersTree getModel() {
        return this.tree;
    }

    /**
     * Gets the classifier.
     * 
     * @return The classifier.
     */
    public FilteredClassifier getClassifier() {
        return this.classifier;
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
     * @param operation The operation.
     * @param epoch The epoch.
     * @param path The path.
     * @param nClients The number of clients.
     * @param id The identifier.
     * @param data The data.
     * @param iteration The iteration.
     * @param time The time.
     */
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        Weka_Instances weka = (Weka_Instances) data;
        Instances train = weka.getTrain();
        Instances test = weka.getTest();

        String statistics = "";
        String metrics;

        String bbdd = data.getName();
        int instances = train.numInstances();
        statistics += bbdd + "," + id + "," + operation + "," + epoch + "," + iteration + "," + instances + ",";

        wdBayes algorithm = (wdBayes) this.classifier.getClassifier();
        int maxIterations = algorithm.getM_MaxIterations();
        statistics += maxIterations + ",";

        metrics = getClassificationMetrics(this.classifier, train);
        statistics += metrics;

        metrics = getClassificationMetrics(this.classifier, test);
        statistics += metrics;

        statistics += time + "\n";

        String output = "results/ " + epoch + "/" + path;
        saveExperiment(output, header, statistics);
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
