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
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.estimators.Estimator;
import EBNC.wdBayes;

/**
 * Local application imports.
 */
import bayesfl.experiments.utils.ExperimentUtils;
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;

import static bayesfl.experiments.utils.ExperimentUtils.getClassificationMetrics;


/**
 * A class representing class-conditional Bayesian networks.
 */
public class PT implements Model {

    /**
     * The model parameters. Used in wdBayes (NBw, NBb and NBe algorithms).
     */
    private double[] parameters;

    /**
     * The class distribution. Used in WEKA NaiveBayes.
     */
    private Estimator m_ClassDistribution;

    /**
     * The distributions of the variables. Used in WEKA NaiveBayes.
     */
    private Estimator[][] m_Distributions;

    /**
     * The classifier.
     */
    private FilteredClassifier classifier;

    /**
     * The header for the file.
     */
    private String header = "bbdd,id,cv,algorithm,bins,seed,nClients,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

    /**
     * Constructor. wdBayes.
     *
     * @param parameters The parameters of the model.
     * @param classifier The classifier.
     */
    public PT(double[] parameters, FilteredClassifier classifier) {
        this.parameters = parameters;
        this.classifier = classifier;
    }

    /**
     * Constructor. WEKA NaiveBayes.
     *
     * @param m_ClassDistribution The class distribution.
     * @param m_Distributions The distributions of the variables.
     * @param classifier The classifier.
     */
    public PT(Estimator m_ClassDistribution, Estimator[][] m_Distributions, FilteredClassifier classifier) {
        this.m_ClassDistribution = m_ClassDistribution;
        this.m_Distributions = m_Distributions;
        this.classifier = classifier;
    }

    /**
     * Gets the model.
     * 
     * @return The model.
     */
    public double[] getModel() {
        return this.parameters;
    }

    /**
     * Gets the class distribution.
     * @return The class distribution.
     */
    public Estimator getM_ClassDistribution() {
        return this.m_ClassDistribution;
    }

    /**
     * Gets the distributions of the variables.
     * @return The distributions of the variables.
     */
    public Estimator[][] getM_Distributions() {
        return this.m_Distributions;
    }

    /**
     * Gets the classifier.
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

        Classifier classifier = this.classifier.getClassifier();

        String statistics = "";
        String metrics;

        String bbdd = data.getName();
        int instances = train.numInstances();
        statistics += bbdd + "," + id + "," + operation + "," + epoch + "," + iteration + "," + instances + ",";

        int maxIterations = 0;
        if (classifier instanceof wdBayes) {
            maxIterations = ((wdBayes) classifier).getM_MaxIterations();
        }
        statistics +=  + maxIterations + ",";

        metrics = getClassificationMetrics(this.classifier, train);
        statistics += metrics;

        metrics = getClassificationMetrics(this.classifier, test);
        statistics += metrics;

        statistics += time + "\n";

        ExperimentUtils.saveExperiment("results/" + epoch + "/" + path, header, statistics);
    }


    /**
     * Get the score of the model. This method is unused and throws an exception if called.
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
