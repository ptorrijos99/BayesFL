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
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.estimators.DiscreteEstimator;

import java.util.List;
import java.util.Map;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.privacy.DenoisableModel;
import bayesfl.privacy.NoiseGenerator;
import weka.estimators.Estimator;

import static bayesfl.experiments.utils.ExperimentUtils.*;

/**
 * A class representing naive Bayes.
 */
public class PT implements DenoisableModel {

    /**
     * The list of classifiers (can be only one, for example for Naive Bayes).
     */
    public List<AbstractClassifier> ensemble;

    /**
     * The list of generated indexes used for AnDE with n>0.
     */
    public List<int[]> combinations;

    /**
     * The list of generated classes used for AnDE with n>0.
     */
    public final List<Map<String, Integer>> syntheticClassMaps;

    /**
     * The header for the file.
     */
    private String header = "bbdd,id,cv,algorithm,bins,seed,nClients,fusParams,fusProbs,dptype,epsilon,delta,rho,sensitivity,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

    /**
     * Constructor
     *
     * @param ensemble The list of classifiers (can be only one, for example for Naive Bayes).
     * @param combinations The list of generated classes used for AnDE with n>0.
     */
    public PT(List<AbstractClassifier> ensemble, List<int[]> combinations, List<Map<String, Integer>> syntheticClassMaps) {
        this.ensemble = ensemble;
        this.combinations = combinations;
        this.syntheticClassMaps = syntheticClassMaps;
    }

    /**
     * Applies differential privacy noise to the internal probabilistic model.
     * <p>
     * This method iterates over all {@link FilteredClassifier} instances in the ensemble.
     * If the underlying base classifier is a {@link NaiveBayes}, it accesses both the
     * class distribution and the attribute-conditional distributions, and applies the
     * specified {@link NoiseGenerator} to perturb their probabilities.
     * </p>
     * <p>
     * The noise is added in a way that preserves the estimator structure by using the
     * {@code addValue(i, delta)} method, ensuring Weka maintains consistency.
     * </p>
     *
     * @param noise the {@link NoiseGenerator} used to apply noise (e.g., Laplace, Gaussian, zCDP)
     */
    @Override
    public void applyNoise(NoiseGenerator noise) {
        for (AbstractClassifier classifier : ensemble) {
            if (classifier instanceof FilteredClassifier fc) {
                if (fc.getClassifier() instanceof NaiveBayes nb) {

                    // Privatize class distribution
                    DiscreteEstimator classDist = (DiscreteEstimator) nb.getClassEstimator();
                    int numClasses = classDist.getNumSymbols(); // Get the number of classes
                    double[] originalClassProbs = new double[numClasses];
                    applyNoiseToEstimator(noise, classDist, numClasses, originalClassProbs);

                    // Privatize conditional estimators (conditional probabilities for each attribute given class)
                    Estimator[][] conds = nb.getConditionalEstimators();
                    for (Estimator[] cond : conds) {
                        for (Estimator est : cond) {
                            if (est instanceof DiscreteEstimator de) {
                                int n = de.getNumSymbols();
                                double[] original = new double[n];
                                applyNoiseToEstimator(noise, de, n, original);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Applies differential privacy noise to a DiscreteEstimator (e.g., class distribution or conditional probabilities).
     * <p>
     * This method retrieves the original probabilities from the estimator, privatizes them by applying noise,
     * and then updates the estimator with the privatized values.
     * </p>
     *
     * @param noise             the noise generator to apply to the probabilities
     * @param classDist         the DiscreteEstimator containing class probabilities (or conditional probabilities)
     * @param numClasses        the number of classes (or symbols)
     * @param originalClassProbs an array to store the original probabilities before noise is added
     */
    private void applyNoiseToEstimator(NoiseGenerator noise, DiscreteEstimator classDist, int numClasses, double[] originalClassProbs) {
        for (int i = 0; i < numClasses; i++) {
            originalClassProbs[i] = classDist.getProbability(i); // Get original probabilities
        }
        double[] noisyClassProbs = noise.privatize(originalClassProbs); // Add noise
        for (int i = 0; i < numClasses; i++) {
            classDist.addValue(i, noisyClassProbs[i] - originalClassProbs[i]);  // Update the class distribution
        }
    }


    /**
     * Gets the model.
     * 
     * @return The model.
     */
    public Object getModel() {
        return this.ensemble;
    }

    /**
     * Sets the model.
     * 
     * @param model The model.
     */
    public void setModel(Object model) {
        if (model instanceof List) {
            // Check if the model is a list of classifiers
            for (Object obj : (List<?>) model) {
                if (!(obj instanceof AbstractClassifier)) {
                    throw new UnsupportedOperationException("Model must be a list of classifiers");
                }
            }
            this.ensemble = (List<AbstractClassifier>) model;
        } else {
            throw new UnsupportedOperationException("Model must be a list of classifiers");
        }
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

        int maxIterations = 0;
        statistics += maxIterations + ",";

        metrics = getClassificationMetricsEnsemble(ensemble, syntheticClassMaps, train);
        statistics += metrics;

        metrics = getClassificationMetricsEnsemble(ensemble, syntheticClassMaps, test);
        statistics += metrics;

        statistics += time + "\n";

        saveExperiment("results/" + epoch + "/" + path, header, statistics);
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
