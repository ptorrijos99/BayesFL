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
    private String header = "bbdd,id,cv,algorithm,bins,seed,nClients,fusParams,fusProbs,dptype,epsilon,delta,rho,sensitivity,autoSens,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

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
     * Applies differential privacy noise to the internal probabilistic model by perturbing discrete counts.
     * <p>
     * This method iterates over all {@link FilteredClassifier} instances in the ensemble. If the underlying
     * base classifier is a {@link NaiveBayes}, it accesses both the class distribution and the conditional
     * distributions for each attribute given the class. For each discrete estimator, it extracts raw counts,
     * applies Laplace noise scaled to the desired privacy budget, clips negative values, applies smoothing,
     * normalizes the resulting vector, and updates the estimator accordingly.
     * </p>
     * <p>
     * This approach ensures that the shared parameters satisfy ε-differential privacy while preserving the
     * structure expected by Weka classifiers.
     * </p>
     *
     * @param noise the {@link NoiseGenerator} used to apply noise (e.g., Laplace)
     */
    @Override
    public void applyNoise(NoiseGenerator noise) {
        for (AbstractClassifier classifier : ensemble) {
            if (classifier instanceof FilteredClassifier fc) {
                if (fc.getClassifier() instanceof NaiveBayes nb) {

                    // Privatize class distribution
                    DiscreteEstimator classDist = (DiscreteEstimator) nb.getClassEstimator();
                    applyNoiseToEstimator(noise, classDist);

                    // Privatize conditional estimators (conditional probabilities for each attribute given class)
                    Estimator[][] conds = nb.getConditionalEstimators();
                    for (Estimator[] cond : conds) {
                        for (Estimator est : cond) {
                            if (est instanceof DiscreteEstimator de) {
                                applyNoiseToEstimator(noise, de);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Applies differential privacy noise to a {@link DiscreteEstimator} by perturbing raw counts.
     * <p>
     * This method retrieves the original symbol counts from the estimator, adds Laplace noise scaled to
     * the specified privacy budget, clips negative values, applies Laplace smoothing, and normalizes the result
     * to obtain a valid probability distribution. The estimator is then updated to reflect the privatized distribution.
     * </p>
     *
     * @param noise          the {@link NoiseGenerator} that adds Laplace noise to the counts
     * @param estimator      the {@link DiscreteEstimator} containing the counts to privatize
     */
    void applyNoiseToEstimator(NoiseGenerator noise, DiscreteEstimator estimator) {
        int k = estimator.getNumSymbols();

        /* 1. original counts */
        double[] counts = new double[k];
        double oldSum = 0.0;
        for (int i = 0; i < k; i++) {
            counts[i] = estimator.getCount(i);
            oldSum   += counts[i];
        }

        /* 2. Laplace noise */
        double[] noisy = noise.privatize(counts);

        /* 3. clip + smoothing */
        double alpha = 1e-3;
        for (int i = 0; i < k; i++) {
            noisy[i] = Math.max(0.0, noisy[i]) + alpha;
        }

        /* 4. rescale to keep the same total mass */
        double newSum = 0.0;
        for (double v : noisy) newSum += v;
        double scale = oldSum / newSum;          // preserves magnitude expected by Weka
        for (int i = 0; i < k; i++) noisy[i] *= scale;

        /* 5. overwrite internal arrays via reflection */
        try {
            java.lang.reflect.Field fCounts = DiscreteEstimator.class.getDeclaredField("m_Counts");
            java.lang.reflect.Field fSum    = DiscreteEstimator.class.getDeclaredField("m_SumOfCounts");
            fCounts.setAccessible(true);
            fSum.setAccessible(true);

            double[] mCounts = (double[]) fCounts.get(estimator);
            System.arraycopy(noisy, 0, mCounts, 0, k);
            fSum.setDouble(estimator, oldSum);        // = sum(noisy) after rescale
        } catch (Exception e) {
            throw new RuntimeException("Unable to set privatized counts", e);
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
