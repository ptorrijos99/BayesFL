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
import bayesfl.privacy.NoiseGenerator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.estimators.DiscreteEstimator;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Local application imports.
 */
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.privacy.NumericDenoisableModel;
import bayesfl.privacy.NumericNoiseGenerator;
import weka.estimators.Estimator;

import static bayesfl.experiments.utils.ExperimentUtils.*;

/**
 * A class representing naive Bayes.
 */
public class PT implements NumericDenoisableModel {

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
     * Number of instances used to train this model.
     */
    private int numInstances = 0;

    /**
     * Header (0-row Instances) of the relabelled dataset of each ensemble member,
     * used to map parent-attribute value strings to value indices during the
     * deterministic reconstruction of parent estimators under DP.
     * Only set on client-built models (see PT_AnDE); fusion-built models keep null.
     */
    private List<Instances> headers = null;

    /**
     * Class index of the ORIGINAL dataset (before relabelling), needed to map
     * original parent-attribute indices to positions in the relabelled data.
     */
    private int classIndex = -1;

    /**
     * Whether this model's attributes are discretized on-the-fly by the
     * classifier's filter (cutPoints != null in PT_AnDE) rather than being
     * pre-discretized categorical data. Count-space DP requires the latter.
     */
    private boolean onTheFlyDiscretization = false;

    public void setPrivacyMetadata(List<Instances> headers, int classIndex, boolean onTheFlyDiscretization) {
        this.headers = headers;
        this.classIndex = classIndex;
        this.onTheFlyDiscretization = onTheFlyDiscretization;
    }

    /**
     * The header for the file.
     */
    private final String header = "bbdd,id,cv,algorithm,node,bins,seed,nClients,fusParams,fusProbs,dptype,epsilon,delta,rho,sensitivity,autoSens,alpha,epoch,iteration,instances,maxIterations,trAcc,trPr,trRc,trF1,trLogLoss,trBrier,trTime,teAcc,tePr,teRc,teF1,teLogLoss,teBrier,teTime,time\n";

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
     * Constructor with numInstances
     */
    public PT(List<AbstractClassifier> ensemble, List<int[]> combinations, List<Map<String, Integer>> syntheticClassMaps, int numInstances) {
        this(ensemble, combinations, syntheticClassMaps);
        this.numInstances = numInstances;
    }

    /**
     * Applies the Laplace mechanism to the generative release of the model.
     * <p>
     * For each SPnDE (a relabelled NB with synthetic class (y, x_P)), the
     * released statistics are the synthetic-class prior counts and the
     * conditional counts of the (a - n) NON-parent attributes; their joint
     * per-record L1 sensitivity is C(a,n)*(1+a-n) (Proposition 1). The n
     * parent-attribute tables are deterministic given the synthetic class, so
     * they are NOT noised nor released: they are rebuilt from the noisy class
     * prior (post-processing, no additional privacy cost), preserving the
     * near-indicator structure that the ensemble prediction relies on.
     * Each invocation consumes a fresh privacy budget: with nIterations > 1
     * the client re-counts and re-releases, so the total cost composes to
     * epsilon*T (the paper's generative runs use nIterations = 1).
     * </p>
     *
     * @param noise the {@link NoiseGenerator} used to apply noise (e.g., Laplace)
     */
    @Override
    public void applyNoise(NoiseGenerator noise) {
        if (!(noise instanceof NumericNoiseGenerator numericNoise)) {
            throw new IllegalArgumentException("Noise generator must be a NumericNoiseGenerator");
        }

        if (onTheFlyDiscretization) {
            throw new UnsupportedOperationException(
                    "Count-space DP requires pre-discretized categorical data (cutPoints must be null)");
        }

        for (int i = 0; i < ensemble.size(); i++) {
            if (!(ensemble.get(i) instanceof FilteredClassifier fc)
                    || !(fc.getClassifier() instanceof NaiveBayes nb)) {
                continue;
            }

            int[] parents = combinations.get(i);
            if (parents.length > 0 && (headers == null || classIndex < 0)) {
                throw new IllegalStateException(
                        "DP on AnDE (n>=1) requires privacy metadata; call setPrivacyMetadata at build time");
            }

            // 1. Privatize the synthetic-class prior counts
            DiscreteEstimator classDist = (DiscreteEstimator) nb.getClassEstimator();
            applyNoiseToEstimator(numericNoise, classDist);

            // 2. Privatize the conditional counts of NON-parent attributes only
            Set<Integer> parentAtts = new HashSet<>();
            for (int p : parents) {
                parentAtts.add(p < classIndex ? p : p - 1);
            }

            Estimator[][] conds = nb.getConditionalEstimators();
            for (int att = 0; att < conds.length; att++) {
                if (parentAtts.contains(att)) continue;
                for (Estimator est : conds[att]) {
                    if (est instanceof DiscreteEstimator de) {
                        applyNoiseToEstimator(numericNoise, de);
                    }
                }
            }

            // 3. Rebuild parent estimators from the noisy class prior
            if (parents.length > 0) {
                reconstructParentEstimators(nb, i);
            }
        }
    }

    /**
     * Rebuilds the parent-attribute estimators of ensemble member {@code i}
     * as deterministic functions of the NOISY class prior: for each synthetic
     * class value, all mass goes to the parent value encoded in its label.
     */
    private void reconstructParentEstimators(NaiveBayes nb, int i) {
        Instances header = headers.get(i);
        Map<String, Integer> classMap = syntheticClassMaps.get(i);
        int[] parents = combinations.get(i);
        DiscreteEstimator classDist = (DiscreteEstimator) nb.getClassEstimator();
        Estimator[][] conds = nb.getConditionalEstimators();

        for (Map.Entry<String, Integer> entry : classMap.entrySet()) {
            String[] parts = entry.getKey().split("\\|\\|\\|");
            int yIdx = entry.getValue();
            double noisyCount = Math.max(0.0, classDist.getCount(yIdx) - 1.0);

            for (int p = 0; p < parents.length; p++) {
                int att = parents[p] < classIndex ? parents[p] : parents[p] - 1;
                int vStar = header.attribute(att).indexOfValue(parts[p]);
                if (vStar < 0) {
                    throw new IllegalStateException("Parent value '" + parts[p]
                            + "' not found in attribute " + header.attribute(att).name()
                            + " — DP requires pre-discretized categorical data");
                }
                setDeterministicCounts((DiscreteEstimator) conds[att][yIdx], vStar, noisyCount);
            }
        }
    }

    /**
     * Overwrites a {@link DiscreteEstimator} so that value {@code vStar} carries
     * {@code mass} raw counts and every other value carries zero, keeping Weka's
     * +1 initialization convention (server recovers raw counts as getCount-1).
     */
    private void setDeterministicCounts(DiscreteEstimator estimator, int vStar, double mass) {
        int k = estimator.getNumSymbols();
        double[] counts = new double[k];
        double sum = 0.0;
        for (int v = 0; v < k; v++) {
            counts[v] = (v == vStar ? mass : 0.0) + 1.0;
            sum += counts[v];
        }
        try {
            java.lang.reflect.Field fCounts = DiscreteEstimator.class.getDeclaredField("m_Counts");
            java.lang.reflect.Field fSum = DiscreteEstimator.class.getDeclaredField("m_SumOfCounts");
            fCounts.setAccessible(true);
            fSum.setAccessible(true);
            System.arraycopy(counts, 0, (double[]) fCounts.get(estimator), 0, k);
            fSum.setDouble(estimator, sum);
        } catch (Exception e) {
            throw new RuntimeException("Unable to set deterministic counts", e);
        }
    }

    /**
     * Applies differential privacy noise to a {@link DiscreteEstimator} by perturbing raw counts.
     * <p>
     * The client perturbs only the raw counts (the +1 Weka adds at construction is stripped
     * before noising and re-added after, so {@link PT_Fusion_Server} can keep its existing
     * {@code getCount(i) - 1.0} convention to recover noisy raw counts). Per-client Laplace
     * smoothing (extra α) and total-mass rescaling are intentionally NOT applied here:
     * smoothing must happen only once, at the server, after aggregation — otherwise each
     * client biases the local estimate toward uniform and the bias accumulates with K.
     * </p>
     *
     * @param noise          the {@link NumericNoiseGenerator} that adds Laplace noise to the counts
     * @param estimator      the {@link DiscreteEstimator} containing the counts to privatize
     */
    void applyNoiseToEstimator(NumericNoiseGenerator noise, DiscreteEstimator estimator) {
        int k = estimator.getNumSymbols();

        /* 1. raw counts (undo Weka's built-in +1 init) */
        double[] raw = new double[k];
        for (int i = 0; i < k; i++) raw[i] = estimator.getCount(i) - 1.0;

        /* 2. Laplace noise on raw counts */
        double[] noisy = noise.privatize(raw);

        /* 3. clip negatives (consistency only — no smoothing α here) */
        for (int i = 0; i < k; i++) noisy[i] = Math.max(0.0, noisy[i]);

        /* 4. re-add Weka's +1 so the server's getCount(i)-1.0 still recovers the noisy count */
        double newSum = 0.0;
        for (int i = 0; i < k; i++) { noisy[i] += 1.0; newSum += noisy[i]; }

        /* 5. overwrite internal arrays via reflection */
        try {
            java.lang.reflect.Field fCounts = DiscreteEstimator.class.getDeclaredField("m_Counts");
            java.lang.reflect.Field fSum    = DiscreteEstimator.class.getDeclaredField("m_SumOfCounts");
            fCounts.setAccessible(true);
            fSum.setAccessible(true);

            double[] mCounts = (double[]) fCounts.get(estimator);
            System.arraycopy(noisy, 0, mCounts, 0, k);
            fSum.setDouble(estimator, newSum);
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
        System.out.println(statistics);

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

    /**
     * Gets the number of instances.
     * * @return The number of instances.
     */
    public int getNumInstances() {
        return this.numInstances;
    }
}
