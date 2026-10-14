/*
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2026 Universidad de Castilla-La Mancha, España
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

import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import weka.core.Instances;

import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import static bayesfl.experiments.utils.ExperimentUtils.saveExperiment;

/**
 * AnDE ensemble of SPODEs with class-conditional discriminative parameters
 * (dCCBN) or discriminative weights on generative tables (wCCBN), predicted with
 * the softmax restricted to the synthetic classes compatible with the observed
 * parents (see {@link SpodeStructure}).
 * <p>
 * A model only holds one parameter vector per SPODE and, for wCCBN, the tables the
 * weights multiply. When {@link #logTables} is {@code null} for a wCCBN model the
 * tables are the client's local ones, recomputed from its training data when needed,
 * so no client keeps a copy of them between rounds.
 * </p>
 */
public class SPODE_Model implements Model {

    /** The global layout of the SPODEs. */
    public final SpodeStructure structure;

    /** Parameters of each SPODE. */
    public final double[][] params;

    /** Feature tables of each SPODE for wCCBN (null: local tables, or dCCBN). */
    public final double[][] logTables;

    /** Raw counts released in round 1 to build global tables (null otherwise). */
    public final double[][] counts;

    /** Whether the parameters are wCCBN weights (true) or dCCBN parameters (false). */
    public final boolean weighted;

    /** Whether the class feature of wCCBN is log P(c) (Zaidi) or P(c) (wdBayes). */
    public final boolean logPrior;

    /** Number of training instances behind the model. */
    private final int numInstances;

    /** L-BFGS iterations per round, written to the statistics. */
    private final int maxIterations;

    /**
     * Constructor.
     *
     * @param structure     The global layout of the SPODEs.
     * @param params        The parameters of each SPODE.
     * @param logTables     The feature tables of each SPODE (null: local tables or dCCBN).
     * @param counts        The raw counts released for global tables (null otherwise).
     * @param weighted      Whether the parameters are wCCBN weights.
     * @param logPrior      Whether the wCCBN class feature is log P(c).
     * @param numInstances  The number of training instances behind the model.
     * @param maxIterations The L-BFGS iterations per round.
     */
    public SPODE_Model(SpodeStructure structure, double[][] params, double[][] logTables, double[][] counts,
                       boolean weighted, boolean logPrior, int numInstances, int maxIterations) {
        this.structure = structure;
        this.params = params;
        this.logTables = logTables;
        this.counts = counts;
        this.weighted = weighted;
        this.logPrior = logPrior;
        this.numInstances = numInstances;
        this.maxIterations = maxIterations;
    }

    /**
     * Counts of a SPODE over encoded data, in the parameter layout: synthetic class counts first,
     * then the (class, attribute value) counts. A missing feature counts as value 0, as in wdBayes.
     *
     * @param st The SPODE layout.
     * @param s  The SPODE.
     * @param X  The encoded instances.
     * @param yS The synthetic class of each instance in this SPODE.
     * @return The counts.
     */
    public static double[] counts(SpodeStructure st, int s, int[][] X, int[] yS) {
        double[] cnt = new double[st.np[s]];
        for (int i = 0; i < X.length; i++) {
            int c = yS[i];
            if (c < 0) continue;
            cnt[c]++;
            for (int u = 0; u < st.nAttributes; u++) {
                cnt[st.index(s, c, u, SpodeStructure.feature(X[i], u))]++;
            }
        }
        return cnt;
    }

    /**
     * Turns counts into the feature tables of wCCBN, in place: m-estimates (m = 1) as in
     * wdBayes, conditionals in log scale clamped at 1e-75, class prior in log or linear scale.
     *
     * @param st       The SPODE layout.
     * @param s        The SPODE.
     * @param cnt      The counts (overwritten).
     * @param logPrior Whether the class prior is in log scale.
     * @return The tables ({@code cnt}).
     */
    public static double[] toTables(SpodeStructure st, int s, double[] cnt, boolean logPrior) {
        int nc = st.nc[s];
        double total = 0;
        for (int c = 0; c < nc; c++) total += cnt[c];
        for (int u = 0; u < st.nAttributes; u++) {
            int k = st.card[u];
            for (int c = 0; c < nc; c++) {
                int base = st.index(s, c, u, 0);
                double denom = 0;
                for (int v = 0; v < k; v++) denom += cnt[base + v];
                for (int v = 0; v < k; v++) {
                    cnt[base + v] = Math.log(Math.max((cnt[base + v] + 1.0 / k) / (denom + 1.0), 1e-75));
                }
            }
        }
        for (int c = 0; c < nc; c++) {
            double p = (cnt[c] + 1.0 / nc) / (total + 1.0);
            cnt[c] = logPrior ? Math.log(p) : p;
        }
        return cnt;
    }

    /**
     * Adds the class distribution of SPODE {@code s} for an encoded instance to {@code votes}.
     *
     * @param s      The SPODE.
     * @param p      The parameters of the SPODE.
     * @param f      The feature tables of the SPODE (null for dCCBN).
     * @param x      The encoded instance.
     * @param votes  The class votes, updated in place.
     * @param buffer Scratch space of at least {@code nc[s]} values.
     */
    void addVotes(int s, double[] p, double[] f, int[] x, double[] votes, double[] buffer) {
        SpodeStructure st = structure;
        int[] cs = st.compatible(s, x);
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < cs.length; j++) {
            int c = cs[j];
            double v = f == null ? p[c] : p[c] * f[c];
            for (int u = 0; u < st.nAttributes; u++) {
                int q = st.index(s, c, u, SpodeStructure.feature(x, u));
                v += f == null ? p[q] : p[q] * f[q];
            }
            buffer[j] = v;
            if (v > max) max = v;
        }
        double z = 0;
        for (int j = 0; j < cs.length; j++) {
            buffer[j] = Math.exp(buffer[j] - max);
            z += buffer[j];
        }
        for (int j = 0; j < cs.length; j++) {
            votes[st.synthY[s][cs[j]]] += buffer[j] / z;
        }
    }

    /**
     * Class distribution of every instance of a dataset: the average over SPODEs of their
     * restricted posteriors. SPODEs run in parallel in fixed chunks summed in order, so the
     * result does not depend on the number of threads. For a wCCBN model without tables, each
     * SPODE uses the local tables of {@code train}, computed one SPODE at a time.
     *
     * @param data  The instances to predict.
     * @param train The client's training data (local tables), or null.
     * @return The class distribution of every instance.
     */
    public double[][] distributions(Instances data, Instances train) {
        SpodeStructure st = structure;
        int n = data.numInstances();
        int[][] X = new int[n][];
        for (int i = 0; i < n; i++) X[i] = st.encode(data.instance(i));

        int[][] trainX = null;
        int[] trainY = null;
        if (weighted && logTables == null) {
            if (train == null) throw new IllegalStateException("wCCBN model without tables needs the training data");
            trainX = new int[train.numInstances()][];
            trainY = new int[train.numInstances()];
            for (int i = 0; i < trainX.length; i++) {
                trainX[i] = st.encode(train.instance(i));
                trainY[i] = (int) train.instance(i).classValue();
            }
        }
        final int[][] tX = trainX;
        final int[] tY = trainY;

        int m = st.nSpodes();
        int chunks = Math.min(m, 64);
        double[][][] partial = new double[chunks][][];
        try {
            SpodeStructure.pool().submit(() -> IntStream.range(0, chunks).parallel().forEach(k -> {
                double[][] votes = new double[n][st.nClasses];
                double[] buffer = new double[maxNc()];
                for (int s = k; s < m; s += chunks) {
                    double[] f = null;
                    if (weighted) {
                        f = logTables != null ? logTables[s] : localTables(s, tX, tY);
                    }
                    for (int i = 0; i < n; i++) addVotes(s, params[s], f, X[i], votes[i], buffer);
                }
                partial[k] = votes;
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }

        double[][] dist = new double[n][st.nClasses];
        for (int k = 0; k < chunks; k++) {
            for (int i = 0; i < n; i++) {
                for (int y = 0; y < st.nClasses; y++) dist[i][y] += partial[k][i][y];
            }
        }
        for (double[] d : dist) {
            double z = 0;
            for (double v : d) z += v;
            if (z > 0) for (int y = 0; y < d.length; y++) d[y] /= z;
        }
        return dist;
    }

    /** Local tables of SPODE {@code s} from encoded training data. */
    private double[] localTables(int s, int[][] X, int[] y) {
        int[] yS = new int[X.length];
        for (int i = 0; i < X.length; i++) yS[i] = structure.synthClass(s, X[i], y[i]);
        return toTables(structure, s, counts(structure, s, X, yS), logPrior);
    }

    /** @return the largest number of synthetic classes of any SPODE. */
    private int maxNc() {
        int max = 1;
        for (int v : structure.nc) max = Math.max(max, v);
        return max;
    }

    /**
     * Metrics in the format of {@code ExperimentUtils.getClassificationMetricsEnsemble}:
     * accuracy, macro precision, recall and F1 (0 when undefined, as with Weka), log-loss,
     * Brier score and evaluation time.
     *
     * @param data  The instances to evaluate.
     * @param train The client's training data (local tables), or null.
     * @return The comma-terminated metrics.
     */
    public String metrics(Instances data, Instances train) {
        long start = System.currentTimeMillis();
        int nY = structure.nClasses;
        int n = data.numInstances();
        double[][] dist = distributions(data, train);

        double[][] confusion = new double[nY][nY];
        double logLoss = 0, brier = 0;
        for (int i = 0; i < n; i++) {
            int y = (int) data.instance(i).classValue();
            double[] p = dist[i];
            int pred = 0;
            for (int k = 1; k < nY; k++) if (p[k] > p[pred]) pred = k;
            confusion[y][pred]++;
            logLoss -= Math.log(Math.max(Math.min(p[y], 1 - 1e-15), 1e-15));
            for (int k = 0; k < nY; k++) {
                double t = k == y ? 1 : 0;
                brier += (p[k] - t) * (p[k] - t);
            }
        }

        double correct = 0, precision = 0, recall = 0, f1 = 0;
        for (int k = 0; k < nY; k++) {
            correct += confusion[k][k];
            double predicted = 0, actual = 0;
            for (int j = 0; j < nY; j++) {
                predicted += confusion[j][k];
                actual += confusion[k][j];
            }
            double pr = predicted > 0 ? confusion[k][k] / predicted : 0;
            double rc = actual > 0 ? confusion[k][k] / actual : 0;
            precision += pr;
            recall += rc;
            f1 += pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0;
        }
        double time = (System.currentTimeMillis() - start) / 1000.0;
        return correct / n + "," + precision / nY + "," + recall / nY + "," + f1 / nY + ","
                + logLoss / n + "," + brier / n + "," + time + ",";
    }

    @Override
    public Object getModel() {
        return this;
    }

    @Override
    public void setModel(Object model) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /**
     * Saves the statistics with the columns of {@link WDPT#HEADER}.
     */
    @Override
    public void saveStats(String operation, String epoch, String path, int nClients, int id, Data data, int iteration, double time) {
        Weka_Instances weka = (Weka_Instances) data;
        Instances train = weka.getTrain();
        Instances test = weka.getTest();

        String statistics = data.getName() + "," + id + "," + operation + "," + epoch + "," + iteration + ","
                + train.numInstances() + "," + maxIterations + ","
                + metrics(train, train) + metrics(test, train) + time + "\n";

        saveExperiment("results/" + epoch + "/" + path, WDPT.HEADER, statistics);
    }

    @Override
    public double getScore() {
        throw new UnsupportedOperationException("Method not implemented");
    }

    @Override
    public double getScore(Data data) {
        throw new UnsupportedOperationException("Method not implemented");
    }

    /** @return the number of training instances behind the model. */
    public int getNumInstances() {
        return numInstances;
    }

    /** @return the L-BFGS iterations per round. */
    public int getMaxIterations() {
        return maxIterations;
    }
}
