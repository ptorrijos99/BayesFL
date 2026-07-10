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
 *    mSP1DE_CG.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 *    Conditional-Gaussian (HAODE-style) variant of {@link mSP1DE}.
 *
 *    The structure is identical to the discrete mSP1DE: a single super-parent
 *    Xi and a set of children. The super-parent Xi is kept DISCRETIZED (same as
 *    the discrete model, so the tree structure and the federated structural
 *    union are untouched). The difference is purely in how each child Xj is
 *    parameterised:
 *
 *      - numeric child  -> P(Xj | y, Xi_disc) is a Gaussian density whose mean
 *                          and variance are estimated per (class y, Xi state).
 *      - nominal child  -> P(Xj | y, Xi_disc) is a contingency table with
 *                          Laplace smoothing, exactly as in the discrete model.
 *
 *    This makes the model reduce to the discrete one when every child is nominal,
 *    and lets it work on mixed data sets.
 */

package org.albacete.simd.mAnDE;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import bayesfl.privacy.NumericNoiseGenerator;
import weka.core.Instance;
import weka.core.Utils;

public class mSP1DE_CG implements mSPnDE, Serializable {

    private static final long serialVersionUID = 1L;

    /** Lower bound for any variance, to avoid degenerate (zero-variance) cells. */
    private static final double MIN_VARIANCE = 1e-6;

    /** ID of the (discretized) super-parent of this SP1DE. */
    private final int xi;

    /** Children of this SP1DE. */
    private final Set<Integer> listChildren;

    /** Joint P(y, Xi_disc), shared by all children. Indexed [y][xiState]. */
    private double[][] globalProb;

    /** Children that are numeric in the ORIGINAL (non-discretized) data. */
    private final Set<Integer> numericChildren = ConcurrentHashMap.newKeySet();

    /** Gaussian means per numeric child. Indexed children -> [y][xiState]. */
    private final HashMap<Integer, double[][]> mean = new HashMap<>();

    /** Gaussian variances per numeric child. Indexed children -> [y][xiState]. */
    private final HashMap<Integer, double[][]> var = new HashMap<>();

    /**
     * Conditional tables P(Xj | y, Xi_disc) for nominal children, with Laplace
     * smoothing. Indexed children -> [y][xiState][xjValue].
     */
    private final HashMap<Integer, double[][][]> nominalTables = new HashMap<>();

    /**
     * Constructor.
     *
     * @param xi the (discretized) super-parent variable.
     */
    public mSP1DE_CG(int xi) {
        this.xi = xi;
        this.listChildren = ConcurrentHashMap.newKeySet();
    }

    /**
     * Builds the parameters of this SP1DE from the model's training data.
     *
     * The super-parent Xi and the class y are read from the discretized data
     * ({@code model.data}); numeric children are read from the original numeric
     * data ({@code model.dataOriginal}); nominal children are read from the
     * discretized data.
     */
    @Override
    public void buildTables(mAnDE model) {
        int nClass = model.classNumValues;
        int nXi = model.varNumValues[xi];

        // ---- Joint P(y, Xi_disc) with Laplace smoothing (alpha = 1) ----
        globalProb = new double[nClass][nXi];
        for (int i = 0; i < model.numInstances; i++) {
            Instance inst = model.data.get(i);
            globalProb[(int) inst.value(model.y)][(int) inst.value(xi)] += 1;
        }
        double globalDenom = model.numInstances + (double) nClass * nXi;
        for (int y = 0; y < nClass; y++) {
            for (int j = 0; j < nXi; j++) {
                globalProb[y][j] = (globalProb[y][j] + 1.0) / globalDenom;
            }
        }

        // ---- Per-child parameters ----
        String hybrid = model.getCgHybrid();
        boolean doHybrid = !"none".equals(hybrid);
        double d0 = model.getCgPriorVarDof();

        // Split children into originally-nominal vs numeric candidates.
        Set<Integer> numericCandidates = new LinkedHashSet<>();
        for (int xj : listChildren) {
            if (model.dataOriginal.attribute(xj).isNumeric()) {
                numericCandidates.add(xj);
            } else {
                nominalTables.put(xj, new double[nClass][nXi][model.varNumValues[xj]]);
            }
        }

        // Accumulators: Gaussian sufficient statistics for every numeric child,
        // plus (only when a hybrid strategy is active) discrete counts on the
        // supervised-DISCRETIZED value of the same child.
        HashMap<Integer, double[][]> cnt = new HashMap<>();
        HashMap<Integer, double[][]> sum = new HashMap<>();
        HashMap<Integer, double[][]> sumSq = new HashMap<>();
        HashMap<Integer, double[][][]> numDisc = new HashMap<>();
        for (int xj : numericCandidates) {
            cnt.put(xj, new double[nClass][nXi]);
            sum.put(xj, new double[nClass][nXi]);
            sumSq.put(xj, new double[nClass][nXi]);
            if (doHybrid) numDisc.put(xj, new double[nClass][nXi][model.varNumValues[xj]]);
        }

        // Single pass over the data accumulating everything.
        for (int i = 0; i < model.numInstances; i++) {
            Instance instD = model.data.get(i);          // discretized
            Instance instO = model.dataOriginal.get(i);  // original numeric
            int y = (int) instD.value(model.y);
            int xiVal = (int) instD.value(xi);

            for (int xj : listChildren) {
                if (numericCandidates.contains(xj)) {
                    double v = instO.value(xj);
                    cnt.get(xj)[y][xiVal] += 1.0;
                    sum.get(xj)[y][xiVal] += v;
                    sumSq.get(xj)[y][xiVal] += v * v;
                    if (doHybrid) numDisc.get(xj)[y][xiVal][(int) instD.value(xj)] += 1.0;
                } else {
                    nominalTables.get(xj)[y][xiVal][(int) instD.value(xj)] += 1.0;
                }
            }
        }

        // Finalise the Gaussian parameters of every numeric candidate.
        HashMap<Integer, double[][]> gMean = new HashMap<>();
        HashMap<Integer, double[][]> gVar = new HashMap<>();
        for (int xj : numericCandidates) {
            double[][][] mv = finalizeGaussian(cnt.get(xj), sum.get(xj), sumSq.get(xj),
                    nClass, nXi, d0);
            gMean.put(xj, mv[0]);
            gVar.put(xj, mv[1]);
        }

        // Route each numeric child to a Gaussian density or a discretized table.
        for (int xj : numericCandidates) {
            double[][][] discTable = null;
            if (doHybrid) {
                discTable = numDisc.get(xj);
                laplaceNormalize(discTable);
            }
            boolean useDiscrete = switch (hybrid) {
                case "alldisc"  -> true;
                // All tree-selected children have >=2 bins, so the informative cut
                // is at >=3: >=3 bins = multimodal/non-monotone structure a unimodal
                // Gaussian cannot capture (use bins); exactly 2 bins = a single
                // threshold the Gaussian handles fine.
                case "manybins" -> model.varNumValues[xj] >= 3;
                case "ll"       -> discreteBeatsGaussian(model, xj,
                        gMean.get(xj), gVar.get(xj), discTable);
                default         -> false; // "none"
            };
            if (useDiscrete) {
                nominalTables.put(xj, discTable);   // already normalised
            } else {
                numericChildren.add(xj);
                mean.put(xj, gMean.get(xj));
                var.put(xj, gVar.get(xj));
            }
        }

        // Finalise the ORIGINALLY-nominal children (raw counts -> Laplace tables);
        // numeric-as-discrete tables routed above are already normalised.
        for (int xj : listChildren) {
            if (!numericCandidates.contains(xj)) {
                laplaceNormalize(nominalTables.get(xj));
            }
        }
    }

    /**
     * Finalises the per-cell Gaussian parameters of a numeric child with
     * empirical-Bayes (limma / regularized-DDA style) variance shrinkage:
     * {@code s2_shrunk = (d0*pooledVar + (n-1)*s2_cell)/(d0+(n-1))}, pooledVar
     * being the child's pooled within-cell variance. Rescues the small-N /
     * high-dimensional regime; {@code d0=0} recovers the per-cell MLE. Returns
     * {@code {mean, var}}.
     */
    private double[][][] finalizeGaussian(double[][] c, double[][] sx, double[][] sxx,
                                          int nClass, int nXi, double d0) {
        double[][] m = new double[nClass][nXi];
        double[][] s2raw = new double[nClass][nXi];
        double pooledNum = 0.0, pooledDen = 0.0;
        for (int y = 0; y < nClass; y++) {
            for (int x = 0; x < nXi; x++) {
                double n = c[y][x];
                if (n >= 1.0) {
                    double mu = sx[y][x] / n;
                    m[y][x] = mu;
                    if (n > 1.0) {
                        double v = (sxx[y][x] - n * mu * mu) / (n - 1.0);
                        s2raw[y][x] = v;
                        pooledNum += (n - 1.0) * v;
                        pooledDen += (n - 1.0);
                    } else {
                        s2raw[y][x] = Double.NaN; // single sample: no variance
                    }
                } else {
                    m[y][x] = Double.NaN;     // empty cell, resolved below
                    s2raw[y][x] = Double.NaN;
                }
            }
        }
        double pooledVar = (pooledDen > 0.0) ? Math.max(pooledNum / pooledDen, MIN_VARIANCE) : 1.0;

        // Empty-cell MEANS back off to the per-CLASS marginal mean P(Xj|y).
        resolveEmptyMeans(m, c, sx);

        double[][] s2 = new double[nClass][nXi];
        for (int y = 0; y < nClass; y++) {
            for (int x = 0; x < nXi; x++) {
                double cellDof = Math.max(c[y][x] - 1.0, 0.0);
                double cellVar = (cellDof > 0.0 && !Double.isNaN(s2raw[y][x])) ? s2raw[y][x] : pooledVar;
                double denom = d0 + cellDof;
                double sh = (denom > 0.0) ? (d0 * pooledVar + cellDof * cellVar) / denom : pooledVar;
                s2[y][x] = Math.max(sh, MIN_VARIANCE);
            }
        }
        return new double[][][]{m, s2};
    }

    /** Laplace-smooths (alpha=1) and normalises a [y][xi][value] count table in place. */
    private static void laplaceNormalize(double[][][] table) {
        for (double[][] tY : table) {
            for (double[] tYX : tY) {
                int nXj = tYX.length;
                double denom = Utils.sum(tYX) + nXj;
                for (int k = 0; k < nXj; k++) {
                    tYX[k] = (tYX[k] + 1.0) / denom;
                }
            }
        }
    }

    /**
     * "ll" hybrid criterion: returns true iff, on the training data, the
     * discretized table gives a LOWER single-child class-posterior log-loss than
     * the Gaussian density for child {@code xj}. The posteriors are normalised
     * over classes, so the comparison is unit-consistent (density vs probability
     * scales cancel).
     */
    private boolean discreteBeatsGaussian(mAnDE model, int xj, double[][] m,
                                          double[][] s2, double[][][] discTable) {
        int nClass = globalProb.length;
        double gLoss = 0.0, dLoss = 0.0;
        double[] lg = new double[nClass];
        double[] ld = new double[nClass];
        for (int i = 0; i < model.numInstances; i++) {
            Instance instD = model.data.get(i);
            Instance instO = model.dataOriginal.get(i);
            int yTrue = (int) instD.value(model.y);
            int xiVal = (int) instD.value(xi);
            double v = instO.value(xj);
            int bin = (int) instD.value(xj);
            for (int y = 0; y < nClass; y++) {
                double base = Math.log(globalProb[y][xiVal]);
                lg[y] = base + logGaussian(v, m[y][xiVal], s2[y][xiVal]);
                ld[y] = base + Math.log(discTable[y][xiVal][bin]);
            }
            gLoss -= logPosteriorTrue(lg, yTrue);
            dLoss -= logPosteriorTrue(ld, yTrue);
        }
        return dLoss < gLoss;
    }

    /** log(softmax(logits)[yTrue]) = logits[yTrue] - logsumexp(logits). */
    private static double logPosteriorTrue(double[] logits, int yTrue) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) if (l > max) max = l;
        double s = 0.0;
        for (double l : logits) s += Math.exp(l - max);
        return logits[yTrue] - (max + Math.log(s));
    }

    /**
     * Replaces empty-cell (NaN) Gaussian means by a hierarchical backoff: first
     * the per-class marginal mean P(Xj|y) (over all Xi states of that class),
     * and only if the class itself has no observations the global marginal mean.
     * Backing off to the class mean (rather than the global mean) preserves
     * class-discriminative information when conditioning on Xi fragments a small
     * minority class into near-empty cells. The corresponding variance is
     * supplied by the pooled shrinkage target in {@link #buildTables}.
     */
    private void resolveEmptyMeans(double[][] m, double[][] cnt, double[][] sum) {
        int nClass = m.length, nXi = m[0].length;
        double[] classSum = new double[nClass];
        double[] classN = new double[nClass];
        double gSum = 0.0, gN = 0.0;
        for (int y = 0; y < nClass; y++) {
            for (int x = 0; x < nXi; x++) {
                classSum[y] += sum[y][x];
                classN[y] += cnt[y][x];
                gSum += sum[y][x];
                gN += cnt[y][x];
            }
        }
        double gMean = (gN > 0) ? gSum / gN : 0.0;
        for (int y = 0; y < nClass; y++) {
            double cMean = (classN[y] > 0) ? classSum[y] / classN[y] : gMean;
            for (int x = 0; x < nXi; x++) {
                if (Double.isNaN(m[y][x])) {
                    m[y][x] = cMean;
                }
            }
        }
    }

    /**
     * Computes the (normalised) class probabilities for an instance.
     *
     * Accumulation is done in log-space for numerical stability (microarray
     * SP1DEs can chain many densities whose product would otherwise under/overflow).
     *
     * @param inst  the DISCRETIZED test instance (used for y-dimension, Xi and
     *              nominal children).
     * @param model the model, used to reach the ORIGINAL numeric test instance
     *              ({@code model.currentNumericInstance}) for numeric children.
     */
    @Override
    public double[] probsForInstance(Instance inst, mAnDE model) {
        int nClass = globalProb.length;
        int xiVal = (int) inst.value(xi);
        Instance instO = model.currentNumericInstance;
        boolean studentT = model.isCgStudentT();
        double nu = model.getCgStudentDof();
        double priorW = model.getCgPriorWeight();
        double T = model.getCgTemperature();

        // Class prior log P(y, Xi), optionally re-weighted (priorW).
        double[] prior = new double[nClass];
        for (int y = 0; y < nClass; y++) {
            prior[y] = Math.log(globalProb[y][xiVal]);
            // globalProb = P(y)*P(Xi|y); adding (priorW-1)*log P(y) raises the
            // prior to the power priorW (priorW=1 empirical, priorW=0 uniform).
            if (priorW != 1.0) {
                double classPrior = 0.0;
                for (double pxi : globalProb[y]) classPrior += pxi;
                prior[y] += (priorW - 1.0) * Math.log(classPrior);
            }
        }

        // Summed child log-densities (the likelihood term).
        double[] lik = new double[nClass];
        for (int xj : listChildren) {
            if (numericChildren.contains(xj)) {
                double v = instO.value(xj);
                double[][] m = mean.get(xj);
                double[][] s2 = var.get(xj);
                for (int y = 0; y < nClass; y++) {
                    lik[y] += studentT
                            ? logStudentT(v, m[y][xiVal], s2[y][xiVal], nu)
                            : logGaussian(v, m[y][xiVal], s2[y][xiVal]);
                }
            } else {
                double[][][] table = nominalTables.get(xj);
                int xjVal = (int) inst.value(xj);
                for (int y = 0; y < nClass; y++) {
                    lik[y] += Math.log(table[y][xiVal][xjVal]);
                }
            }
        }

        // Temperature-scale the (over-confident) likelihood, then add the prior.
        double[] logp = new double[nClass];
        for (int y = 0; y < nClass; y++) {
            logp[y] = prior[y] + ((T == 1.0) ? lik[y] : lik[y] / T);
        }

        // Log-sum-exp normalisation.
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logp) if (v > max) max = v;
        double[] res = new double[nClass];
        double sum = 0.0;
        for (int y = 0; y < nClass; y++) {
            res[y] = Math.exp(logp[y] - max);
            sum += res[y];
        }
        if (sum != 0) {
            for (int y = 0; y < nClass; y++) res[y] /= sum;
        }
        return res;
    }

    /** Log of the Gaussian density N(x; mu, var). */
    private static double logGaussian(double x, double mu, double v) {
        double d = x - mu;
        return -0.5 * Math.log(2.0 * Math.PI * v) - (d * d) / (2.0 * v);
    }

    /**
     * Log of the Student-t density with location {@code mu}, scale^2 {@code v}
     * (the per-cell mean and shrunk variance) and {@code nu} degrees of freedom.
     * Terms that depend only on {@code nu} (the log-Gamma normaliser and the
     * {@code -0.5*log(nu*pi)} constant) are dropped: they are identical across
     * classes and cancel in the per-class log-sum-exp normalisation. Heavy tails
     * (small nu) bound the penalty paid by test outliers, unlike the Gaussian.
     */
    private static double logStudentT(double x, double mu, double v, double nu) {
        double d = x - mu;
        return -0.5 * Math.log(v) - 0.5 * (nu + 1.0) * Math.log(1.0 + (d * d) / (nu * v));
    }

    // ------------------------------------------------------------------
    // Federated parameter learning: TODO. The Gaussian sufficient statistics
    // (n, sum, sumSq) are additive across clients, so these will mirror the
    // discrete mSP1DE count-table aggregation. Left unimplemented for the
    // centralized-first milestone.
    // ------------------------------------------------------------------

    @Override
    public void buildCountTables(mAnDE model) {
        throw new UnsupportedOperationException(
                "mSP1DE_CG: federated count tables not implemented yet");
    }

    @Override
    public void addCounts(mSPnDE other) {
        throw new UnsupportedOperationException(
                "mSP1DE_CG: federated count aggregation not implemented yet");
    }

    @Override
    public void normalizeCounts() {
        throw new UnsupportedOperationException(
                "mSP1DE_CG: federated normalization not implemented yet");
    }

    @Override
    public void applyNoise(NumericNoiseGenerator noise) {
        throw new UnsupportedOperationException(
                "mSP1DE_CG: federated DP noise not implemented yet");
    }

    // ------------------------------------------------------------------
    // Structure handling (identical to mSP1DE).
    // ------------------------------------------------------------------

    @Override
    public void moreChildren(int child) {
        if ((child != -1) && !(child == xi)) {
            listChildren.add(child);
        }
    }

    @Override
    public void moreChildren(Collection<Integer> children) {
        children.forEach(this::moreChildren);
    }

    @Override
    public Set<Integer> getChildren() {
        return listChildren;
    }

    @Override
    public int getNChildren() {
        return listChildren.size();
    }

    @Override
    public mSPnDE copyDeep() {
        mSP1DE_CG copy = new mSP1DE_CG(this.xi);
        copy.listChildren.addAll(this.listChildren);
        return copy;
    }

    @Override
    public boolean hasProbTables() {
        return globalProb != null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof mSP1DE_CG that)) return false;
        return this.xi == that.xi;
    }

    @Override
    public int hashCode() {
        return this.xi;
    }

    @Override
    public String toString() {
        return "mSP1DE_CG{" + "xi=" + xi + ", listChildren=" + listChildren + '}';
    }
}
