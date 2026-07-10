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
 * Package containing algorithms related with differential privacy for
 * federated Bayesian networks.
 */
package bayesfl.algorithms.dp;

/**
 * Third party imports.
 */
import java.util.Random;

/**
 * Joint multi-tree full-batch DP-SGD. Each record's per-sample gradient is
 * clipped jointly across all trees to L2 norm C (so the release sensitivity is
 * C regardless of the number of SPnDE trees), Gaussian noise is added once per
 * step, and the data-independent L2-regularization gradient is added in the
 * clear before a fixed-learning-rate step.
 *
 * <p>The optimizer operates purely on {@code double[][]} parameter arrays and
 * the {@link SampleGradient} interface; it does not touch any wdBayes tree
 * object, and re-syncing a model tree with the final parameters is the
 * caller's responsibility.</p>
 */
public final class DPSGDOptimizer {

    /** Immutable DP-SGD configuration (clip norm, sigma, LR, step counts, seed). */
    private final DPSGDConfig cfg;

    /** Single shared RNG so noise draws are reproducible across the whole run. */
    private final Random rng;

    /**
     * @param cfg the DP-SGD configuration; also seeds the shared noise RNG.
     */
    public DPSGDOptimizer(DPSGDConfig cfg) {
        this.cfg = cfg;
        this.rng = new Random(cfg.seed());
    }

    /**
     * Per-sample clip scale {@code min(1, C / ||concat||_2)} across all trees.
     *
     * @param perTreeGrad per-tree gradient vectors for a single record.
     * @param C            the L2 clipping norm.
     * @return the scale factor to apply to every coordinate of every tree's gradient.
     */
    public static double clipScale(double[][] perTreeGrad, double C) {
        double sq = 0.0;
        for (double[] g : perTreeGrad) for (double x : g) sq += x * x;
        double norm = Math.sqrt(sq);
        return norm <= C ? 1.0 : C / norm;
    }

    /**
     * Performs {@code localSteps} full-batch joint DP-SGD steps, mutating
     * each {@code treeParams[t]} in place.
     *
     * @param treeParams     per-tree parameter vectors, updated in place.
     * @param treeGrads      per-tree per-sample gradient providers.
     * @param lambdaPerTree  per-tree L2 regularization coefficient (0 to disable).
     * @param numInstances   number of training instances (full batch size).
     */
    public void runRound(double[][] treeParams, SampleGradient[] treeGrads,
                          double[] lambdaPerTree, int numInstances) {
        int nTrees = treeParams.length;
        double C = cfg.clipC();
        double noiseStd = cfg.sigma() * C;

        for (int step = 0; step < cfg.localSteps(); step++) {
            double[][] acc = new double[nTrees][];
            for (int t = 0; t < nTrees; t++) acc[t] = new double[treeParams[t].length];

            // 1. Per-sample joint clip + accumulate.
            double[][] perTree = new double[nTrees][];
            for (int i = 0; i < numInstances; i++) {
                for (int t = 0; t < nTrees; t++) perTree[t] = treeGrads[t].gradient(i, treeParams[t]);
                double scale = clipScale(perTree, C);
                for (int t = 0; t < nTrees; t++)
                    for (int k = 0; k < acc[t].length; k++)
                        acc[t][k] += perTree[t][k] * scale;
            }

            // 2. Add Gaussian noise once per step (only when enabled).
            if (noiseStd > 0.0)
                for (int t = 0; t < nTrees; t++)
                    for (int k = 0; k < acc[t].length; k++)
                        acc[t][k] += noiseStd * rng.nextGaussian();

            // 3. Reg gradient in the clear, average, and step.
            for (int t = 0; t < nTrees; t++) {
                double lambda = lambdaPerTree[t];
                for (int k = 0; k < acc[t].length; k++) {
                    double grad = acc[t][k] / numInstances + lambda * treeParams[t][k];
                    treeParams[t][k] -= cfg.learningRate() * grad;
                }
            }
        }
    }
}
