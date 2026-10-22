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
 * Package containing algorithms related with differential privacy for
 * federated Bayesian networks.
 */
package bayesfl.algorithms.dp;

/**
 * Third party imports.
 */
import DataStructure.wdBayesNode;
import DataStructure.wdBayesParametersTree;
import EBNC.wdBayes;
import Utils.SUtils;
import objectiveFunction.ObjectiveFunctionCLL_w;
import objectiveFunction.parallel.ParallelObjectiveFunctionCLL_w;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Per-instance discriminative (CLL) data-loss gradient, ported from the EBNC
 * library objective the model was built with: {@code ObjectiveFunctionCLL_d}
 * for dCCBN (log-linear coefficients over indicator features) and
 * {@code ObjectiveFunctionCLL_w} for wCCBN (weights that multiply the
 * log-probability tables, so each term is scaled by its table value). Only the non-regularized per-instance data term is
 * reproduced here: DP-SGD clips and noises the data-loss gradient per
 * sample, then adds the data-independent regularization gradient in the
 * clear afterwards (post-processing), so the regularization branch of the
 * original library method is intentionally not ported.
 *
 * <p>Correctness gate: summing {@link #gradient(int, double[])} over every
 * instance of a dataset must reproduce exactly the batch gradient that
 * library objective's {@code getValues(double[])} computes when
 * {@code wdBayes#getRegularization()} is {@code false}.</p>
 */
public final class PerSampleCLLGradient implements SampleGradient {

    /**
     * The wdBayes model whose parameter tree and training instances this
     * class differentiates against.
     */
    private final wdBayes algorithm;

    /**
     * The discriminative parameter tree shared with the wdBayes model.
     */
    private final wdBayesParametersTree tree;

    /**
     * The training instances the model was built on.
     */
    private final Instances instances;

    /**
     * The attribute processing order used by the model.
     */
    private final int[] order;

    /**
     * Number of class values.
     */
    private final int nc;

    /**
     * Number of attributes (excluding the class).
     */
    private final int n;

    /**
     * Number of discriminative parameters, i.e. the gradient length.
     */
    private final int np;

    /**
     * Whether the tree holds wCCBN weights, whose features are the model's
     * log-probability tables, rather than dCCBN coefficients.
     */
    private final boolean weighted;

    /**
     * Builds a per-sample gradient computer bound to an already-built
     * wdBayes model.
     *
     * @param algorithm a wdBayes classifier that has already been fit via
     *                  {@code buildClassifier}, so its parameter tree and
     *                  training instances are available.
     */
    public PerSampleCLLGradient(wdBayes algorithm) {
        this.algorithm = algorithm;
        this.tree = algorithm.getdParameters_();
        this.instances = algorithm.getM_Instances();
        this.order = algorithm.getM_Order();
        this.nc = algorithm.getNc();
        this.n = algorithm.getnAttributes();
        this.np = tree.getNp();
        this.weighted = algorithm.getObjectiveFunction() instanceof ObjectiveFunctionCLL_w
                || algorithm.getObjectiveFunction() instanceof ParallelObjectiveFunctionCLL_w;
    }

    /**
     * Computes the length-{@code getNp()} non-regularized discriminative
     * (CLL) data-loss gradient of a single instance, evaluated at the given
     * parameter vector.
     *
     * @param instanceIndex index of the instance in the training data.
     * @param params        the parameter vector to evaluate the gradient
     *                      at; copied into the shared parameter tree before
     *                      evaluation, exactly as
     *                      {@code ObjectiveFunctionCLL_d#getValues(double[])}
     *                      does.
     * @return the per-instance gradient, non-regularized.
     */
    public double[] gradient(int instanceIndex, double[] params) {
        tree.copyParameters(params);
        double[] g = new double[np];
        Instance instance = instances.instance(instanceIndex);
        int x_C = (int) instance.classValue();

        wdBayesNode[] nodes = new wdBayesNode[n];
        wdBayes.findNodesForInstance(nodes, instance, tree);

        // wCCBN scales every parameter by its log-probability table entry;
        // dCCBN's features are indicators, i.e. a factor of one.
        double[] classCounts = tree.getClassCounts();
        double[] probs = new double[nc];
        for (int c = 0; c < nc; c++) {
            probs[c] = tree.getClassParameter(c) * (weighted ? classCounts[c] : 1.0);
        }
        for (int u = 0; u < nodes.length; u++) {
            wdBayesNode node = nodes[u];
            int v = (int) instance.value(order[u]);
            for (int c = 0; c < nc; c++) {
                probs[c] += node.getXYParameter(v, c) * (weighted ? node.getXYCount(v, c) : 1.0);
            }
        }
        SUtils.normalizeInLogDomain(probs);
        SUtils.exp(probs);

        for (int c = 0; c < nc; c++) {
            g[c] += -(SUtils.ind(c, x_C) - probs[c]) * (weighted ? classCounts[c] : 1.0);
        }
        for (int u = 0; u < nodes.length; u++) {
            wdBayesNode node = nodes[u];
            int v = (int) instance.value(order[u]);
            for (int c = 0; c < nc; c++) {
                int posp = node.getXYIndex(v, c);
                g[posp] += -(SUtils.ind(c, x_C) - probs[c]) * (weighted ? node.getXYCount(v, c) : 1.0);
            }
        }
        return g;
    }

    /**
     * @return the number of discriminative parameters (gradient length).
     */
    public int numParameters() {
        return np;
    }
}
