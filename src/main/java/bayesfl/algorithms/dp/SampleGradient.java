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
 * A source of per-sample (non-regularized) data-loss gradients over a fixed
 * parameter vector, for a single tree of a joint multi-tree DP-SGD run.
 *
 * <p>Implementations are expected to be stateless with respect to the
 * supplied {@code params}: each call re-evaluates the gradient at exactly
 * the parameter vector passed in, without depending on the outcome of prior
 * calls, so that {@link DPSGDOptimizer} may call this repeatedly at a fixed
 * parameter vector while accumulating the full-batch gradient.</p>
 */
public interface SampleGradient {

    /**
     * Computes the per-instance, non-regularized data-loss gradient.
     *
     * @param instanceIndex index of the training instance.
     * @param params        the parameter vector to evaluate the gradient at.
     * @return a freshly allocated gradient vector of length {@link #numParameters()}.
     */
    double[] gradient(int instanceIndex, double[] params);

    /** @return the number of parameters (gradient length) for this tree. */
    int numParameters();
}
