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
 * Package containing tests for the DP-SGD discriminative gradient channel.
 */
package bayesfl.algorithms.dp;

/**
 * A minimal {@link SampleGradient} test double for a single quadratic-bowl
 * "tree": {@code f(w) = 0.5 * ||w - target||^2}, whose gradient is
 * {@code w - target}. There is a single logical instance (index 0); the
 * gradient is evaluated at whatever parameter vector the optimizer passes
 * in, exactly as a real {@link SampleGradient} would.
 */
final class FakeGrad implements SampleGradient {

    /** The quadratic bowl's minimizer. */
    private final double[] target;

    /** Number of parameters (gradient length). */
    private final int np;

    private FakeGrad(double[] w, double[] target) {
        this.np = w.length;
        this.target = target;
    }

    /** Builds a fake gradient provider for the bowl centered at {@code target}. */
    static FakeGrad of(double[] w, double[] target) {
        return new FakeGrad(w, target);
    }

    @Override
    public double[] gradient(int instanceIndex, double[] params) {
        double[] g = new double[np];
        for (int k = 0; k < np; k++) {
            g[k] = params[k] - target[k];
        }
        return g;
    }

    @Override
    public int numParameters() {
        return np;
    }
}
