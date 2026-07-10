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

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DPSGDOptimizerTest {

    @Test
    void sigmaZeroWhenEpsilonInfinite() {
        DPSGDConfig cfg = new DPSGDConfig(Double.POSITIVE_INFINITY, 1e-5, 1.0, 10, 1, 0.1, 42L);
        assertEquals(0.0, cfg.sigma(), 0.0);
    }

    @Test
    void sigmaPositiveWhenEpsilonFinite() {
        DPSGDConfig cfg = new DPSGDConfig(1.0, 1e-5, 1.0, 10, 2, 0.1, 42L);
        assertTrue(cfg.sigma() > 0);
    }

    @Test
    void perSampleClipBoundsConcatenatedNorm() {
        // Two "trees" of 2 params each; a fake gradient provider returning a huge vector.
        double C = 1.0;
        double[] huge = {10.0, 10.0}; // norm ~14.14 per tree, ~20 concatenated
        double clipped = DPSGDOptimizer.clipScale(new double[][]{huge, huge}, C);
        // scaled concatenated norm must be <= C (within fp tolerance)
        double n2 = 0;
        for (double[] t : new double[][]{huge, huge}) for (double x : t) n2 += (x * clipped) * (x * clipped);
        assertTrue(Math.sqrt(n2) <= C + 1e-9);
    }

    @Test
    void noiselessFullBatchStepDecreasesConvexObjective() {
        // f(w) = 0.5 * ||w - target||^2 ; gradient = w - target ; single tree.
        double[] target = {1.0, -2.0, 0.5};
        double[] w = {0.0, 0.0, 0.0};
        double lr = 0.5;
        DPSGDConfig cfg = new DPSGDConfig(Double.POSITIVE_INFINITY, 1e-5, 1e9, 1, 20, lr, 7L);
        DPSGDOptimizer opt = new DPSGDOptimizer(cfg);
        // gradient provider: one "instance" whose gradient is (w - target)
        SampleGradient fake = FakeGrad.of(w, target);
        opt.runRound(new double[][]{w}, new SampleGradient[]{fake}, new double[]{0.0}, 1);
        double err = 0; for (int k = 0; k < 3; k++) err += (w[k]-target[k])*(w[k]-target[k]);
        assertTrue(Math.sqrt(err) < 1.0, "should move toward target; err=" + Math.sqrt(err));
    }
}
