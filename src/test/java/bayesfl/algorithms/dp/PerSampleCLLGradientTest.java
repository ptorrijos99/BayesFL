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
 * Package containing tests for the DP-SGD discriminative gradient channel.
 */
package bayesfl.algorithms.dp;

/**
 * Third party imports.
 */
import EBNC.wdBayes;
import objectiveFunction.ObjectiveFunction;
import optimize.FunctionValues;
import org.junit.jupiter.api.Test;
import weka.core.Instances;

import java.io.StringReader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Characterization test: the per-instance non-regularized discriminative
 * (CLL) gradient, summed over the whole dataset, must reproduce exactly the
 * batch gradient computed by EBNC's {@code ObjectiveFunctionCLL_d} when
 * regularization is off. This is the correctness gate for
 * {@link PerSampleCLLGradient}, which ports the same per-instance data term
 * so that DP-SGD can clip and noise it per sample.
 */
class PerSampleCLLGradientTest {

    private static Instances tinyData() throws Exception {
        String arff =
            "@relation t\n" +
            "@attribute a {x,y}\n" +
            "@attribute b {p,q}\n" +
            "@attribute class {0,1}\n" +
            "@data\n" +
            "x,p,0\nx,q,1\ny,p,1\ny,q,0\nx,p,1\ny,q,1\n";
        Instances d = new Instances(new StringReader(arff));
        d.setClassIndex(d.numAttributes() - 1);
        return d;
    }

    @Test
    void perSampleSumMatchesBatchGradient() throws Exception {
        Instances data = tinyData();
        wdBayes alg = new wdBayes();
        // Naive Bayes structure ("-S NB"), discriminative CLL training
        // ("-P dCCBN", EBNC.wdBayes m_P default is "MAP"), a single
        // optimizer iteration ("-I 1", irrelevant here since the gradient
        // is evaluated at an explicit parameter vector). Regularization
        // ("-R") is NOT passed: EBNC.wdBayes#m_Regularization defaults to
        // false, so getRegularization() is already false and the batch
        // gradient is the pure (non-regularized) data term.
        alg.setOptions(new String[]{"-S", "NB", "-P", "dCCBN", "-I", "1"});
        alg.buildClassifier(data);

        assertFalse(alg.getRegularization(), "regularization must be off for this characterization test");

        int np = alg.getdParameters_().getNp();
        double[] params = new double[np]; // evaluate at zero weights

        // Library batch gradient (regularization off => pure data term).
        ObjectiveFunction obj = alg.getObjectiveFunction();
        FunctionValues fv = obj.getValues(params);
        double[] batch = fv.gradient;

        // Sum of per-sample gradients.
        PerSampleCLLGradient psg = new PerSampleCLLGradient(alg);
        double[] sum = new double[np];
        for (int i = 0; i < data.numInstances(); i++) {
            double[] gi = psg.gradient(i, params);
            for (int k = 0; k < np; k++) sum[k] += gi[k];
        }

        assertArrayEquals(batch, sum, 1e-9,
            "sum of per-sample gradients must equal library batch gradient");
    }
}
