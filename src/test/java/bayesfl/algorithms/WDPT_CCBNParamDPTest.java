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
 * Package containing tests for federated Bayesian network algorithms.
 */
package bayesfl.algorithms;

/**
 * Third party imports.
 */
import org.junit.jupiter.api.Test;
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.algorithms.dp.DPSGDConfig;
import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.model.Classes;
import bayesfl.model.WDPT;

import java.io.StringReader;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Characterization test for the DP-SGD private path wired into
 * {@link WDPT_CCBN#buildLocalModel(bayesfl.model.Model, Data)}.
 * <p>
 * The synthetic class maps required by {@code WDPT_CCBN} are built exactly
 * the way {@code CCBNExperiment#getAlgorithm} does it: by running the
 * structure-only {@link Classes_AnDE} algorithm and reusing its
 * {@link Classes#getSyntheticClassMaps()}.
 */
class WDPT_CCBNParamDPTest {

    private static Data tinyData() throws Exception {
        String arff =
            "@relation t\n@attribute a {x,y}\n@attribute b {p,q}\n@attribute class {0,1}\n" +
            "@data\nx,p,0\nx,q,1\ny,p,1\ny,q,0\nx,p,1\ny,q,1\n";
        Instances d = new Instances(new StringReader(arff));
        d.setClassIndex(d.numAttributes() - 1);
        return new Weka_Instances("tiny", d, d);
    }

    private static String[] nbOptions() {
        // Naive Bayes AnDE (n=0), discriminative CLL training ("-P dCCBN",
        // NOT "-D" -- confirmed in Task 3 as the flag that actually turns on
        // discriminative parameter learning for EBNC.wdBayes).
        return new String[]{"-S", "A0DE", "-P", "dCCBN", "-I", "1"};
    }

    /**
     * Builds the synthetic class maps the same way
     * {@code CCBNExperiment#getAlgorithm} does for "WDPT_CCBN": via the
     * structure-only {@link Classes_AnDE} local algorithm.
     */
    private static List<Map<String, Integer>> globalMapsFor(Data data) {
        Classes structure = (Classes) new Classes_AnDE(nbOptions()).buildLocalModel(data);
        return structure.getSyntheticClassMaps();
    }

    @Test
    void enabledDPProducesFiniteDifferentParameters() throws Exception {
        Data data = tinyData();
        List<Map<String, Integer>> maps = globalMapsFor(data);

        // Baseline: no param DP.
        WDPT_CCBN base = new WDPT_CCBN(nbOptions(), null, maps);
        WDPT m0 = (WDPT) base.buildLocalModel(data);
        WDPT m1 = (WDPT) base.buildLocalModel(m0, data);
        double[] plain = m1.getTrees().get(0).getParameters().clone();

        // Param DP enabled at a finite epsilon.
        DPSGDConfig dp = new DPSGDConfig(1.0, 1e-5, 1.0, 1, 5, 0.1, 123L);
        WDPT_CCBN priv = new WDPT_CCBN(nbOptions(), null, maps, null, dp);
        WDPT p0 = (WDPT) priv.buildLocalModel(data);
        WDPT p1 = (WDPT) priv.buildLocalModel(p0, data);
        double[] noisy = p1.getTrees().get(0).getParameters();

        for (double v : noisy) assertTrue(Double.isFinite(v), "params must stay finite");
        assertFalse(java.util.Arrays.equals(plain, noisy), "DP path must change parameters");
    }
}
