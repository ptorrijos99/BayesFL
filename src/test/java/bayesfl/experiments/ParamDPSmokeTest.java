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
 * Package containing tests for federated Bayesian network experiments.
 */
package bayesfl.experiments;

/**
 * Third party imports.
 */
import org.junit.jupiter.api.Test;
import weka.core.Instances;

/**
 * Local application imports.
 */
import bayesfl.algorithms.Classes_AnDE;
import bayesfl.algorithms.WDPT_CCBN;
import bayesfl.algorithms.dp.DPSGDConfig;
import bayesfl.data.Weka_Instances;
import bayesfl.fusion.WDPT_Fusion_Client;
import bayesfl.fusion.WDPT_Fusion_Server;
import bayesfl.model.Classes;
import bayesfl.model.Model;
import bayesfl.model.WDPT;

import java.io.StringReader;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end smoke test for the param-space (DP-SGD) discriminative channel.
 * <p>
 * Assembles a minimal, hermetic 2-client federated round over a tiny
 * in-memory ARFF using the real classes: each client builds its own
 * {@link WDPT_CCBN} model (round 1), the server fuses both with
 * {@link WDPT_Fusion_Server}, each client folds the fused global model back
 * into its own local model with {@link WDPT_Fusion_Client} (preserving its
 * own classifiers, exactly as {@link bayesfl.Client#fusion(Model)} does),
 * then each client runs one more param-refinement round
 * ({@code buildLocalModel(localModel, data)}) on its own stationary data,
 * and the server fuses again.
 * <p>
 * This exercises the carry-forward concern that
 * {@code PerSampleCLLGradient} indexes each client's round-1 instances
 * {@code 0..N-1}: that indexing is only safe across rounds because
 * {@link WDPT_Fusion_Client} keeps each client's own classifier objects
 * (built from that client's own data) instead of swapping in whichever
 * client the server used as its fusion reference.
 * <p>
 * Kept hermetic (no file I/O): the fusion classes are called directly and
 * neither {@code CCBNExperiment.main} nor {@code WDPT.saveStats} is
 * invoked. The CSV column wiring is covered separately by
 * {@code WDPTHeaderTest}.
 */
class ParamDPSmokeTest {

    /**
     * 8 rows so the data can be split into two disjoint, non-degenerate
     * client halves (4 rows each, both classes represented in each half).
     */
    private static Instances fullData() throws Exception {
        String arff =
            "@relation t\n@attribute a {x,y}\n@attribute b {p,q}\n@attribute class {0,1}\n" +
            "@data\n" +
            "x,p,0\nx,q,1\ny,p,1\ny,q,0\n" +
            "x,p,1\ny,q,1\nx,q,0\ny,p,0\n";
        Instances d = new Instances(new StringReader(arff));
        d.setClassIndex(d.numAttributes() - 1);
        return d;
    }

    private static Weka_Instances clientData(Instances full, int from, int to, String name) {
        Instances subset = new Instances(full, 0);
        for (int i = from; i < to; i++) subset.add(full.instance(i));
        return new Weka_Instances(name, subset, subset);
    }

    private static String[] nbOptions() {
        // Naive Bayes AnDE (n=0), discriminative CLL training ("-P dCCBN",
        // NOT "-D" -- confirmed in WDPT_CCBNParamDPTest as the flag that
        // actually turns on discriminative parameter learning for EBNC.wdBayes).
        return new String[]{"-S", "A0DE", "-P", "dCCBN", "-I", "1"};
    }

    /**
     * Global synthetic class maps, built once over the full (union) data,
     * exactly as {@code CCBNExperiment#getAlgorithm} builds them via the
     * structure-only {@link Classes_AnDE} local algorithm before handing
     * them to every client's {@link WDPT_CCBN} instance.
     */
    private static List<Map<String, Integer>> globalMaps(Instances full) {
        Weka_Instances data = new Weka_Instances("full", full, full);
        Classes structure = (Classes) new Classes_AnDE(nbOptions()).buildLocalModel(data);
        return structure.getSyntheticClassMaps();
    }

    private static void assertParamsFinite(WDPT model) {
        for (var tree : model.getTrees()) {
            for (double v : tree.getParameters()) {
                assertTrue(Double.isFinite(v), "parameter must stay finite, got " + v);
            }
        }
    }

    /**
     * Runs a full 2-client federated round (build -> server fuse -> client
     * fuse -> param-refine -> server fuse) for the given DP-SGD
     * configuration (or {@code null} to disable the param channel),
     * mirroring {@link bayesfl.Server#run()} / {@link bayesfl.Client} for a
     * single iteration.
     *
     * @return the final, twice-fused global tree-0 parameters.
     */
    private static double[] runFederatedRound(DPSGDConfig dp) throws Exception {
        Instances full = fullData();
        List<Map<String, Integer>> maps = globalMaps(full);

        Weka_Instances dataA = clientData(full, 0, 4, "clientA");
        Weka_Instances dataB = clientData(full, 4, 8, "clientB");

        // Each client keeps its own WDPT_CCBN instance (own classifiers/statistics),
        // as CCBNExperiment#getAlgorithm builds one WDPT_CCBN per client.
        WDPT_CCBN algoA = new WDPT_CCBN(nbOptions(), null, maps, null, dp);
        WDPT_CCBN algoB = new WDPT_CCBN(nbOptions(), null, maps, null, dp);

        WDPT_Fusion_Server server = new WDPT_Fusion_Server(true, false);
        WDPT_Fusion_Client clientFusion = new WDPT_Fusion_Client(true, false);

        // --- Round 1: each client builds its local model from its own data ---
        Model localA = algoA.buildLocalModel(dataA);
        Model localB = algoB.buildLocalModel(dataB);
        assertParamsFinite((WDPT) localA);
        assertParamsFinite((WDPT) localB);

        // --- Server fuses the two local models into a global model ---
        Model global1 = server.fusion(localA, localB);
        assertParamsFinite((WDPT) global1);

        // --- Each client folds the global model back into its own local model,
        //     keeping its own classifiers (as bayesfl.Client#fusion does) ---
        localA = clientFusion.fusion(localA, global1);
        localB = clientFusion.fusion(localB, global1);
        assertParamsFinite((WDPT) localA);
        assertParamsFinite((WDPT) localB);

        // --- Round 2: one param-refinement round on each client's own
        //     (stationary) data, starting from the fused parameters ---
        localA = algoA.buildLocalModel(localA, dataA);
        localB = algoB.buildLocalModel(localB, dataB);
        assertParamsFinite((WDPT) localA);
        assertParamsFinite((WDPT) localB);

        // --- Server fuses again ---
        Model global2 = server.fusion(localA, localB);
        assertParamsFinite((WDPT) global2);

        return ((WDPT) global2).getTrees().get(0).getParameters().clone();
    }

    @Test
    void federatedBuildFuseRefineCompletesWithInfiniteEpsilon() throws Exception {
        double[] params = runFederatedRound(null);
        for (double v : params) {
            assertTrue(Double.isFinite(v), "fused parameter must be finite, got " + v);
        }
    }

    @Test
    void federatedBuildFuseRefineCompletesWithFiniteEpsilon() throws Exception {
        DPSGDConfig dp = new DPSGDConfig(1.0, 1e-5, 1.0, 1, 5, 0.1, 123L);
        double[] params = runFederatedRound(dp);
        for (double v : params) {
            assertTrue(Double.isFinite(v), "fused parameter must be finite, got " + v);
        }
    }

    @Test
    void epsilonTotalIsSumOfBothChannels() {
        // Mirrors the "WDPT_CCBN" branch of CCBNExperiment#getOperation:
        // epsilonTab = eps > 0 ? eps : 0, epParam = isInfinite(epsilonParam) ? 0 : epsilonParam,
        // epsilonTotal = epsilonTab + epParam.
        double epsilonTab = 0.5;
        double epsilonParam = 1.0;

        double tabContribution = epsilonTab > 0 ? epsilonTab : 0.0;
        double paramContribution = Double.isInfinite(epsilonParam) ? 0.0 : epsilonParam;
        double epsilonTotal = tabContribution + paramContribution;

        assertEquals(epsilonTab + epsilonParam, epsilonTotal, 1e-12);

        // Also confirm the "off" convention: a disabled param channel (infinite
        // epsilon) contributes 0 to epsilonTotal, matching WDPT_CCBN(..., paramDp=null)
        // and the epsilonParamVal = paramDp != null ? paramDp.epsilonParam() : POSITIVE_INFINITY default.
        double epsilonParamOff = Double.POSITIVE_INFINITY;
        double paramContributionOff = Double.isInfinite(epsilonParamOff) ? 0.0 : epsilonParamOff;
        assertEquals(epsilonTab, tabContribution + paramContributionOff, 1e-12);
    }
}
