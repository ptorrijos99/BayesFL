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
 * Package containing fusion methods for federated Bayesian networks.
 */
package bayesfl.fusion;

import bayesfl.model.Model;
import bayesfl.model.SPODE_Model;
import bayesfl.model.SpodeStructure;

/**
 * Server fusion of {@link SPODE_Model}s: FedAvg of the parameters weighted by the
 * clients' instances and, for wCCBN against global tables, global tables written from
 * the sum of the clients' round-1 counts (the centralized m-estimates). The global
 * tables are kept and sent again in every later round.
 */
public class SPODE_Fusion_Server implements StreamingFusion {

    /** Whether the parameters are averaged. */
    private final boolean fuseParameters;

    /** Whether the tables are made global. */
    private final boolean fuseProbabilities;

    /** Running weighted sum of the parameters. */
    private double[][] sumParams;

    /** Running sum of the round-1 counts. */
    private double[][] sumCounts;

    /** Running number of instances. */
    private double total;

    /** The last model added (structure and flags). */
    private SPODE_Model last;

    /** The global tables, once built. */
    private double[][] globalTables;

    /**
     * Constructor.
     *
     * @param fuseParameters    Whether the parameters are averaged.
     * @param fuseProbabilities Whether the tables are made global.
     */
    public SPODE_Fusion_Server(boolean fuseParameters, boolean fuseProbabilities) {
        if (!fuseParameters) {
            throw new IllegalArgumentException("SPODE_CCBN needs parameter fusion (-FP)");
        }
        this.fuseParameters = fuseParameters;
        this.fuseProbabilities = fuseProbabilities;
    }

    @Override
    public void begin() {
        sumParams = null;
        sumCounts = null;
        total = 0;
        last = null;
    }

    @Override
    public void accumulate(Model model) {
        SPODE_Model local = (SPODE_Model) model;
        int m = local.params.length;
        double w = local.getNumInstances();
        if (sumParams == null) {
            sumParams = new double[m][];
            for (int s = 0; s < m; s++) sumParams[s] = new double[local.params[s].length];
        }
        for (int s = 0; s < m; s++) {
            double[] acc = sumParams[s];
            double[] p = local.params[s];
            for (int q = 0; q < acc.length; q++) acc[q] += w * p[q];
        }
        if (fuseProbabilities && local.counts != null) {
            if (sumCounts == null) {
                sumCounts = new double[m][];
                for (int s = 0; s < m; s++) sumCounts[s] = new double[local.counts[s].length];
            }
            for (int s = 0; s < m; s++) {
                double[] acc = sumCounts[s];
                double[] c = local.counts[s];
                for (int q = 0; q < acc.length; q++) acc[q] += c[q];
            }
        }
        total += w;
        last = local;
    }

    @Override
    public Model finish() {
        SpodeStructure st = last.structure;
        double scale = total > 0 ? 1.0 / total : 0;
        for (double[] acc : sumParams) {
            for (int q = 0; q < acc.length; q++) acc[q] *= scale;
        }
        if (sumCounts != null) {
            for (int s = 0; s < sumCounts.length; s++) SPODE_Model.toTables(st, s, sumCounts[s], last.logPrior);
            globalTables = sumCounts;
        }
        double[][] tables = fuseProbabilities && last.weighted ? globalTables : null;
        SPODE_Model global = new SPODE_Model(st, sumParams, tables, null, last.weighted, last.logPrior,
                (int) total, last.getMaxIterations());
        sumParams = null;
        sumCounts = null;
        last = null;
        return global;
    }

    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[]{model1, model2});
    }
}
