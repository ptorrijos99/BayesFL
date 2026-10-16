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
 * Server fusion of {@link SPODE_Model}s: FedAvg of the parameters, each one weighted by
 * its support on every client when the clients send it (only the clients whose data
 * give it gradient move it) or by the clients' instances otherwise, and, for wCCBN against global tables, global tables written from
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

    /** Support-weighted sums of the parameters and total support (null: plain average only). */
    private double[][] sumSupported, sumSupport;

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
        sumSupported = null;
        sumSupport = null;
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
        if (local.support != null) {
            if (sumSupport == null) {
                sumSupported = new double[m][];
                sumSupport = new double[m][];
                for (int s = 0; s < m; s++) {
                    sumSupported[s] = new double[local.params[s].length];
                    sumSupport[s] = new double[local.params[s].length];
                }
            }
            for (int s = 0; s < m; s++) {
                double[] acc = sumSupported[s], sup = sumSupport[s];
                double[] p = local.params[s], a = local.support[s];
                for (int q = 0; q < acc.length; q++) {
                    acc[q] += a[q] * p[q];
                    sup[q] += a[q];
                }
            }
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
        for (int s = 0; s < sumParams.length; s++) {
            double[] acc = sumParams[s];
            for (int q = 0; q < acc.length; q++) {
                // Average over the clients that trained the parameter, plain average if none did
                acc[q] = sumSupport != null && sumSupport[s][q] > 0 ? sumSupported[s][q] / sumSupport[s][q]
                        : acc[q] * scale;
            }
            if (sumSupport != null) {
                sumSupported[s] = null;
                sumSupport[s] = null;
            }
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
        sumSupported = null;
        sumSupport = null;
        last = null;
        return global;
    }

    @Override
    public Model fusion(Model model1, Model model2) {
        return fusion(new Model[]{model1, model2});
    }
}
