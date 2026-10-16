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
 * Package containing algorithms related with federated Bayesian networks.
 */
package bayesfl.algorithms;

import bayesfl.data.Data;
import bayesfl.data.Weka_Instances;
import bayesfl.model.Classes;
import bayesfl.model.Model;
import bayesfl.model.SPODE_Model;
import bayesfl.model.SpodeStructure;
import optimize.DifferentiableFunction;
import optimize.FunctionValues;
import optimize.Minimizer;
import optimize.StopConditions;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import static org.albacete.simd.utils.Utils.hasActiveCutPoints;

/**
 * Class-conditional AnDE (dCCBN / wCCBN parameters per SPODE) trained with the
 * conditional log-likelihood of each SPODE, P(y | x), as in Zaidi et al. (2017).
 * <p>
 * It replaces the {@code WDPT_CCBN} training path (wdBayes over a synthetic class),
 * whose softmax runs over every (parents, y) synthetic class, by the softmax over the
 * classes compatible with the observed parents. The client encodes its data once,
 * keeps no model between rounds (the fused model carries the global parameters and,
 * with global tables, the global tables) and trains the SPODEs in parallel.
 * </p>
 * Options: {@code -S AnDE}, {@code -P dCCBNr|wCCBNr}, {@code -I} L-BFGS iterations per
 * round, {@code -GT} (wCCBN against global tables, round-1 counts released),
 * {@code -INIT zero|gen} (start at 0, as wdBayes, or at the generative solution; by default
 * dCCBN and wCCBN against global tables start at the generative solution, Zaidi's "(I)"
 * variants, and wCCBN against local tables at 0, since w = 1 on tables built from a few
 * instances is a poor start),
 * {@code -LAMBDA} (L2 towards the generative solution, Zaidi 2017 Sec. 5.2; default 0)
 * and {@code -LOGPRIOR} (wCCBN class feature log P(c) instead of P(c)). The default class
 * feature is P(c), as in wdBayes: with a free weight per class both give the same model
 * family, but log P(c) of local tables built from a few instances is a large, noisy feature
 * ({@code -LINPRIOR} is accepted as a no-op).
 * With the restricted softmax a parameter only gets gradient on the clients whose data use it,
 * so each local model sends the support of its parameters and the server averages every
 * parameter over the clients that trained it; {@code -NOSUPPORT} falls back to the plain
 * average weighted by the number of instances.
 */
public class SPODE_CCBN implements LocalAlgorithm {

    /** Stop criterion on the gradient norm, as in {@code WDPT_CCBN} (effectively off). */
    private static final double MAX_GRADIENT_NORM = 1e-32;

    /** Layouts shared by every client of the same federated structure. */
    private static final Map<Classes, SpodeStructure> LAYOUTS = new WeakHashMap<>();

    /** The federated AnDE structure. */
    private final Classes classes;

    /** The global layout of the SPODEs (shared, built from the first client's header). */
    private SpodeStructure structure;

    /** Whether the parameters are wCCBN weights. */
    private final boolean weighted;

    /** Whether wCCBN uses global tables (hybrid-global). */
    private final boolean globalTables;

    /** Whether the parameters start at the generative solution. */
    private final boolean generativeInit;

    /** Strength of the L2 penalty towards the generative solution. */
    private final double lambda;

    /** Whether the wCCBN class feature is log P(c). */
    private final boolean logPrior;

    /** Whether local models send the support of their parameters. */
    private final boolean sendSupport;

    /** L-BFGS iterations per round. */
    private final int maxIterations;

    /** The client's encoded training instances. */
    private int[][] X;

    /** Synthetic class of each training instance, per SPODE. */
    private int[][] yS;

    /** Compatible synthetic classes of each training instance, per SPODE. */
    private int[][][] compat;

    /** Position of the true synthetic class inside {@code compat}, per SPODE. */
    private int[][] truePos;

    /**
     * Constructor.
     *
     * @param options   The algorithm options.
     * @param cutPoints The discretization cut points (must be inactive: categorical data only).
     * @param classes   The federated AnDE structure (combinations and synthetic class maps).
     */
    public SPODE_CCBN(String[] options, double[][] cutPoints, Classes classes) {
        if (hasActiveCutPoints(cutPoints)) {
            throw new UnsupportedOperationException("SPODE_CCBN supports categorical data only");
        }
        String[] opts = Arrays.copyOf(options, options.length);
        try {
            String scheme = Utils.getOption("P", opts);
            this.weighted = scheme.toLowerCase().startsWith("w");
            String iters = Utils.getOption("I", opts);
            this.maxIterations = iters.isEmpty() ? 10 : Integer.parseInt(iters);
            this.globalTables = Utils.getFlag("GT", opts);
            String init = Utils.getOption("INIT", opts);
            this.generativeInit = init.isEmpty() ? !weighted || globalTables : init.equalsIgnoreCase("gen");
            String lam = Utils.getOption("LAMBDA", opts);
            this.lambda = lam.isEmpty() ? 0 : Double.parseDouble(lam);
            this.logPrior = Utils.getFlag("LOGPRIOR", opts);
            Utils.getFlag("LINPRIOR", opts);
            this.sendSupport = !Utils.getFlag("NOSUPPORT", opts);
        } catch (Exception e) {
            throw new IllegalArgumentException("Invalid SPODE_CCBN options: " + Arrays.toString(options), e);
        }
        this.classes = classes;
    }

    /** @return the layout shared by all clients of this structure. */
    private static SpodeStructure layout(Classes classes, Instances header) {
        synchronized (LAYOUTS) {
            return LAYOUTS.computeIfAbsent(classes,
                    k -> new SpodeStructure(header, k.getCombinations(), k.getSyntheticClassMaps()));
        }
    }

    /**
     * Encodes the client's training data once: value indices, and per SPODE the synthetic
     * class, the compatible classes and the position of the true class among them.
     */
    private void encode(Data data) {
        if (X != null) return;
        Instances train = ((Weka_Instances) data).getTrain();
        structure = layout(classes, train);
        SpodeStructure st = structure;
        int n = train.numInstances();
        X = new int[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            X[i] = st.encode(train.instance(i));
            y[i] = (int) train.instance(i).classValue();
        }
        int m = st.nSpodes();
        yS = new int[m][n];
        compat = new int[m][n][];
        truePos = new int[m][n];
        for (int s = 0; s < m; s++) {
            for (int i = 0; i < n; i++) {
                int c = st.synthClass(s, X[i], y[i]);
                if (c < 0) throw new IllegalStateException("Training instance without a synthetic class");
                yS[s][i] = c;
                compat[s][i] = st.compatible(s, X[i]);
                truePos[s][i] = Arrays.binarySearch(compat[s][i], c);
            }
        }
    }

    @Override
    public Model buildLocalModel(Data data) {
        return buildLocalModel(null, data);
    }

    /**
     * One round of local training. Round 1 starts from 0 or the generative solution and,
     * for wCCBN, uses local tables (as wdBayes does before the tables are fused); later
     * rounds start from the fused global parameters and use the global tables when they
     * exist, or the local ones recomputed from the data.
     */
    @Override
    public Model buildLocalModel(Model localModel, Data data) {
        encode(data);
        SPODE_Model previous = (SPODE_Model) localModel;
        SpodeStructure st = structure;
        int m = st.nSpodes();
        boolean releaseCounts = previous == null && weighted && globalTables;

        double[][] params = new double[m][];
        double[][] tables = weighted ? new double[m][] : null;
        double[][] counts = releaseCounts ? new double[m][] : null;
        double[][] support = sendSupport ? new double[m][] : null;
        // The generative solution is the start of round 1 and the centre of the L2 penalty
        boolean needGen = previous == null && generativeInit || lambda > 0;

        try {
            SpodeStructure.pool().submit(() -> IntStream.range(0, m).parallel().forEach(s -> {
                double[] given = previous != null && previous.logTables != null ? previous.logTables[s] : null;
                double[] local = null;
                if (given == null && (weighted || needGen)) {
                    double[] cnt = SPODE_Model.counts(st, s, X, yS[s]);
                    if (releaseCounts) counts[s] = cnt.clone();
                    local = SPODE_Model.toTables(st, s, cnt, logPrior);
                }
                double[] f = weighted ? (given != null ? given : local) : null;
                double[] gen = needGen ? generative(s, f, local) : null;

                double[] start = previous != null ? previous.params[s].clone()
                        : generativeInit ? gen.clone() : new double[st.np[s]];
                params[s] = optimize(s, f, start, gen);
                if (weighted) tables[s] = f;
                if (sendSupport) support[s] = support(s);
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }

        // Clients keep only the global tables between rounds; local ones are recomputed
        double[][] kept = weighted && previous != null && previous.logTables != null ? previous.logTables : tables;
        return new SPODE_Model(st, params, kept, counts, support, weighted, logPrior, X.length, maxIterations);
    }

    /**
     * Support of every parameter of SPODE {@code s} on the client's data: the number of instances
     * whose restricted softmax uses it, that is, the only ones that give it gradient.
     */
    private double[] support(int s) {
        SpodeStructure st = structure;
        double[] sup = new double[st.np[s]];
        int[][] cs = compat[s];
        for (int i = 0; i < X.length; i++) {
            int[] x = X[i];
            for (int c : cs[i]) {
                sup[c]++;
                for (int u = 0; u < st.nAttributes; u++) sup[st.index(s, c, u, SpodeStructure.feature(x, u))]++;
            }
        }
        return sup;
    }

    /**
     * The generative solution in parameter space: w = 1 for wCCBN, the log tables for dCCBN.
     */
    private double[] generative(int s, double[] f, double[] local) {
        if (weighted) {
            double[] ones = new double[structure.np[s]];
            Arrays.fill(ones, 1.0);
            return ones;
        }
        double[] g = local.clone();
        if (!logPrior) {
            for (int c = 0; c < structure.nc[s]; c++) g[c] = Math.log(g[c]);
        }
        return g;
    }

    /**
     * L-BFGS-B on the restricted conditional log-likelihood of SPODE {@code s}.
     */
    private double[] optimize(int s, double[] f, double[] start, double[] center) {
        Minimizer minimizer = new Minimizer();
        StopConditions sc = minimizer.getStopConditions();
        sc.setMaxGradientNorm(MAX_GRADIENT_NORM);
        sc.setMaxIterations(maxIterations);
        try {
            return minimizer.run(objective(s, f, center), start).point;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Negative conditional log-likelihood of SPODE {@code s} over the client's data and its
     * gradient, with the softmax restricted to the classes compatible with each instance's
     * parents. The score of synthetic class c is sum_q p[q] f[q] over its class parameter and
     * the parameters of the observed attribute values (f = 1 for dCCBN).
     */
    DifferentiableFunction objective(int s, double[] f, double[] center) {
        SpodeStructure st = structure;
        int a = st.nAttributes;
        int[][] cs = compat[s];
        int[] tp = truePos[s];
        int[] off = new int[a];
        int[] card = st.card;
        for (int u = 0; u < a; u++) off[u] = st.offset(s, u);
        int maxC = 1;
        for (int[] c : cs) maxC = Math.max(maxC, c.length);
        final int width = maxC;

        return p -> {
            double[] g = new double[p.length];
            double[] score = new double[width];
            double nll = 0;
            for (int i = 0; i < X.length; i++) {
                int[] x = X[i];
                int[] classes = cs[i];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < classes.length; j++) {
                    int c = classes[j];
                    double v = f == null ? p[c] : p[c] * f[c];
                    for (int u = 0; u < a; u++) {
                        int q = off[u] + c * card[u] + SpodeStructure.feature(x, u);
                        v += f == null ? p[q] : p[q] * f[q];
                    }
                    score[j] = v;
                    if (v > max) max = v;
                }
                double z = 0;
                for (int j = 0; j < classes.length; j++) z += Math.exp(score[j] - max);
                double logZ = max + Math.log(z);
                nll -= score[tp[i]] - logZ;
                for (int j = 0; j < classes.length; j++) {
                    int c = classes[j];
                    double r = Math.exp(score[j] - logZ) - (j == tp[i] ? 1 : 0);
                    g[c] += f == null ? r : r * f[c];
                    for (int u = 0; u < a; u++) {
                        int q = off[u] + c * card[u] + SpodeStructure.feature(x, u);
                        g[q] += f == null ? r : r * f[q];
                    }
                }
            }
            if (lambda > 0) {
                for (int q = 0; q < p.length; q++) {
                    double d = p[q] - center[q];
                    nll += lambda / 2 * d * d;
                    g[q] += lambda * d;
                }
            }
            return new FunctionValues(nll, g);
        };
    }

    @Override
    public Model refinateLocalModel(Model oldModel, Model localModel, Data data) {
        return localModel;
    }

    @Override
    public String getAlgorithmName() {
        return "SPODE_CCBN";
    }

    @Override
    public String getRefinementName() {
        return "None";
    }

    /** @return the global layout of the SPODEs. */
    public SpodeStructure getStructure() {
        return structure;
    }
}
