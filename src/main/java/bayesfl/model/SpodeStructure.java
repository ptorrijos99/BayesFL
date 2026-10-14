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
 * Package containing models related with federated Bayesian networks.
 */
package bayesfl.model;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

/**
 * Immutable, globally shared layout of the SPODEs of an AnDE ensemble trained with
 * the restricted class-conditional objective ({@code SPODE_CCBN}).
 * <p>
 * Each SPODE is the class-conditional network of the class and its super-parents.
 * As in the {@code WDPT_CCBN} encoding, the joint value (parents, y) is a synthetic
 * class taken from the federated class maps, and every synthetic class has a block
 * of parameters per attribute value. Unlike that encoding, the softmax of an
 * instance only runs over the synthetic classes compatible with its observed parent
 * values, so it models P(y | x) as in Zaidi et al. (2017) instead of P(parents, y | x).
 * </p>
 * Parameter layout of SPODE {@code s}: the {@code nc[s]} class parameters, then one
 * block of {@code nc[s] * card[u]} parameters per attribute {@code u}, indexed by
 * {@link #index(int, int, int, int)}.
 */
public final class SpodeStructure {

    /** Number of predictive attributes. */
    public final int nAttributes;

    /** Number of values of the original class. */
    public final int nClasses;

    /** Number of values of each predictive attribute. */
    public final int[] card;

    /** Super-parent indices of each SPODE. */
    public final int[][] parents;

    /** Number of synthetic classes of each SPODE. */
    public final int[] nc;

    /** Original class of each synthetic class, per SPODE. */
    public final int[][] synthY;

    /** Offset of the block of attribute {@code u} in SPODE {@code s}. */
    private final int[][] offsets;

    /** Number of parameters of each SPODE. */
    public final int[] np;

    /** Synthetic classes compatible with each parent configuration, per SPODE. */
    private final Map<Long, int[]>[] compatible;

    /** Index of each (parent configuration, class) pair, per SPODE. */
    private final Map<Long, Integer>[] synthIndex;

    /** All synthetic classes of each SPODE (fallback for unseen parent configurations). */
    private final int[][] allClasses;

    /** Pool that runs the SPODEs in parallel. */
    private static ForkJoinPool pool;

    /**
     * Builds the layout from the data header and the federated synthetic class maps.
     *
     * @param header       The header of the (categorical) data, class last.
     * @param combinations The super-parent indices of each SPODE.
     * @param classMaps    The global synthetic class map of each SPODE (key "p1|||...|||y").
     */
    @SuppressWarnings("unchecked")
    public SpodeStructure(Instances header, List<int[]> combinations, List<Map<String, Integer>> classMaps) {
        this.nAttributes = header.numAttributes() - 1;
        if (header.classIndex() != nAttributes) {
            throw new IllegalArgumentException("SPODE_CCBN expects the class as the last attribute");
        }
        this.nClasses = header.numClasses();
        this.card = new int[nAttributes];
        int sumCard = 0;
        for (int u = 0; u < nAttributes; u++) {
            Attribute att = header.attribute(u);
            if (!att.isNominal()) {
                throw new IllegalArgumentException("SPODE_CCBN needs categorical attributes: " + att.name());
            }
            card[u] = att.numValues();
            sumCard += card[u];
        }

        int m = combinations.size();
        this.parents = new int[m][];
        this.nc = new int[m];
        this.synthY = new int[m][];
        this.offsets = new int[m][];
        this.np = new int[m];
        this.compatible = new Map[m];
        this.synthIndex = new Map[m];
        this.allClasses = new int[m][];

        Attribute classAtt = header.classAttribute();
        for (int s = 0; s < m; s++) {
            int[] par = combinations.get(s);
            parents[s] = par.clone();
            Map<String, Integer> map = classMaps.get(s);
            nc[s] = map.size();
            synthY[s] = new int[nc[s]];

            Map<Long, List<Integer>> byParent = new HashMap<>();
            synthIndex[s] = new HashMap<>(2 * nc[s]);
            for (Map.Entry<String, Integer> e : map.entrySet()) {
                String[] parts = e.getKey().split("\\|\\|\\|", -1);
                int c = e.getValue();
                int y = classAtt.indexOfValue(parts[par.length]);
                if (y < 0) throw new IllegalStateException("Unknown class label in synthetic class " + e.getKey());
                synthY[s][c] = y;
                long code = 0;
                for (int j = 0; j < par.length; j++) {
                    int v = parts[j].equals("?") ? card[par[j]] : header.attribute(par[j]).indexOfValue(parts[j]);
                    if (v < 0) throw new IllegalStateException("Unknown parent label in synthetic class " + e.getKey());
                    code = code * (card[par[j]] + 1) + v;
                }
                byParent.computeIfAbsent(code, k -> new java.util.ArrayList<>()).add(c);
                synthIndex[s].put(code * nClasses + y, c);
            }
            compatible[s] = new HashMap<>(2 * byParent.size());
            for (Map.Entry<Long, List<Integer>> e : byParent.entrySet()) {
                compatible[s].put(e.getKey(), e.getValue().stream().mapToInt(Integer::intValue).sorted().toArray());
            }
            allClasses[s] = new int[nc[s]];
            for (int c = 0; c < nc[s]; c++) allClasses[s][c] = c;

            offsets[s] = new int[nAttributes];
            int off = nc[s];
            for (int u = 0; u < nAttributes; u++) {
                offsets[s][u] = off;
                off += nc[s] * card[u];
            }
            np[s] = nc[s] * (1 + sumCard);
        }
    }

    /** @return the number of SPODEs. */
    public int nSpodes() {
        return parents.length;
    }

    /**
     * @return the index of the parameter of attribute {@code u} with value {@code v} under synthetic class {@code c}.
     */
    public int index(int s, int c, int u, int v) {
        return offsets[s][u] + c * card[u] + v;
    }

    /** @return the offset of the block of attribute {@code u} in SPODE {@code s}. */
    public int offset(int s, int u) {
        return offsets[s][u];
    }

    /**
     * Encodes an instance as attribute value indices; a missing value is -1.
     *
     * @param inst The instance.
     * @return The value index of every predictive attribute.
     */
    public int[] encode(Instance inst) {
        int[] x = new int[nAttributes];
        for (int u = 0; u < nAttributes; u++) {
            x[u] = inst.isMissing(u) ? -1 : (int) inst.value(u);
        }
        return x;
    }

    /**
     * @return the code of the parent configuration of an encoded instance (missing is its own value).
     */
    public long parentCode(int s, int[] x) {
        long code = 0;
        for (int p : parents[s]) {
            code = code * (card[p] + 1) + (x[p] < 0 ? card[p] : x[p]);
        }
        return code;
    }

    /**
     * @return the synthetic classes compatible with the parents of {@code x}, or every class if the
     *         configuration was never observed in training.
     */
    public int[] compatible(int s, int[] x) {
        int[] cs = compatible[s].get(parentCode(s, x));
        return cs != null ? cs : allClasses[s];
    }

    /**
     * @return the synthetic class of a training instance with class {@code y}, or -1 if unknown.
     */
    public int synthClass(int s, int[] x, int y) {
        Integer c = synthIndex[s].get(parentCode(s, x) * nClasses + y);
        return c == null ? -1 : c;
    }

    /**
     * Value of attribute {@code u} used as a feature: a missing value is read as value 0,
     * as in wdBayes ({@code (int) NaN == 0}).
     */
    public static int feature(int[] x, int u) {
        return x[u] < 0 ? 0 : x[u];
    }

    /**
     * Pool used to train and evaluate SPODEs in parallel. Its size is the system property
     * {@code bayesfl.threads}, or the available processors.
     *
     * @return The shared pool.
     */
    public static synchronized ForkJoinPool pool() {
        if (pool == null) {
            int threads = Integer.getInteger("bayesfl.threads", Runtime.getRuntime().availableProcessors());
            pool = new ForkJoinPool(Math.max(1, threads));
        }
        return pool;
    }
}
