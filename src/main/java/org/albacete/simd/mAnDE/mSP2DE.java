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
 *    mSP2DE.java
 *    Copyright (C) 2026 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import bayesfl.privacy.NumericNoiseGenerator;
import weka.core.Instance;
import weka.core.Utils;

public class mSP2DE implements mSPnDE, Serializable {

    /**
     * ID of the first Super-Parent of the mSP2DE.
     */
    private final int xi1;

    /**
     * ID of the second Super-Parent of the mSP2DE.
     */
    private final int xi2;

    /**
     * Link the name of the children of the mSP2DE with their probability table.
     */
    private final HashMap<Integer, double[][][][]> children;

    /**
     * List of children of the mSP2DE.
     */
    private final Set<Integer> listChildren;

    /**
     * Overall probability table of the mSP2DE.
     */
    private double[][][] globalProbs;

    /**
     * Raw count table P(y, Xi1, Xi2) — used for federated parameter aggregation.
     */
    private double[][][] globalCounts;

    /**
     * Raw count tables P(Xj, y, Xi1, Xi2) per child — used for federated parameter
     * aggregation.
     */
    private HashMap<Integer, double[][][][]> childCounts;

    /** The keys and tables of {@link #children}, in the iteration order of the map. */
    private static final class ChildIndex {
        final int[] ids;
        final double[][][][][] tables;

        ChildIndex(HashMap<Integer, double[][][][]> children) {
            ids = new int[children.size()];
            tables = new double[children.size()][][][][];
            int k = 0;
            for (Map.Entry<Integer, double[][][][]> e : children.entrySet()) {
                ids[k] = e.getKey();
                tables[k++] = e.getValue();
            }
        }
    }

    /** {@link #children} as arrays; null until the tables are built (or after they change). */
    private transient volatile ChildIndex childIndex;

    /**
     * The tables learnt by {@link #buildTables}: the row of every state (y, xi1, xi2) seen in the
     * data ({@code stateRow}, -1 if unseen) and, per child in the order of the product, its
     * P(Xj | y, Xi1, Xi2) for the seen states only (row-major, state x value). An unseen state
     * has the Laplace estimate {@code unseen[k]} = 1 / |Xj| for every value. Null when the
     * tables come from {@link #normalizeCounts} (federated learning), which keeps
     * {@link #children}.
     */
    private Compact compact;

    private static final class Compact implements Serializable {
        private static final long serialVersionUID = 1L;
        final int n1, n2;
        final int[] stateRow;
        final int[] ids;
        final int[] cards;
        final double[][] probs;
        final double[] unseen;

        Compact(int n1, int n2, int[] stateRow, int[] ids, int[] cards, double[][] probs, double[] unseen) {
            this.n1 = n1;
            this.n2 = n2;
            this.stateRow = stateRow;
            this.ids = ids;
            this.cards = cards;
            this.probs = probs;
            this.unseen = unseen;
        }
    }

    /**
     * Constructor. Creates an mSP2DE passing it as an argument the name of the two
     * variables xi1 and xi2 that are going to be Super-Parents of the rest of the
     * variables together with the class 'y'.
     *
     * @param xi1 Parent variable 1
     * @param xi2 Parent variable 2
     */
    public mSP2DE(int xi1, int xi2) {
        this.xi1 = xi1;
        this.xi2 = xi2;
        this.listChildren = ConcurrentHashMap.newKeySet();
        this.children = new HashMap<>();
    }

    /**
     * Create the probability tables for the mSP2DE, both the global P(y,Xi) and the
     * conditional for each variable P(Xj|y,Xi).
     */
    @Override
    public void buildTables(mAnDE model) {
        this.globalProbs = new double[model.classNumValues] // y
        [model.varNumValues[xi1]] // Xi1
        [model.varNumValues[xi2]]; // Xi2

        // Creation of the contingency tables, column by column (the counts are integers, so
        // the order in which they are added does not matter).
        int m = model.numInstances;
        int nClass = model.classNumValues;
        int n1 = model.varNumValues[xi1];
        int n2 = model.varNumValues[xi2];
        int[] yc = model.column(model.y);
        int[] x1c = model.column(xi1);
        int[] x2c = model.column(xi2);
        for (int i = 0; i < m; i++) {
            globalProbs[yc[i]][x1c[i]][x2c[i]] += 1;
        }

        // The children get a row only for the states (y, xi1, xi2) seen in the data: an unseen
        // state has a zero row, whose Laplace estimate is 1 / |Xj| for every value.
        int[] stateRow = new int[nClass * n1 * n2];
        Arrays.fill(stateRow, -1);
        int[] stateOf = new int[m];
        int seen = 0;
        for (int i = 0; i < m; i++) {
            int g = (yc[i] * n1 + x1c[i]) * n2 + x2c[i];
            if (stateRow[g] < 0) {
                stateRow[g] = seen++;
            }
            stateOf[i] = stateRow[g];
        }

        // The children in the iteration order of the HashMap the tables used to be kept in,
        // which is the order of the product in probsForInstance.
        HashMap<Integer, Boolean> order = new HashMap<>();
        listChildren.forEach((child) -> order.put(child, Boolean.TRUE));
        int[] ids = new int[order.size()];
        int[] cards = new int[ids.length];
        double[] unseen = new double[ids.length];
        double[][] probs = new double[ids.length][];
        int k = 0;
        for (Integer xj : order.keySet()) {
            ids[k] = xj;
            cards[k] = model.varNumValues[xj];
            k++;
        }
        for (k = 0; k < ids.length; k++) {
            int card = cards[k];
            double[] table = new double[seen * card];
            int[] jc = model.column(ids[k]);
            for (int i = 0; i < m; i++) {
                table[stateOf[i] * card + jc[i]] += 1;
            }
            // Conditional P(Xj | y, Xi1, Xi2) with Laplace smoothing (alpha = 1)
            for (int s0 = 0; s0 < table.length; s0 += card) {
                double rowSum = 0;
                for (int q = 0; q < card; q++) {
                    rowSum += table[s0 + q];
                }
                double denom = rowSum + card;
                for (int q = 0; q < card; q++) {
                    table[s0 + q] = (table[s0 + q] + 1.0) / denom;
                }
            }
            probs[k] = table;
            unseen[k] = (0.0 + 1.0) / (0.0 + card);
        }
        compact = new Compact(n1, n2, stateRow, ids, cards, probs, unseen);
        childIndex = null;

        // Joint P(y, Xi1, Xi2) with Laplace smoothing (alpha = 1)
        int classCard = globalProbs.length;
        int xi1Card   = globalProbs[0].length;
        int xi2Card   = globalProbs[0][0].length;
        double globalDenom = model.numInstances + (double) classCard * xi1Card * xi2Card;
        for (int y = 0; y < globalProbs.length; y++) {
            for (int v1 = 0; v1 < globalProbs[y].length; v1++) {
                for (int v2 = 0; v2 < globalProbs[y][v1].length; v2++) {
                    globalProbs[y][v1][v2] = (globalProbs[y][v1][v2] + 1.0) / globalDenom;
                }
            }
        }
    }

    /**
     * Builds raw count tables P(y,Xi1,Xi2) and P(Xj,y,Xi1,Xi2) from local data
     * without normalizing.
     */
    @Override
    public void buildCountTables(mAnDE model) {
        this.globalCounts = new double[model.classNumValues][model.varNumValues[xi1]][model.varNumValues[xi2]];
        this.childCounts = new HashMap<>();
        listChildren.forEach((child) -> this.childCounts.put(child,
                new double[model.classNumValues][model.varNumValues[xi1]][model.varNumValues[xi2]][model.varNumValues[child]]));
        for (int i = 0; i < model.numInstances; i++) {
            Instance inst = model.data.get(i);
            globalCounts[(int) inst.value(model.y)][(int) inst.value(xi1)][(int) inst.value(xi2)] += 1;
            childCounts.forEach((Integer xj,
                    double[][][][] tableXj) -> tableXj[(int) inst.value(model.y)][(int) inst.value(xi1)][(int) inst
                            .value(xi2)][(int) inst.value(xj)] += 1);
        }
    }

    /**
     * Adds the raw counts from another mSP2DE (same xi1, xi2, same children) to
     * this one.
     */
    @Override
    public void addCounts(mSPnDE other) {
        if (!(other instanceof mSP2DE that)) {
            throw new IllegalArgumentException("Cannot add counts from a non-mSP2DE");
        }
        for (int y = 0; y < globalCounts.length; y++) {
            for (int v1 = 0; v1 < globalCounts[y].length; v1++) {
                for (int v2 = 0; v2 < globalCounts[y][v1].length; v2++) {
                    globalCounts[y][v1][v2] += that.globalCounts[y][v1][v2];
                }
            }
        }
        childCounts.forEach((Integer xj, double[][][][] tableXj) -> {
            double[][][][] thatTable = that.childCounts.get(xj);
            if (thatTable != null) {
                for (int y = 0; y < tableXj.length; y++) {
                    for (int v1 = 0; v1 < tableXj[y].length; v1++) {
                        for (int v2 = 0; v2 < tableXj[y][v1].length; v2++) {
                            for (int k = 0; k < tableXj[y][v1][v2].length; k++) {
                                tableXj[y][v1][v2][k] += thatTable[y][v1][v2][k];
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Adds DP noise to every count cell (globalCounts and childCounts).
     * Negative values produced by the noise are clamped to {@code 0} so
     * downstream normalization with Laplace smoothing remains valid.
     */
    @Override
    public void applyNoise(NumericNoiseGenerator noise) {
        if (globalCounts == null) {
            throw new IllegalStateException("applyNoise called before buildCountTables");
        }
        for (int y = 0; y < globalCounts.length; y++) {
            for (int v1 = 0; v1 < globalCounts[y].length; v1++) {
                for (int v2 = 0; v2 < globalCounts[y][v1].length; v2++) {
                    double n = noise.privatize(globalCounts[y][v1][v2]);
                    globalCounts[y][v1][v2] = Math.max(0.0, n);
                }
            }
        }
        childCounts.forEach((Integer xj, double[][][][] tableXj) -> {
            for (int y = 0; y < tableXj.length; y++) {
                for (int v1 = 0; v1 < tableXj[y].length; v1++) {
                    for (int v2 = 0; v2 < tableXj[y][v1].length; v2++) {
                        for (int k = 0; k < tableXj[y][v1][v2].length; k++) {
                            double n = noise.privatize(tableXj[y][v1][v2][k]);
                            tableXj[y][v1][v2][k] = Math.max(0.0, n);
                        }
                    }
                }
            }
        });
    }

    /**
     * Normalizes the accumulated count tables into probability distributions.
     */
    @Override
    public void normalizeCounts() {
        double totalInstances = 0;
        for (double[][] mat : globalCounts)
            for (double[] row : mat)
                for (double v : row)
                    totalInstances += v;

        int classCard = globalCounts.length;
        int xi1Card   = globalCounts[0].length;
        int xi2Card   = globalCounts[0][0].length;
        double globalDenom = totalInstances + (double) classCard * xi1Card * xi2Card;

        this.globalProbs = new double[globalCounts.length][globalCounts[0].length][globalCounts[0][0].length];
        for (int y = 0; y < globalCounts.length; y++) {
            for (int v1 = 0; v1 < globalCounts[y].length; v1++) {
                for (int v2 = 0; v2 < globalCounts[y][v1].length; v2++) {
                    globalProbs[y][v1][v2] = (globalCounts[y][v1][v2] + 1.0) / globalDenom;
                }
            }
        }
        childCounts.forEach((Integer xj, double[][][][] tableXj) -> {
            double[][][][] condTable = new double[tableXj.length][tableXj[0].length][tableXj[0][0].length][tableXj[0][0][0].length];
            int xjCard = tableXj[0][0][0].length;
            for (int y = 0; y < tableXj.length; y++) {
                for (int v1 = 0; v1 < tableXj[y].length; v1++) {
                    for (int v2 = 0; v2 < tableXj[y][v1].length; v2++) {
                        double rowSum = Utils.sum(tableXj[y][v1][v2]);
                        double denom = rowSum + xjCard;
                        for (int k = 0; k < tableXj[y][v1][v2].length; k++) {
                            condTable[y][v1][v2][k] = (tableXj[y][v1][v2][k] + 1.0) / denom;
                        }
                    }
                }
            }
            this.children.put(xj, condTable);
        });
        childIndex = null;
        compact = null;
    }

    /**
     * Calculates the probabilities for each value of the class given an instance.
     * To do this, the formula is applied: P(y,Xi1,Xi2) * (\prod_{i=1}^{Children}
     * P(Xj|y,Xi1,Xi2)),
     * with Xi1 and Xi2 being the parent variables in the mSP2DE, and Xj each of the
     * child variables.
     *
     * @param inst Instance on which to calculate the class.
     * @return Probabilities for each value of the class for the given instance.
     */
    @Override
    public double[] probsForInstance(Instance inst, mAnDE model) {
        double[] res = new double[model.classNumValues];
        double xi1 = inst.value(this.xi1);
        double xi2 = inst.value(this.xi2);

        // We initialise the probability of each class value to P(y,xi).
        for (int i = 0; i < res.length; i++) {
            res[i] = globalProbs[i][(int) xi1][(int) xi2];
        }

        /*
         * For each child Xj, we multiply P(Xj|y,Xi1,Xi2) by the result
         * accumulated for each of the values of the class
         */
        int v1 = (int) xi1, v2 = (int) xi2;
        Compact c = compact;
        if (c != null) {
            int[] rows = new int[res.length];
            for (int i = 0; i < res.length; i++) {
                rows[i] = c.stateRow[(i * c.n1 + v1) * c.n2 + v2];
            }
            for (int k = 0; k < c.ids.length; k++) {
                double[] table = c.probs[k];
                int card = c.cards[k];
                int xjv = (int) inst.value(c.ids[k]);
                for (int i = 0; i < res.length; i++) {
                    res[i] *= rows[i] < 0 ? c.unseen[k] : table[rows[i] * card + xjv];
                }
            }
        } else {
            ChildIndex index = childIndex;
            if (index == null) {
                index = new ChildIndex(children);
                childIndex = index;
            }
            for (int k = 0; k < index.ids.length; k++) {
                double[][][][] tableXj = index.tables[k];
                int xjv = (int) inst.value(index.ids[k]);
                for (int i = 0; i < res.length; i++) {
                    res[i] *= tableXj[i][v1][v2][xjv];
                }
            }
        }

        // We normalise the values by dividing them by the sum of all the values.
        double sum = Utils.sum(res);
        if (sum != 0) {
            for (int i = 0; i < res.length; i++) {
                res[i] /= sum;
            }
        }

        return res;
    }

    /**
     * Add a variable as a child in the mSP2DE.
     *
     * @param child Name of the variable to add as a child.
     */
    @Override
    public void moreChildren(int child) {
        if ((child != -1) && !(child == xi1) && !(child == xi2)) {
            listChildren.add(child);
        }
    }

    /**
     * Add several variables as children in the mSP2DE.
     *
     * @param children Name of the variables to be added as children.
     */
    @Override
    public void moreChildren(Collection<Integer> children) {
        children.forEach(this::moreChildren);
    }

    /**
     * Returns the children of mSP2DE.
     * 
     * @return The children.
     */
    @Override
    public Set<Integer> getChildren() {
        return listChildren;
    }

    /**
     * Returns the number of children of the mSP2DE.
     * 
     * @return The number of children of the mSP2DE.
     */
    @Override
    public int getNChildren() {
        return listChildren.size();
    }

    @Override
    public mSPnDE copyDeep() {
        mSP2DE copy = new mSP2DE(this.xi1, this.xi2);
        copy.listChildren.addAll(this.listChildren);
        return copy;
    }

    @Override
    public boolean hasProbTables() {
        return globalProbs != null;
    }

    /**
     *
     * @param o Objeto a comparar.
     * @return True si los objetos son iguales y False si no lo son.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof mSP2DE)) {
            return false;
        }

        mSP2DE that = (mSP2DE) o;
        return (this.xi1 == that.xi1 && this.xi2 == that.xi2) ||
                (this.xi1 == that.xi2 && this.xi2 == that.xi1);
    }

    @Override
    public int hashCode() {
        return (this.xi1 << 16) + this.xi2;
    }

    @Override
    public String toString() {
        return "mSP2DE{" + "xi=" + xi1 + "-" + xi2 + ", listChildren=" + listChildren + '}';
    }
}
