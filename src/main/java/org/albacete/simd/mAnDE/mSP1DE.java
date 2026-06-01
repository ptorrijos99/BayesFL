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
 *    mSP1DE.java
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

public class mSP1DE implements mSPnDE, Serializable {

    /**
     * ID of the Super-Parent of the mSP1DE.
     */
    private final int xi;

    /**
     * Link the id of the children of the mSP1DE with their probability table.
     */
    private final HashMap<Integer, double[][][]> children;

    /**
     * List of children of the mSP1DE.
     */
    private final Set<Integer> listChildren;

    /**
     * Global probability table of the mSP1DE.
     */
    private double[][] globalProb;

    /**
     * Raw count table P(y, Xi): used for federated parameter aggregation.
     */
    private double[][] globalCounts;

    /**
     * Raw count tables P(Xj, y, Xi) per child: used for federated parameter
     * aggregation.
     */
    private HashMap<Integer, double[][][]> childCounts;

    /**
     * Constructor. Build to mSP1DE passing it as argument the name of the variable
     * xi that is going to be Super-Parent of the rest of the variables next to the
     * class 'y'.
     * 
     * @param xi Father variable
     */
    public mSP1DE(int xi) {
        this.xi = xi;
        this.listChildren = ConcurrentHashMap.newKeySet();
        this.children = new HashMap<>();
    }

    /**
     * Create the probability tables for the mSP1DE, both the global P(y,Xi) and
     * the conditional for each variable P(Xj|y,Xi).
     */
    @Override
    public void buildTables(mAnDE model) {
        this.globalProb = new double[model.classNumValues] // y
        [model.varNumValues[xi]]; // Xi

        listChildren.forEach((child) -> {
            this.children.put(child, new double[model.classNumValues] // y
            [model.varNumValues[xi]] // Xi
            [model.varNumValues[child]]); // Xj
        });

        // Creation of the contingency tables
        for (int i = 0; i < model.numInstances; i++) {
            Instance inst = model.data.get(i);

            // Creation of the probability table P(y,Xi)
            globalProb[(int) inst.value(model.y)][(int) inst.value(xi)] += 1;

            // Creation of the probability table P(y,Xi)
            children.forEach((Integer xj, double[][][] tablaXj) -> {
                tablaXj[(int) inst.value(model.y)][(int) inst.value(xi)][(int) inst.value(xj)] += 1;
            });
        }

        // Joint P(y, Xi) with Laplace smoothing (alpha = 1)
        int classCard = globalProb.length;
        int xiCard    = globalProb[0].length;
        double globalDenom = model.numInstances + (double) classCard * xiCard;
        for (int y = 0; y < globalProb.length; y++) {
            for (int j = 0; j < globalProb[y].length; j++) {
                globalProb[y][j] = (globalProb[y][j] + 1.0) / globalDenom;
            }
        }

        // Conditional P(Xj | y, Xi) with Laplace smoothing (alpha = 1)
        children.forEach((Integer xj, double[][][] tableXj) -> {
            int xjCard = tableXj[0][0].length;
            for (double[][] tableXj_y : tableXj) {
                for (double[] tableXj_y_xi : tableXj_y) {
                    double rowSum = Utils.sum(tableXj_y_xi);
                    double denom = rowSum + xjCard;
                    for (int k = 0; k < tableXj_y_xi.length; k++) {
                        tableXj_y_xi[k] = (tableXj_y_xi[k] + 1.0) / denom;
                    }
                }
            }
        });
    }

    /**
     * Builds raw count tables P(y,Xi) and P(Xj,y,Xi) from local data without
     * normalizing.
     * Used in federated parameter learning so that counts can be aggregated across
     * clients
     * before a single global normalization step.
     */
    @Override
    public void buildCountTables(mAnDE model) {
        this.globalCounts = new double[model.classNumValues][model.varNumValues[xi]];
        this.childCounts = new HashMap<>();
        listChildren.forEach((child) -> this.childCounts.put(child,
                new double[model.classNumValues][model.varNumValues[xi]][model.varNumValues[child]]));
        for (int i = 0; i < model.numInstances; i++) {
            Instance inst = model.data.get(i);
            globalCounts[(int) inst.value(model.y)][(int) inst.value(xi)] += 1;
            childCounts.forEach((Integer xj,
                    double[][][] tableXj) -> tableXj[(int) inst.value(model.y)][(int) inst.value(xi)][(int) inst
                            .value(xj)] += 1);
        }
    }

    /**
     * Adds the raw counts from another mSP1DE (same xi, same children) to this one.
     */
    @Override
    public void addCounts(mSPnDE other) {
        if (!(other instanceof mSP1DE that)) {
            throw new IllegalArgumentException("Cannot add counts from a non-mSP1DE");
        }
        for (int y = 0; y < globalCounts.length; y++) {
            for (int v = 0; v < globalCounts[y].length; v++) {
                globalCounts[y][v] += that.globalCounts[y][v];
            }
        }
        childCounts.forEach((Integer xj, double[][][] tableXj) -> {
            double[][][] thatTable = that.childCounts.get(xj);
            if (thatTable != null) {
                for (int y = 0; y < tableXj.length; y++) {
                    for (int v = 0; v < tableXj[y].length; v++) {
                        for (int k = 0; k < tableXj[y][v].length; k++) {
                            tableXj[y][v][k] += thatTable[y][v][k];
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
            for (int v = 0; v < globalCounts[y].length; v++) {
                double n = noise.privatize(globalCounts[y][v]);
                globalCounts[y][v] = Math.max(0.0, n);
            }
        }
        childCounts.forEach((Integer xj, double[][][] tableXj) -> {
            for (int y = 0; y < tableXj.length; y++) {
                for (int v = 0; v < tableXj[y].length; v++) {
                    for (int k = 0; k < tableXj[y][v].length; k++) {
                        double n = noise.privatize(tableXj[y][v][k]);
                        tableXj[y][v][k] = Math.max(0.0, n);
                    }
                }
            }
        });
    }

    /**
     * Normalizes the accumulated count tables into probability distributions.
     * Converts globalCounts to globalProb (joint) and childCounts to children
     * (conditional).
     */
    @Override
    public void normalizeCounts() {
        double totalInstances = 0;
        for (double[] row : globalCounts)
            for (double v : row)
                totalInstances += v;

        int classCard = globalCounts.length;
        int xiCard    = globalCounts[0].length;
        double globalDenom = totalInstances + (double) classCard * xiCard;

        this.globalProb = new double[globalCounts.length][globalCounts[0].length];
        for (int y = 0; y < globalCounts.length; y++) {
            for (int v = 0; v < globalCounts[y].length; v++) {
                globalProb[y][v] = (globalCounts[y][v] + 1.0) / globalDenom;
            }
        }
        childCounts.forEach((Integer xj, double[][][] tableXj) -> {
            double[][][] condTable = new double[tableXj.length][tableXj[0].length][tableXj[0][0].length];
            int xjCard = tableXj[0][0].length;
            for (int y = 0; y < tableXj.length; y++) {
                for (int v = 0; v < tableXj[y].length; v++) {
                    double rowSum = Utils.sum(tableXj[y][v]);
                    double denom = rowSum + xjCard;
                    for (int k = 0; k < tableXj[y][v].length; k++) {
                        condTable[y][v][k] = (tableXj[y][v][k] + 1.0) / denom;
                    }
                }
            }
            this.children.put(xj, condTable);
        });
    }

    /**
     * Calculates the probabilities for each value of the class given an instance.
     * To do this, the formula is applied: P(y,Xi) * (\prod_{i=1}^{Children}
     * P(Xj|y,Xi)), with Xi being the parent variable in the mSP1DE, and Xj each of
     * the child variables.
     *
     * @param inst Instance on which to compute the class.
     * @return Probabilities for each value of the class for the given instance.
     */
    @Override
    public double[] probsForInstance(Instance inst, mAnDE model) {
        double[] res = new double[model.classNumValues];
        double xi = inst.value(this.xi);

        // We initialise the probability of each class value to P(y,xi).
        for (int i = 0; i < res.length; i++) {
            res[i] = globalProb[i][(int) xi];
        }

        /*
         * For each child Xj, we multiply P(Xj|y,Xi) by the result
         * accumulated for each of the values of the class
         */
        children.forEach((Integer xj, double[][][] tablaXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tablaXj[i][(int) xi][(int) inst.value(xj)];
            }
        });

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
     * Add a variable as a child in the mSP1DE.
     *
     * @param child Name of the variable to be added as a child.
     */
    @Override
    public void moreChildren(int child) {
        if ((child != -1) && !(child == xi)) {
            listChildren.add(child);
        }
    }

    /**
     * Add several variables as children in the mSP1DE.
     *
     * @param children Name of the variables to be added as children.
     */
    @Override
    public void moreChildren(Collection<Integer> children) {
        children.forEach(this::moreChildren);
    }

    /**
     * Returns the children of mSP1DE.
     * 
     * @return The children.
     */
    @Override
    public Set<Integer> getChildren() {
        return listChildren;
    }

    /**
     * Returns the number of children of mSP1DE.
     * 
     * @return The number of children
     */
    @Override
    public int getNChildren() {
        return listChildren.size();
    }

    @Override
    public mSPnDE copyDeep() {
        mSP1DE copy = new mSP1DE(this.xi);
        copy.listChildren.addAll(this.listChildren);
        return copy;
    }

    @Override
    public boolean hasProbTables() {
        return globalProb != null;
    }

    /**
     *
     * @param o Object to compare.
     * @return True if the objects are equal and False if they are not.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }
        if (!(o instanceof mSP1DE that)) {
            return false;
        }

        return this.xi == that.xi;
    }

    @Override
    public int hashCode() {
        return this.xi;
    }

    @Override
    public String toString() {
        return "mSP1DE{" + "xi=" + xi + ", listChildren=" + listChildren + '}';
    }

}
