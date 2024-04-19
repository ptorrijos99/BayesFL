/*
 *  The MIT License (MIT)
 *  
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
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
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.mAnDE;

import java.io.Serializable;
import java.util.*;

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
    private final HashSet<Integer> listChildren;

    /**
     * Overall probability table of the mSP2DE.
     */
    private double[][][] globalProbs;

    /**
     * Constructor. Creates an mSP2DE passing it as an argument the name of the two variables xi1 and xi2 that are going to be Super-Parents of the rest of the variables together with the class 'y'.
     *
     * @param xi1 Parent variable 1
     * @param xi2 Parent variable 2
     */
    public mSP2DE(int xi1, int xi2) {
        this.xi1 = xi1;
        this.xi2 = xi2;
        this.listChildren = new HashSet<>();
        this.children = new HashMap<>();
    }

    /**
     * Create the probability tables for the mSP2DE, both the global P(y,Xi) and the conditional for each variable P(Xj|y,Xi).
     */
    @Override
    public void buildTables() {
        this.globalProbs = new double[mAnDE.classNumValues] //y
                [mAnDE.varNumValues[xi1]] //Xi1
                [mAnDE.varNumValues[xi2]];    //Xi2

        listChildren.forEach((child) -> {
            this.children.put(child, new double[mAnDE.classNumValues] //y
                    [mAnDE.varNumValues[xi1]] //Xi1
                    [mAnDE.varNumValues[xi2]] //Xi2
                    [mAnDE.varNumValues[child]]); //Xj
        });

        // Creation of contingency tables
        for (int i = 0; i < mAnDE.numInstances; i++) {
            Instance inst = mAnDE.data.get(i);

            // Creation of the probability table P(y,Xi1,Xi2)
            globalProbs[(int) inst.value(mAnDE.y)][(int) inst.value(xi1)][(int) inst.value(xi2)] += 1;

            // Creation of the probability table P(Xj|y,Xi1,Xi2)
            children.forEach((Integer xj, double[][][][] tableXj) -> {
                tableXj[(int) inst.value(mAnDE.y)][(int) inst.value(xi1)][(int) inst.value(xi2)][(int) inst.value(xj)] += 1;
            });
        }

        // Conversion to Joint Probability Distribution
        for (double[][] globalProbs_y : globalProbs) {
            for (double[] globalProbs_y_x1 : globalProbs_y) {
                for (int j = 0; j < globalProbs_y_x1.length; j++) {
                    globalProbs_y_x1[j] /= mAnDE.numInstances;
                }
            }
        }

        // Conversion to Conditional Probability Distribution
        children.forEach((Integer xj, double[][][][] tableXj) -> {
            double sum;
            for (double[][][] tableXj_y : tableXj) {
                for (double[][] tableXj_y_xi1 : tableXj_y) {
                    for (double[] tableXj_y_xi1_xi2 : tableXj_y_xi1) {
                        sum = Utils.sum(tableXj_y_xi1_xi2);
                        if (sum != 0) {
                            for (int k = 0; k < tableXj_y_xi1_xi2.length; k++) {
                                tableXj_y_xi1_xi2[k] /= sum;
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Calculates the probabilities for each value of the class given an instance. To do this, the formula is applied: P(y,Xi1,Xi2) * (\prod_{i=1}^{Children} P(Xj|y,Xi1,Xi2)), with Xi1 and Xi2 being the parent variables in the mSP2DE, and Xj each of the child variables.
     *
     * @param inst Instance on which to calculate the class.
     * @return Probabilities for each value of the class for the given instance.
     */
    @Override
    public double[] probsForInstance(Instance inst) {
        double[] res = new double[mAnDE.classNumValues];
        double xi1 = inst.value(this.xi1);
        double xi2 = inst.value(this.xi2);

        // We initialise the probability of each class value to P(y,xi).
        for (int i = 0; i < res.length; i++) {
            res[i] = globalProbs[i][(int) xi1][(int) xi2];
        }

        /* For each child Xj, we multiply P(Xj|y,Xi1,Xi2) by the result 
         * accumulated for each of the values of the class
         */
        children.forEach((Integer xj, double[][][][] tableXj) -> {
            for (int i = 0; i < res.length; i++) {
                res[i] *= tableXj[i][(int) xi1][(int) xi2][(int) inst.value(xj)];
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
     * @return The children.
     */
    @Override
    public HashSet<Integer> getChildren() {
        return listChildren;
    }

    /**
     * Returns the number of children of the mSP2DE.
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
        return this.xi1<<16 + this.xi2;
    }

    @Override
    public String toString() {
        return "mSP2DE{" + "xi=" + xi1 + "-" + xi2 + ", listChildren=" + listChildren + '}';
    }
}
