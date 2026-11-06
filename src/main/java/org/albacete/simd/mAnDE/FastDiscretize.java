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
package org.albacete.simd.mAnDE;

import weka.core.ContingencyTables;
import weka.core.Instances;
import weka.core.Utils;

/**
 * WEKA's supervised (Fayyad and Irani MDL) discretisation with the cut points computed on
 * primitive arrays.
 *
 * <p>{@link weka.filters.supervised.attribute.Discretize} sorts the whole dataset by every
 * attribute and reads it through {@link weka.core.Instance}. Here every numeric column is copied
 * once with its class labels, sorted by value, and the cut points are searched with the same
 * code and the same {@link ContingencyTables} functions. When every instance weighs 1 the class
 * counts are integers, so the order of instances with equal values does not matter and the cut
 * points are identical; otherwise (and with Kononenko's criterion or the better encoding) the
 * WEKA code is used. The filter is otherwise WEKA's.
 */
public class FastDiscretize extends weka.filters.supervised.attribute.Discretize {

    private static final long serialVersionUID = 1L;

    /** Moves between exact recomputations of the running sum of the cut entropy estimate. */
    private static final int REFRESH = 64;

    /**
     * Safety margin of the estimate of a cut entropy (in bits). Its error is bounded by about
     * (4 REFRESH + 4 numClasses) unit roundoffs of n log n / (n log 2), i.e. below 1e-12 for any
     * realistic n, so 1e-9 leaves a factor of more than 1000.
     */
    private static final double MARGIN = 1e-9;

    @Override
    protected void calculateCutPoints() {
        Instances in = getInputFormat();
        if (getUseKononenko() || getUseBetterEncoding() || !unitWeights(in)) {
            super.calculateCutPoints();
            return;
        }
        int m = in.numInstances();
        int numClasses = in.numClasses();
        int[] cls = new int[m];
        for (int r = 0; r < m; r++) {
            cls[r] = (int) in.instance(r).classValue();
        }
        double[] sortedValues = new double[m];
        int[] sortedClasses = new int[m];

        m_CutPoints = new double[in.numAttributes()][];
        for (int i = in.numAttributes() - 1; i >= 0; i--) {
            if (m_DiscretizeCols.isInRange(i) && in.attribute(i).isNumeric()) {
                int k = 0;
                for (int r = 0; r < m; r++) {
                    double v = in.instance(r).value(i);
                    if (!Double.isNaN(v)) {
                        sortedValues[k] = v;
                        sortedClasses[k++] = cls[r];
                    }
                }
                sortPairs(sortedValues, sortedClasses, 0, k);
                m_CutPoints[i] = cutPointsForSubset(sortedValues, sortedClasses, numClasses, 0, k);
            }
        }
    }

    private static boolean unitWeights(Instances data) {
        for (int r = 0; r < data.numInstances(); r++) {
            if (data.instance(r).weight() != 1.0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Sorts {@code values[lo, hi)} in ascending order, moving each class with its value
     * (quicksort). Equal values may end in any order, which is all the cut point search needs:
     * it only cuts between different values ({@code <}) and its class counts are integers.
     */
    private static void sortPairs(double[] values, int[] classes, int lo, int hi) {
        while (hi - lo > 16) {
            int mid = (lo + hi) >>> 1;
            double a = values[lo], b = values[mid], c = values[hi - 1];
            double pivot = a < b ? (b < c ? b : (a < c ? c : a)) : (a < c ? a : (b < c ? c : b));
            int i = lo, j = hi - 1;
            while (i <= j) {
                while (values[i] < pivot) {
                    i++;
                }
                while (values[j] > pivot) {
                    j--;
                }
                if (i <= j) {
                    double tv = values[i];
                    values[i] = values[j];
                    values[j] = tv;
                    int tc = classes[i];
                    classes[i] = classes[j];
                    classes[j] = tc;
                    i++;
                    j--;
                }
            }
            // Recurse into the smaller part, loop on the larger one.
            if (j + 1 - lo < hi - i) {
                sortPairs(values, classes, lo, j + 1);
                lo = i;
            } else {
                sortPairs(values, classes, i, hi);
                hi = j + 1;
            }
        }
        for (int i = lo + 1; i < hi; i++) {
            double v = values[i];
            int c = classes[i];
            int j = i - 1;
            while (j >= lo && values[j] > v) {
                values[j + 1] = values[j];
                classes[j + 1] = classes[j];
                j--;
            }
            values[j + 1] = v;
            classes[j + 1] = c;
        }
    }

    /** Discretize.cutPointsForSubset on the sorted values and their classes (unit weights). */
    private static double[] cutPointsForSubset(double[] values, int[] classes, int numClasses,
            int first, int lastPlusOne) {
        double[][] counts, bestCounts;
        double[] priorCounts, left, right, cutPoints;
        double currentCutPoint, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
        int bestIndex = -1, numCutPoints;
        double numInstances = 0;

        if ((lastPlusOne - first) < 2) {
            return null;
        }
        counts = new double[2][numClasses];
        for (int i = first; i < lastPlusOne; i++) {
            numInstances += 1.0;
            counts[1][classes[i]] += 1.0;
        }
        priorCounts = new double[numClasses];
        System.arraycopy(counts[1], 0, priorCounts, 0, numClasses);
        priorEntropy = ContingencyTables.entropy(priorCounts);
        bestEntropy = priorEntropy;

        bestCounts = new double[2][numClasses];
        // The exact entropy of a cut (ContingencyTables.entropyConditionedOnRows, 2 x numClasses
        // terms) is only needed when it may beat the best one so far. A running sum of the
        // n log n terms, updated as instances move left and recomputed every REFRESH moves,
        // estimates it within far less than MARGIN (the counts are integers below 2^53 and the
        // estimate is a few hundred roundings away from the exact sum), so a cut whose estimate
        // is MARGIN above the best cannot have a smaller exact entropy and is skipped.
        double scale = numInstances * ContingencyTables.log2;
        double cells = 0;
        for (int j = 0; j < numClasses; j++) {
            cells += ContingencyTables.lnFunc(counts[1][j]);
        }
        double left0 = 0, right0 = numInstances;
        int moves = 0;
        for (int i = first; i < (lastPlusOne - 1); i++) {
            int c = classes[i];
            double l = counts[0][c], r = counts[1][c];
            counts[0][c] += 1.0;
            counts[1][c] -= 1.0;
            left0 += 1.0;
            right0 -= 1.0;
            if (++moves == REFRESH) {
                moves = 0;
                cells = 0;
                for (int j = 0; j < numClasses; j++) {
                    cells += ContingencyTables.lnFunc(counts[0][j]) + ContingencyTables.lnFunc(counts[1][j]);
                }
            } else {
                cells += (ContingencyTables.lnFunc(l + 1.0) - ContingencyTables.lnFunc(l))
                        + (ContingencyTables.lnFunc(r - 1.0) - ContingencyTables.lnFunc(r));
            }
            if (values[i] < values[i + 1]) {
                double estimate = -(cells - ContingencyTables.lnFunc(left0) - ContingencyTables.lnFunc(right0)) / scale;
                if (estimate >= bestEntropy + MARGIN) {
                    continue;
                }
                currentCutPoint = (values[i] + values[i + 1]) / 2.0;
                currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
                if (currentEntropy < bestEntropy) {
                    bestCutPoint = currentCutPoint;
                    bestEntropy = currentEntropy;
                    bestIndex = i;
                    System.arraycopy(counts[0], 0, bestCounts[0], 0, numClasses);
                    System.arraycopy(counts[1], 0, bestCounts[1], 0, numClasses);
                }
            }
        }
        // The default (worse) encoding.
        numCutPoints = (lastPlusOne - first) - 1;

        gain = priorEntropy - bestEntropy;
        if (gain <= 0) {
            return null;
        }
        if (!fayyadAndIranisMDL(priorCounts, bestCounts, numInstances, numCutPoints)) {
            return null;
        }
        left = cutPointsForSubset(values, classes, numClasses, first, bestIndex + 1);
        right = cutPointsForSubset(values, classes, numClasses, bestIndex + 1, lastPlusOne);
        if ((left == null) && (right) == null) {
            cutPoints = new double[1];
            cutPoints[0] = bestCutPoint;
        } else if (right == null) {
            cutPoints = new double[left.length + 1];
            System.arraycopy(left, 0, cutPoints, 0, left.length);
            cutPoints[left.length] = bestCutPoint;
        } else if (left == null) {
            cutPoints = new double[1 + right.length];
            cutPoints[0] = bestCutPoint;
            System.arraycopy(right, 0, cutPoints, 1, right.length);
        } else {
            cutPoints = new double[left.length + right.length + 1];
            System.arraycopy(left, 0, cutPoints, 0, left.length);
            cutPoints[left.length] = bestCutPoint;
            System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
        }
        return cutPoints;
    }

    /** Discretize.FayyadAndIranisMDL. */
    private static boolean fayyadAndIranisMDL(double[] priorCounts, double[][] bestCounts,
            double numInstances, int numCutPoints) {
        double priorEntropy, entropy, gain;
        double entropyLeft, entropyRight, delta;
        int numClassesTotal, numClassesRight, numClassesLeft;

        priorEntropy = ContingencyTables.entropy(priorCounts);
        entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);
        gain = priorEntropy - entropy;

        numClassesTotal = 0;
        for (double priorCount : priorCounts) {
            if (priorCount > 0) {
                numClassesTotal++;
            }
        }
        numClassesLeft = 0;
        for (int i = 0; i < bestCounts[0].length; i++) {
            if (bestCounts[0][i] > 0) {
                numClassesLeft++;
            }
        }
        numClassesRight = 0;
        for (int i = 0; i < bestCounts[1].length; i++) {
            if (bestCounts[1][i] > 0) {
                numClassesRight++;
            }
        }
        entropyLeft = ContingencyTables.entropy(bestCounts[0]);
        entropyRight = ContingencyTables.entropy(bestCounts[1]);

        delta = Utils.log2(Math.pow(3, numClassesTotal) - 2)
                - ((numClassesTotal * priorEntropy) - (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));
        return (gain > (Utils.log2(numCutPoints) + delta) / numInstances);
    }
}
