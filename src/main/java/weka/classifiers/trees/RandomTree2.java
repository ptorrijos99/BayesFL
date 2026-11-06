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
 *    RandomTree2.java
 *    Copyright (C) 2022 Universidad de Castilla-La Mancha, España
 *    @author Pablo Torrijos Arenas
 *
 */

package weka.classifiers.trees;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import org.albacete.simd.mAnDE.NodeInt;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import org.albacete.simd.mAnDE.mSP1DE;
import org.albacete.simd.mAnDE.mSPnDE;


public class RandomTree2 extends RandomTree {

    /**
     * The size of each bag sample, as a percentage of the training size
     *
     * Changued from int to double
     */
    protected double m_BagSizePercentDouble = 100;


    /**
     * Sets the size of each bag, as a percentage of the training set size.
     *
     * @param newBagSizePercentDouble the bag size, as a percentage.
     */
    public void setBagSizePercentDouble(double newBagSizePercentDouble) {

        m_BagSizePercentDouble = newBagSizePercentDouble;
    }

    /**
     * Gets the size of each bag, as a percentage of the training set size.
     *
     * @return the bag size, as a percentage.
     */
    public double getBagSizePercentDouble() {

        return m_BagSizePercentDouble;
    }

    /**
     * Rewrites the split attributes of the tree through {@code map} (index in the data the
     * tree was learnt on -> index in the full data), for trees learnt on a column subset.
     */
    public void remapAttributes(int[] map) {
        LinkedList<Tree> tbExplored = new LinkedList<>();
        tbExplored.add(m_Tree);
        while (!tbExplored.isEmpty()) {
            Tree node = tbExplored.poll();
            if (node.m_Attribute == -1) {
                continue;
            }
            node.m_Attribute = map[node.m_Attribute];
            if (node.m_Successors != null) {
                for (Tree successor : node.m_Successors) {
                    tbExplored.add(successor);
                }
            }
        }
    }

    /**
     * The structure of the trees that {@code Bagging2} (seed {@code seed}, {@code nTrees}
     * iterations, bags of {@code bagSizePercent}%) with this class as base learner and
     * K = all attributes learns on {@code data}, built directly.
     *
     * <p>With K equal to the number of attributes, {@link RandomTree} evaluates every attribute
     * at every node and keeps the highest gain, breaking exact ties by the lowest index, so the
     * order in which its random generator visits them does not change the tree. Only the bags
     * (Walker sampling with {@code Debug.Random(seed + iteration)}, as in
     * {@link Instances#resampleWithWeights}) do. The split attributes, the leaves and the order of
     * every floating point sum (instance weights included, also for the fractional copies of
     * instances with a missing split value) are those of {@link RandomTree}, and the gain is
     * computed with the same {@link ContingencyTables} functions, so the trees are identical.
     * The leaves keep the class counts; split points are not set (nominal splits only).
     *
     * <p>All attributes must be nominal (as after a supervised discretisation); only the
     * attributes in {@code atts} (ascending indices, class excluded) are considered for splits.
     * The instances are stored once, column by column, and every bag is a list of row indices,
     * which avoids copying the data for every tree and the per-value costs of {@link Instance}.
     *
     * <p>Shortcuts that cannot change a decision: an attribute with a single value in a node has
     * a gain of exactly 0 and is not scored; without missing values every weight is an integer,
     * so the copies of a bagged row are counted together, the counts are integers (exact in any
     * order), such an attribute stays single-valued in the whole subtree and is not evaluated
     * there again, and the prior entropy (the same class counts for every attribute) is computed
     * once per node.
     */
    public static RandomTree2[] baggedStructures(Instances data, int[] atts, int nTrees,
            double bagSizePercent, int seed) {
        int m = data.numInstances();
        int classIndex = data.classIndex();
        int numClasses = data.numClasses();
        int[] cls = new int[m];
        double[] weights = new double[m];
        for (int i = 0; i < m; i++) {
            cls[i] = (int) data.instance(i).classValue();
            weights[i] = data.instance(i).weight();
        }
        int[][] col = new int[data.numAttributes()][];
        int[] numValues = new int[data.numAttributes()];
        boolean anyMissing = false;
        for (int a : atts) {
            if (a == classIndex || !data.attribute(a).isNominal()) {
                throw new IllegalArgumentException("nominal attributes only");
            }
            numValues[a] = data.attribute(a).numValues();
            int[] c = new int[m];
            for (int i = 0; i < m; i++) {
                Instance inst = data.instance(i);
                if (inst.isMissing(a)) {
                    c[i] = -1;
                    anyMissing = true;
                } else {
                    c[i] = (int) inst.value(a);
                }
            }
            col[a] = c;
        }
        // Without missing values and with at most 127 values per attribute, the columns are kept
        // as bytes: four times less memory to read at every node.
        byte[][] bcol = null;
        if (!anyMissing) {
            int maxValues = 0;
            for (int a : atts) {
                maxValues = Math.max(maxValues, numValues[a]);
            }
            if (maxValues <= Byte.MAX_VALUE) {
                bcol = new byte[data.numAttributes()][];
                for (int a : atts) {
                    byte[] b = new byte[m];
                    for (int i = 0; i < m; i++) {
                        b[i] = (byte) col[a][i];
                    }
                    bcol[a] = b;
                    col[a] = null;
                }
            }
        }
        StructureBuilder builder = new StructureBuilder(col, bcol, numValues, cls, numClasses, atts, anyMissing);

        RandomTree2[] trees = new RandomTree2[nTrees];
        for (int it = 0; it < nTrees; it++) {
            int[] bag = bagIndices(weights, new weka.core.Debug.Random(seed + it), bagSizePercent);
            double[] w;
            if (anyMissing) {
                w = new double[bag.length];
                Arrays.fill(w, 1);
            } else {
                // Without missing values every weight stays an integer (1 per copy), so the
                // sums are exact in any order and the copies of a row can be counted together:
                // each distinct row once, weighted by its number of copies.
                int[] copies = new int[m];
                int distinct = 0;
                for (int r : bag) {
                    if (copies[r]++ == 0) {
                        distinct++;
                    }
                }
                int[] rows = new int[distinct];
                w = new double[distinct];
                int k = 0;
                for (int r = 0; r < m; r++) {
                    if (copies[r] > 0) {
                        rows[k] = r;
                        w[k++] = copies[r];
                    }
                }
                bag = rows;
            }
            double[] classProbs = new double[numClasses];
            for (int i = 0; i < bag.length; i++) {
                classProbs[cls[bag[i]]] += w[i];
            }
            RandomTree2 tree = new RandomTree2();
            tree.m_Tree = tree.new Tree();
            builder.build(tree, tree.m_Tree, bag, w, classProbs, atts);
            trees[it] = tree;
        }
        return trees;
    }

    /** The rows that {@link Instances#resampleWithWeights} draws, in the order it adds them. */
    static int[] bagIndices(double[] weights, Random random, double sampleSize) {
        int M = weights.length;
        if (M == 0) {
            return new int[0];
        }
        double[] P = weights.clone();
        Utils.normalize(P);
        double[] Q = new double[M];
        int[] A = new int[M];
        int[] W = new int[M];
        int NN = -1;
        int NP = M;
        for (int I = 0; I < M; I++) {
            if (P[I] < 0) {
                throw new IllegalArgumentException("Weights have to be positive.");
            }
            Q[I] = M * P[I];
            if (Q[I] < 1.0) {
                W[++NN] = I;
            } else {
                W[--NP] = I;
            }
        }
        if (NN > -1 && NP < M) {
            for (int S = 0; S < M - 1; S++) {
                int I = W[S];
                int J = W[NP];
                A[I] = J;
                Q[J] += Q[I] - 1.0;
                if (Q[J] < 1.0) {
                    NP++;
                }
                if (NP >= M) {
                    break;
                }
            }
        }
        for (int I = 0; I < M; I++) {
            Q[I] += I;
        }
        int numToBeSampled = (int) (M * (sampleSize / 100.0));
        int[] bag = new int[numToBeSampled];
        for (int i = 0; i < numToBeSampled; i++) {
            double U = M * random.nextDouble();
            int I = (int) U;
            bag[i] = U < Q[I] ? I : A[I];
        }
        return bag;
    }

    /** RandomTree.Tree.buildTree for nominal data and K = all attributes, structure only. */
    private static final class StructureBuilder {
        private final int[][] col;
        private final byte[][] bcol;
        private final int[] numValues;
        private final int[] cls;
        private final int numClasses;
        private final int[] atts;
        private final boolean anyMissing;
        private final double minNum = 1.0; // RandomTree's default

        /** Work buffers of the class counts per value of an attribute (row-major, value x class). */
        private double[] current;
        private double[] best;
        private int[] counts;
        private int[] bestCounts;

        StructureBuilder(int[][] col, byte[][] bcol, int[] numValues, int[] cls, int numClasses, int[] atts,
                boolean anyMissing) {
            this.col = col;
            this.bcol = bcol;
            this.numValues = numValues;
            this.cls = cls;
            this.numClasses = numClasses;
            this.atts = atts;
            this.anyMissing = anyMissing;
            int maxValues = 1;
            for (int a : atts) {
                maxValues = Math.max(maxValues, numValues[a]);
            }
            current = new double[maxValues * numClasses];
            best = new double[maxValues * numClasses];
            counts = new int[2 * maxValues * numClasses];
            bestCounts = new int[maxValues * numClasses];
        }

        /** n log n of the integers below 10000, the cache of ContingencyTables.lnFunc. */
        private static final double[] LN = new double[10000];

        static {
            for (int i = 0; i < LN.length; i++) {
                LN[i] = ContingencyTables.lnFunc(i);
            }
        }

        /** ContingencyTables.lnFunc of an integer count. */
        private static double ln(int x) {
            return x < LN.length ? LN[x] : ContingencyTables.lnFunc(x);
        }

        /** Number of rows of the integer table {@code h} with a non-zero cell. */
        private static int nonEmptyRows(int[] h, int rows, int cols) {
            int k = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (h[i * cols + j] != 0) {
                        k++;
                        break;
                    }
                }
            }
            return k;
        }

        /**
         * {@link #entropyConditionedOnRows(double[], int, int)} of a table of integer counts: the
         * same terms added in the same order (the row sums are exact integers either way).
         */
        private static double entropyConditionedOnRows(int[] h, int rows, int cols) {
            double returnValue = 0, total = 0;
            for (int i = 0; i < rows; i++) {
                int sumForRow = 0;
                for (int j = 0; j < cols; j++) {
                    int x = h[i * cols + j];
                    if (x != 0) {
                        returnValue = returnValue + ln(x);
                        sumForRow += x;
                    }
                }
                if (sumForRow != 0) {
                    returnValue = returnValue - ln(sumForRow);
                    total += sumForRow;
                }
            }
            if (Utils.eq(total, 0)) {
                return 0;
            }
            return -returnValue / (total * ContingencyTables.log2);
        }

        /** {@link #entropyOverColumns(double[], int, int)} of a table of integer counts. */
        private static double entropyOverColumns(int[] h, int rows, int cols) {
            double returnValue = 0, total = 0;
            for (int j = 0; j < cols; j++) {
                int sumForColumn = 0;
                for (int i = 0; i < rows; i++) {
                    sumForColumn += h[i * cols + j];
                }
                returnValue = returnValue - ln(sumForColumn);
                total += sumForColumn;
            }
            if (Utils.eq(total, 0)) {
                return 0;
            }
            return (returnValue + ln((int) total)) / (total * ContingencyTables.log2);
        }

        /** Utils.sum of every row of the value x class table {@code m}. */
        private static double[] rowSums(double[] m, int rows, int cols) {
            double[] sums = new double[rows];
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum += m[i * cols + j];
                }
                sums[i] = sum;
            }
            return sums;
        }

        /** Number of rows of {@code m} with a non-zero cell. */
        private static int nonEmptyRows(double[] m, int rows, int cols) {
            int k = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (m[i * cols + j] != 0) {
                        k++;
                        break;
                    }
                }
            }
            return k;
        }

        /** ContingencyTables.entropyConditionedOnRows on the row-major table {@code m}. */
        private static double entropyConditionedOnRows(double[] m, int rows, int cols) {
            double returnValue = 0, sumForRow, total = 0;
            // A zero cell adds lnFunc(0) = +0, which leaves the (never -0) sum unchanged, so
            // only the non-zero cells are visited.
            for (int i = 0; i < rows; i++) {
                sumForRow = 0;
                for (int j = 0; j < cols; j++) {
                    double x = m[i * cols + j];
                    if (x != 0) {
                        returnValue = returnValue + ContingencyTables.lnFunc(x);
                        sumForRow += x;
                    }
                }
                if (sumForRow != 0) {
                    returnValue = returnValue - ContingencyTables.lnFunc(sumForRow);
                    total += sumForRow;
                }
            }
            if (Utils.eq(total, 0)) {
                return 0;
            }
            return -returnValue / (total * ContingencyTables.log2);
        }

        /** ContingencyTables.entropyOverColumns on the row-major table {@code m}. */
        private static double entropyOverColumns(double[] m, int rows, int cols) {
            double returnValue = 0, sumForColumn, total = 0;
            for (int j = 0; j < cols; j++) {
                sumForColumn = 0;
                for (int i = 0; i < rows; i++) {
                    sumForColumn += m[i * cols + j];
                }
                returnValue = returnValue - ContingencyTables.lnFunc(sumForColumn);
                total += sumForColumn;
            }
            if (Utils.eq(total, 0)) {
                return 0;
            }
            return (returnValue + ContingencyTables.lnFunc(total)) / (total * ContingencyTables.log2);
        }

        /**
         * Grows {@code node} on the rows {@code idx} (with weights {@code w}, in the order the
         * instances have in RandomTree's subset) and the class distribution {@code classProbs}.
         */
        void build(RandomTree2 owner, Tree node, int[] idx, double[] w, double[] classProbs, int[] candidates) {
            int n = idx.length;
            if (n == 0) {
                node.m_Attribute = -1;
                node.m_ClassDistribution = null;
                node.m_Prop = null;
                return;
            }
            double totalWeight = Utils.sum(classProbs);
            if (totalWeight < 2 * minNum
                    || Utils.eq(classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs))) {
                node.m_Attribute = -1;
                node.m_ClassDistribution = classProbs.clone();
                node.m_Prop = null;
                return;
            }

            double val = -Double.MAX_VALUE;
            int bestIndex = 0;
            int bestNv = 0;
            double[] bestProps = null;
            int[] rowCls = new int[n];
            for (int i = 0; i < n; i++) {
                rowCls[i] = cls[idx[i]];
            }
            int[] iw = null;
            if (!anyMissing) {
                iw = new int[n];
                for (int i = 0; i < n; i++) {
                    iw[i] = (int) w[i];
                }
            }
            int nc = numClasses;
            double nodePrior = Double.NaN;
            double[] cur = current;
            double[] top = best;
            // Without missing values, an attribute with a single value in this node has a single
            // value in every node below it (gain exactly 0), so the children only evaluate the
            // attributes that still have two values here ("alive"), in the same order.
            int[] alive = anyMissing ? null : new int[candidates.length];
            int nAlive = 0;
            for (int att : candidates) {
                int nv = numValues[att];
                if (!anyMissing) {
                    // Integer weights: integer counts, in two interleaved tables so that
                    // consecutive rows do not wait on the same cell. Exact in any order, and the
                    // entropies are computed on the integers with the terms of the double version.
                    int len = nv * nc;
                    int[] h = counts;
                    Arrays.fill(h, 0, 2 * len, 0);
                    int i = 0;
                    if (bcol != null) {
                        byte[] c = bcol[att];
                        for (; i + 1 < n; i += 2) {
                            h[c[idx[i]] * nc + rowCls[i]] += iw[i];
                            h[len + c[idx[i + 1]] * nc + rowCls[i + 1]] += iw[i + 1];
                        }
                        if (i < n) {
                            h[c[idx[i]] * nc + rowCls[i]] += iw[i];
                        }
                    } else {
                        int[] c = col[att];
                        for (; i + 1 < n; i += 2) {
                            h[c[idx[i]] * nc + rowCls[i]] += iw[i];
                            h[len + c[idx[i + 1]] * nc + rowCls[i + 1]] += iw[i + 1];
                        }
                        if (i < n) {
                            h[c[idx[i]] * nc + rowCls[i]] += iw[i];
                        }
                    }
                    for (int k = 0; k < len; k++) {
                        h[k] += h[len + k];
                    }
                    // With a single non-empty value the gain is exactly 0 (both entropies are
                    // the same sums), so such an attribute can never be the split, which needs
                    // a gain above Utils.SMALL: skip the entropies.
                    if (nonEmptyRows(h, nv, nc) <= 1) {
                        continue;
                    }
                    alive[nAlive++] = att;
                    // The column sums are the class counts of the node, the same for every
                    // attribute, so the prior entropy is computed once.
                    if (Double.isNaN(nodePrior)) {
                        nodePrior = entropyOverColumns(h, nv, nc);
                    }
                    double currVal = nodePrior - entropyConditionedOnRows(h, nv, nc);
                    if (currVal > val || (currVal == val && att < bestIndex)) {
                        val = currVal;
                        bestIndex = att;
                        bestNv = nv;
                        System.arraycopy(h, 0, bestCounts, 0, len);
                    }
                    continue;
                }

                int[] c = col[att];
                Arrays.fill(cur, 0, nv * nc, 0);
                int firstMissing = n;
                for (int i = 0; i < n; i++) {
                    int v = c[idx[i]];
                    if (v < 0) {
                        if (firstMissing == n) {
                            firstMissing = i;
                        }
                        continue;
                    }
                    cur[v * nc + rowCls[i]] += w[i];
                }
                double[] props = null;
                if (firstMissing < n) {
                    props = rowSums(cur, nv, nc);
                    if (Utils.eq(Utils.sum(props), 0)) {
                        for (int k = 0; k < props.length; k++) {
                            props[k] = 1.0 / props.length;
                        }
                    } else {
                        Utils.normalize(props);
                    }
                    for (int i = firstMissing; i < n; i++) {
                        if (c[idx[i]] < 0) {
                            int y = rowCls[i];
                            for (int j = 0; j < nv; j++) {
                                cur[j * nc + y] += props[j] * w[i];
                            }
                        }
                    }
                }
                if (nonEmptyRows(cur, nv, nc) <= 1) {
                    continue;
                }
                double currVal = entropyOverColumns(cur, nv, nc) - entropyConditionedOnRows(cur, nv, nc);
                if (currVal > val || (currVal == val && att < bestIndex)) {
                    val = currVal;
                    bestIndex = att;
                    bestNv = nv;
                    bestProps = props;
                    double[] t = cur;
                    cur = top;
                    top = t;
                }
            }
            current = cur;
            best = top;

            if (!Utils.gr(val, 0)) {
                node.m_Attribute = -1;
                node.m_ClassDistribution = classProbs.clone();
                return;
            }
            // The distribution of the split as RandomTree's dist, and its props as
            // distribution() computes them when there were no missing values.
            double[][] bestDist = new double[bestNv][nc];
            for (int k = 0; k < bestNv; k++) {
                for (int y = 0; y < nc; y++) {
                    bestDist[k][y] = anyMissing ? top[k * nc + y] : bestCounts[k * nc + y];
                }
            }
            if (bestProps == null) {
                bestProps = new double[bestNv];
                for (int k = 0; k < bestNv; k++) {
                    bestProps[k] = Utils.sum(bestDist[k]);
                }
                if (Utils.eq(Utils.sum(bestProps), 0)) {
                    for (int k = 0; k < bestNv; k++) {
                        bestProps[k] = 1.0 / bestNv;
                    }
                } else {
                    Utils.normalize(bestProps);
                }
            }

            node.m_Attribute = bestIndex;
            node.m_Prop = bestProps;

            // Subsets of the split, in RandomTree.splitData's order.
            int nv = bestProps.length;
            int[] split = new int[n];
            for (int i = 0; i < n; i++) {
                split[i] = bcol != null ? bcol[bestIndex][idx[i]] : col[bestIndex][idx[i]];
            }
            int[] size = new int[nv];
            for (int i = 0; i < n; i++) {
                int v = split[i];
                if (v >= 0) {
                    size[v]++;
                } else {
                    for (int k = 0; k < nv; k++) {
                        if (bestProps[k] > 0) {
                            size[k]++;
                        }
                    }
                }
            }
            int[][] subIdx = new int[nv][];
            double[][] subW = new double[nv][];
            for (int k = 0; k < nv; k++) {
                subIdx[k] = new int[size[k]];
                subW[k] = new double[size[k]];
            }
            int[] pos = new int[nv];
            for (int i = 0; i < n; i++) {
                int v = split[i];
                if (v >= 0) {
                    subIdx[v][pos[v]] = idx[i];
                    subW[v][pos[v]++] = w[i];
                } else {
                    for (int k = 0; k < nv; k++) {
                        if (bestProps[k] > 0) {
                            subIdx[k][pos[k]] = idx[i];
                            subW[k][pos[k]++] = bestProps[k] * w[i];
                        }
                    }
                }
            }
            int[] childCandidates = candidates;
            if (alive != null) {
                // The split attribute has a single value in every child.
                int k2 = 0;
                for (int t = 0; t < nAlive; t++) {
                    if (alive[t] != bestIndex) {
                        alive[k2++] = alive[t];
                    }
                }
                childCandidates = Arrays.copyOf(alive, k2);
            }
            node.m_Successors = new Tree[nv];
            boolean emptySuccessor = false;
            for (int k = 0; k < nv; k++) {
                node.m_Successors[k] = owner.new Tree();
                build(owner, node.m_Successors[k], subIdx[k], subW[k], bestDist[k], childCandidates);
                if (node.m_Successors[k].m_ClassDistribution == null) {
                    emptySuccessor = true;
                }
            }
            if (emptySuccessor) {
                node.m_ClassDistribution = classProbs.clone();
            }
        }
    }

    public void toSP1DE(ConcurrentHashMap<Object, mSPnDE> mSPnDEs) {
        // If the tree is empty, return
        if (m_Tree.m_Attribute == -1) {
            return;
        }

        // Trees to be explored
        LinkedList<Tree> tbExplored = new LinkedList<>();
        tbExplored.add(m_Tree);
        mSPnDEs.putIfAbsent(m_Tree.m_Attribute, new mSP1DE(m_Tree.m_Attribute));

        while (!tbExplored.isEmpty()) {
            Tree node = tbExplored.poll();

            int id = node.m_Attribute;
            mSPnDE mSPnDE = mSPnDEs.get(id);

            // Add childs to the mSPnDE
            for (Tree m_Successor : node.m_Successors) {
                int child_id = m_Successor.m_Attribute;

                // If is not a leaf
                if (child_id != -1) {

                    // Add child
                    mSPnDE.moreChildren(child_id);

                    // Create child if neccesary and add parent as child
                    mSPnDEs.putIfAbsent(child_id, new mSP1DE(child_id));
                    mSPnDEs.get(child_id).moreChildren(id);

                    // Add node to tbExplored
                    tbExplored.add(m_Successor);
                }
            }
        }
    }

    /**
     *  Create the representation of the graph
     */
    public HashMap<Integer, NodeInt> toSP2DE() {
        HashMap<Integer, NodeInt> nodes = new HashMap<>();

        // Trees to be explored
        LinkedList<Tree> tbExplored = new LinkedList<>();
        tbExplored.add(m_Tree);

        while (!tbExplored.isEmpty()) {
            Tree node = tbExplored.poll();
            int id1 = node.m_Attribute;

            if (!nodes.containsKey(id1)) {
                nodes.put(id1, new NodeInt(id1));
            }

            // Add childs
            for (Tree m_Successor : node.m_Successors) {
                int id2 = m_Successor.m_Attribute;

                // If is not a leaf
                if (id2 != -1) {
                    // Add child
                    if (nodes.get(id2) == null) {
                        nodes.put(id2, new NodeInt(id2, nodes.get(id1)));
                    }
                    nodes.get(id1).addChild(nodes.get(id2));

                    // Add node to tbExplored
                    tbExplored.add(m_Successor);
                }
            }
        }

        return nodes;
    }
}
