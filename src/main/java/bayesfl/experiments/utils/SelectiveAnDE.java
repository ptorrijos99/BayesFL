package bayesfl.experiments.utils;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Selective AnDE baselines of Chen, Martínez, Webb and Wang, for nominal data (n = 1 or 2).
 *
 * <ul>
 *   <li>{@code SASA}: sample-based attribute selective AnDE (IEEE TKDE 2017). The full AnDE count
 *   table is learnt, the attributes are ranked by mutual information with the class, and the
 *   nested model with the first r parents (parent tuples for n = 2) and the first s children is
 *   chosen by the root mean squared error of an incremental leave-one-out over the training data
 *   (the sample is the whole training set, the default of the authors' code).</li>
 *   <li>{@code SA}: memory-constrained selective AnDE (Knowledge and Information Systems 2017). The
 *   attributes are ranked by mutual information and conditional mutual information ("dRank"), the
 *   first s become the super-parent candidates and every attribute is a child. s is set by the
 *   memory rule of the paper (A(n)DE of the memory of A(n-1)DE) or to a given number.</li>
 * </ul>
 *
 * Estimates, eligibility of parents and fall-backs follow the authors' C++ code (Petal):
 * m-estimates with m = 1, a parent value is used when it appears in the training data, and NB or
 * AODE is used when no parent is eligible. Missing values are one more value of the attribute.
 * Products of probabilities are computed in log space so that thousands of children do not
 * underflow; this does not change the model.
 */
public class SelectiveAnDE extends AbstractClassifier {

    private static final long serialVersionUID = 1L;

    /** "SASA" or "SA". */
    private String mode = "SASA";
    /** Number of super-parents per component (1 or 2). */
    private int n = 1;
    /**
     * For SA: "mem" (memory rule of the paper), "fixed" (use {@link #sFixed}) or "mini" (as many
     * components as MiniAnDE with the same n and 100 bagged trees learns on the training data).
     */
    private String sRule = "mem";
    /** For SA with the fixed rule: number of parent attributes (n = 1) or of parent tuples (n = 2). */
    private int sFixed = 1;
    /** For SA: "drank" (the ranking of the paper) or "mi" (mutual information only; validation). */
    private String ranking = "drank";

    // Data
    private int d;              // number of attributes without the class
    private int[] attIdx;       // attribute index in the Instances
    private int c;              // number of classes
    private int[] nv;           // number of values, plus one when the attribute has missing values
    private boolean[] missSlot; // attribute has a missing-value slot
    private int numTrain;

    // Counts
    private int[] classCount;
    private int[][] xy;         // xy[a][v*c + y]
    private int[] off;          // off[a] = sum_{b<a} nv[b]
    private int totalValues;    // sum of nv
    private int[][] xxy;        // xxy[i][(vi*off[i] + off[j] + vj)*c + y], j < i
    private long[] pairStart;   // pairStart[j] = sum_{j'<j} nv[j']*off[j']
    private int[][] xxxy;       // xxxy[i][((vi*pairStart[i] + pairStart[j] + vj*off[j] + off[k] + vk))*c + y], k < j < i

    // SA: counts of every child given each selected parent (n=1) or parent tuple (n=2)
    private int[] saParents;          // selected attributes in rank order
    private int[][] saChild1;         // [p][(vp*totalValues + off[j] + vj)*c + y]
    private int[][] saChild2;         // [pairIndex][((vf*nv[m] + vm)*totalValues + off[j] + vj)*c + y]
    private int[][] saPairs;          // pairs of selected attributes {f, m}

    // SASA
    private int[] order;              // attributes ranked by mutual information
    private int optParents;           // number of parents (n=1) or parent tuples (n=2) selected
    private int optChildren;          // number of children selected

    // Log tables
    private double[] logInt;          // log(k + 1) for k = 0..numTrain+1
    private double[][] lnChild;       // lnChild[a][k] = log(k + 1/nv[a])

    public void setMode(String mode) { this.mode = mode; }
    public void setN(int n) { this.n = n; }
    public void setSRule(String sRule) { this.sRule = sRule; }
    public void setSFixed(int sFixed) { this.sFixed = sFixed; }
    public void setRanking(String ranking) { this.ranking = ranking; }
    public int getOptParents() { return optParents; }
    public int getOptChildren() { return optChildren; }
    public int[] getOrder() { return order; }
    public int[] getSelectedParents() { return saParents; }

    // ------------------------------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------------------------------

    @Override
    public void buildClassifier(Instances data) throws Exception {
        data = new Instances(data);
        data.deleteWithMissingClass();
        int ci = data.classIndex();
        c = data.numClasses();
        d = data.numAttributes() - 1;
        attIdx = new int[d];
        for (int a = 0, k = 0; a < data.numAttributes(); a++) {
            if (a != ci) attIdx[k++] = a;
        }
        numTrain = data.numInstances();
        nv = new int[d];
        missSlot = new boolean[d];
        for (int a = 0; a < d; a++) {
            if (!data.attribute(attIdx[a]).isNominal()) {
                throw new IllegalArgumentException("SelectiveAnDE needs nominal attributes");
            }
            nv[a] = data.attribute(attIdx[a]).numValues();
            for (int i = 0; i < numTrain && !missSlot[a]; i++) {
                if (data.instance(i).isMissing(attIdx[a])) missSlot[a] = true;
            }
            if (missSlot[a]) nv[a]++;
        }
        int[][] x = new int[numTrain][];
        int[] yv = new int[numTrain];
        for (int i = 0; i < numTrain; i++) {
            x[i] = values(data.instance(i));
            yv[i] = (int) data.instance(i).classValue();
        }
        off = new int[d];
        totalValues = 0;
        for (int a = 0; a < d; a++) {
            off[a] = totalValues;
            totalValues += nv[a];
        }
        logInt = new double[numTrain + 3];
        for (int k = 0; k < logInt.length; k++) logInt[k] = Math.log(k + 1.0);
        lnChild = new double[d][];
        java.util.Map<Integer, double[]> byValues = new java.util.HashMap<>();
        for (int a = 0; a < d; a++) {
            lnChild[a] = byValues.computeIfAbsent(nv[a], vals -> {
                double[] tab = new double[numTrain + 3];
                for (int k = 0; k < tab.length; k++) tab[k] = Math.log(k + 1.0 / vals);
                return tab;
            });
        }

        // First pass: class and attribute-class counts
        classCount = new int[c];
        xy = new int[d][];
        for (int a = 0; a < d; a++) xy[a] = new int[nv[a] * c];
        for (int i = 0; i < numTrain; i++) {
            classCount[yv[i]]++;
            for (int a = 0; a < d; a++) xy[a][x[i][a] * c + yv[i]]++;
        }

        if (mode.equals("SA") && sRule.equals("mini")) {
            sFixed = miniAnDEComponents(data);
        }
        if (mode.equals("SASA")) {
            buildSASA(x, yv);
        } else if (mode.equals("SA")) {
            buildSA(x, yv);
        } else {
            throw new IllegalArgumentException("Unknown mode " + mode);
        }
    }

    /** Number of components of MiniAnDE (n, 100 bagged trees) learnt on the same data. */
    private int miniAnDEComponents(Instances data) throws Exception {
        org.albacete.simd.mAnDE.mAnDE m = new org.albacete.simd.mAnDE.mAnDE();
        m.setN(n);
        m.setnTrees(100);
        m.setEnsemble("Bagging");
        m.setBagSize(100);
        m.setAddNB(0.2);
        m.buildClassifier(new Instances(data));
        miniComponents = m.mSPnDEs == null ? 0 : m.mSPnDEs.size();
        return Math.max(miniComponents, 1);
    }

    /** Number of MiniAnDE components that set the number of parents with the "mini" rule. */
    private int miniComponents = -1;
    public int getMiniComponents() { return miniComponents; }

    /** Values of an instance; a missing value maps to its slot, or to -1 when the attribute has none. */
    private int[] values(Instance inst) {
        int[] v = new int[d];
        for (int a = 0; a < d; a++) {
            if (inst.isMissing(attIdx[a])) v[a] = missSlot[a] ? nv[a] - 1 : -1;
            else v[a] = (int) inst.value(attIdx[a]);
        }
        return v;
    }

    /** Mutual information of each attribute with the class, as in Petal (base 2, stored as float). */
    private float[] mutualInformation() {
        float[] mi = new float[d];
        double total = numTrain;
        for (int a = 0; a < d; a++) {
            double m = 0.0;
            for (int v = 0; v < nv[a]; v++) {
                long av = 0;
                for (int y = 0; y < c; y++) av += xy[a][v * c + y];
                for (int y = 0; y < c; y++) {
                    int avy = xy[a][v * c + y];
                    if (avy > 0) {
                        m += (avy / total) * log2(avy / ((av / total) * classCount[y]));
                    }
                }
            }
            mi[a] = (float) m;
        }
        return mi;
    }

    private static double log2(double z) {
        return Math.log(z) / Math.log(2.0);
    }

    /** Attributes sorted by decreasing mutual information; ties keep the attribute order. */
    private int[] rankByMI() {
        float[] mi = mutualInformation();
        Integer[] o = new Integer[d];
        for (int a = 0; a < d; a++) o[a] = a;
        Arrays.sort(o, (p, q) -> Float.compare(mi[q], mi[p]));
        int[] r = new int[d];
        for (int a = 0; a < d; a++) r[a] = o[a];
        return r;
    }

    // ---------------------------------------------------------------- SASA

    private void buildSASA(int[][] x, int[] yv) {
        if (n != 1 && n != 2) throw new IllegalArgumentException("n must be 1 or 2");
        // Full pair (and triple) tables: the memory of A1DE (A2DE)
        xxy = new int[d][];
        for (int i = 1; i < d; i++) xxy[i] = new int[Math.multiplyExact(Math.multiplyExact(nv[i], off[i]), c)];
        if (n == 2) {
            pairStart = new long[d + 1];
            for (int j = 0; j < d; j++) pairStart[j + 1] = pairStart[j] + (long) nv[j] * off[j];
            xxxy = new int[d][];
            for (int i = 2; i < d; i++) {
                xxxy[i] = new int[Math.toIntExact((long) nv[i] * pairStart[i] * c)];
            }
        }
        for (int t = 0; t < numTrain; t++) {
            int[] v = x[t];
            int y = yv[t];
            for (int i = 1; i < d; i++) {
                int base = v[i] * off[i];
                for (int j = 0; j < i; j++) {
                    xxy[i][(base + off[j] + v[j]) * c + y]++;
                }
                if (n == 2) {
                    long base3 = (long) v[i] * pairStart[i];
                    for (int j = 1; j < i; j++) {
                        long bj = base3 + pairStart[j] + (long) v[j] * off[j];
                        for (int k = 0; k < j; k++) {
                            xxxy[i][(int) ((bj + off[k] + v[k]) * c + y)]++;
                        }
                    }
                }
            }
        }
        order = rankByMI();
        if (n == 1) looSelect1(x, yv);
        else looSelect2(x, yv);
    }

    private int cnt2(int a, int va, int b, int vb, int y) {
        if (va < 0 || vb < 0) return 0;
        if (a > b) return xxy[a][(va * off[a] + off[b] + vb) * c + y];
        return xxy[b][(vb * off[b] + off[a] + va) * c + y];
    }

    private int cnt3(int a, int va, int b, int vb, int e, int ve, int y) {
        if (va < 0 || vb < 0 || ve < 0) return 0;
        // sort a > b > e
        if (a < b) { int t = a; a = b; b = t; t = va; va = vb; vb = t; }
        if (b < e) { int t = b; b = e; e = t; t = vb; vb = ve; ve = t; }
        if (a < b) { int t = a; a = b; b = t; t = va; va = vb; vb = t; }
        long idx = ((long) va * pairStart[a] + pairStart[b] + (long) vb * off[b] + off[e] + ve) * c + y;
        return xxxy[a][(int) idx];
    }

    private int cnt1(int a, int va, int y) {
        if (va < 0) return 0;
        return xy[a][va * c + y];
    }

    /** log(k + inv). */
    private static double logNum(int k, double inv) {
        return Math.log(k + inv);
    }

    /** log of the m-estimate (x + 1/nv[a]) / (n + 1) of child a. */
    private double logM(int xk, int nk, int a) {
        return lnChild[a][xk] - logInt[nk];
    }

    /** Use the direct O(d^2) leave-one-out of SASA1DE instead of the compressed one (testing). */
    private boolean referenceLoo = false;
    public void setReferenceLoo(boolean b) { referenceLoo = b; }

    private void looSelect1(int[][] x, int[] yv) {
        if (referenceLoo) looSelect1Reference(x, yv);
        else looSelect1Compressed(x, yv);
    }

    /**
     * The same leave-one-out as {@link #looSelect1Reference}, computed only on the columns whose child
     * is not constant. A child with a single value multiplies every component by exactly 1, so the
     * model with that child added has the same predictions (and errors) as the model before it, and
     * every component whose parent is constant is the same component. After the supervised
     * discretisation of a microarray most genes keep a single value, so this is what makes the
     * O(d^2) model space of the method computable there.
     */
    private void looSelect1Compressed(int[][] x, int[] yv) {
        // Effective columns: 0 is the model with no child yet, e >= 1 the e-th non-constant child
        int[] rep = new int[d];
        List<Integer> effPos = new ArrayList<>();
        effPos.add(-1);
        for (int ci = 0; ci < d; ci++) {
            if (nv[order[ci]] > 1) effPos.add(ci);
            rep[ci] = effPos.size() - 1;
        }
        int E = effPos.size();
        int[] effChild = new int[E];
        for (int e = 1; e < E; e++) effChild[e] = order[effPos.get(e)];
        double[][] sq = new double[d][E];
        double[][] sumLin = new double[E][c];
        double[] ref = new double[E];
        boolean[] started = new boolean[E];
        double[][] L = new double[E][c];        // log terms of the current component per column
        double[][] constL = new double[E][c];   // the component of a constant parent
        int[] pCnt = new int[c];
        for (int t = 0; t < numTrain; t++) {
            int[] v = x[t];
            int ty = yv[t];
            for (int e = 0; e < E; e++) {
                started[e] = false;
                Arrays.fill(sumLin[e], 0.0);
            }
            boolean any = false;
            for (int pi = 0; pi < d && !any; pi++) {
                int p = order[pi];
                int s = 0;
                for (int y = 0; y < c; y++) s += cnt1(p, v[p], y) - (y == ty ? 1 : 0);
                if (s > 0) any = true;
            }
            if (!any) continue;
            boolean constReady = false;
            for (int pi = 0; pi < d; pi++) {
                int p = order[pi];
                int vp = v[p];
                int s = 0;
                for (int y = 0; y < c; y++) {
                    pCnt[y] = cnt1(p, vp, y) - (y == ty ? 1 : 0);
                    s += pCnt[y];
                }
                if (s <= 0) continue;
                double[][] row;
                if (nv[p] == 1 && constReady) {
                    row = constL;
                } else {
                    row = nv[p] == 1 ? constL : L;
                    for (int y = 0; y < c; y++) {
                        row[0][y] = logNum(pCnt[y], 1.0 / (c * nv[p])) - logInt[numTrain - 1];
                    }
                    for (int e = 1; e < E; e++) {
                        int ch = effChild[e];
                        for (int y = 0; y < c; y++) {
                            double z = row[e - 1][y];
                            if (ch != p) {
                                int k = cnt2(ch, v[ch], p, vp, y) - (y == ty ? 1 : 0);
                                z += logM(k, pCnt[y], ch);
                            }
                            row[e][y] = z;
                        }
                    }
                    if (nv[p] == 1) constReady = true;
                }
                double[] sqRow = sq[pi];
                for (int e = 0; e < E; e++) {
                    sqRow[e] += addAndError(sumLin[e], ref, started, e, row[e], ty);
                }
            }
        }
        // Model choice on the effective columns; a run of equal columns starts at its effective column
        int[] firstCol = new int[E];
        for (int e = 0; e < E; e++) firstCol[e] = e == 0 ? 0 : effPos.get(e);
        int bestR = -1, bestS = -1;
        double best = Double.POSITIVE_INFINITY;
        for (int r = 0; r < d; r++) {
            int eMin = -1;
            double vMin = 0;
            for (int e = 0; e < E; e++) {
                // Columns before the first non-constant child exist only when order[0] is constant
                if (e == 0 && nv[order[0]] > 1) continue;
                double val = Math.sqrt(sq[r][e] / numTrain);
                if (eMin < 0 || val < vMin) {
                    eMin = e;
                    vMin = val;
                }
            }
            if (bestR < 0 || best > vMin) {
                bestR = r;
                bestS = firstCol[eMin];
                best = vMin;
            }
        }
        optParents = bestR + 1;
        optChildren = bestS + 1;
    }

    /** Accumulates, for one held-out instance, the squared errors of every nested SASA1DE model. */
    private void looSelect1Reference(int[][] x, int[] yv) {
        double[][] sq = new double[d][d];
        double[] lp = new double[c];
        double[][] sumLin = new double[d][c];   // per child column, sum over parents of exp(L - ref)
        double[] ref = new double[d];
        boolean[] started = new boolean[d];
        int[] pCnt = new int[c];
        for (int t = 0; t < numTrain; t++) {
            int[] v = x[t];
            int ty = yv[t];
            for (int col = 0; col < d; col++) {
                started[col] = false;
                Arrays.fill(sumLin[col], 0.0);
            }
            boolean any = false;
            for (int pi = 0; pi < d && !any; pi++) {
                int p = order[pi];
                int s = 0;
                for (int y = 0; y < c; y++) s += cnt1(p, v[p], y) - (y == ty ? 1 : 0);
                if (s > 0) any = true;
            }
            if (!any) continue;
            for (int pi = 0; pi < d; pi++) {
                int p = order[pi];
                int vp = v[p];
                int s = 0;
                for (int y = 0; y < c; y++) {
                    pCnt[y] = cnt1(p, vp, y) - (y == ty ? 1 : 0);
                    s += pCnt[y];
                }
                if (s <= 0) continue;
                for (int y = 0; y < c; y++) {
                    lp[y] = logNum(pCnt[y], 1.0 / (c * nv[p])) - logInt[numTrain - 1];
                }
                for (int ci = 0; ci < d; ci++) {
                    int ch = order[ci];
                    if (ch != p) {
                        for (int y = 0; y < c; y++) {
                            int k = cnt2(ch, v[ch], p, vp, y) - (y == ty ? 1 : 0);
                            lp[y] += logM(k, pCnt[y], ch);
                        }
                    }
                    sq[pi][ci] += addAndError(sumLin[ci], ref, started, ci, lp, ty);
                }
            }
        }
        chooseModel(sq);
    }

    /**
     * Adds exp(lp) to the running sum of a child column (kept relative to a per-column reference
     * in log space) and returns the squared error of the normalised sum on the true class.
     */
    private double addAndError(double[] lin, double[] ref, boolean[] started, int col, double[] lp, int ty) {
        double mx = Double.NEGATIVE_INFINITY;
        for (int y = 0; y < c; y++) mx = Math.max(mx, lp[y]);
        if (!started[col]) {
            ref[col] = mx;
            started[col] = true;
        } else if (mx > ref[col] + 600) {
            double scale = Math.exp(ref[col] - mx);
            for (int y = 0; y < c; y++) lin[y] *= scale;
            ref[col] = mx;
        }
        double tot = 0;
        for (int y = 0; y < c; y++) {
            lin[y] += Math.exp(lp[y] - ref[col]);
            tot += lin[y];
        }
        double e = 1.0 - lin[ty] / tot;
        return e * e;
    }

    /** Model with the lowest RMSE; the first in row-major order among ties, as in Petal. */
    private void chooseModel(double[][] sq) {
        int rows = sq.length;
        double[][] rmse = new double[rows][d];
        for (int r = 0; r < rows; r++) {
            for (int s = 0; s < d; s++) rmse[r][s] = Math.sqrt(sq[r][s] / numTrain);
        }
        int bestR = 0;
        int bestS = indexOfMin(rmse[0]);
        double best = rmse[0][bestS];
        for (int r = 1; r < rows; r++) {
            int s = indexOfMin(rmse[r]);
            if (best > rmse[r][s]) {
                bestR = r;
                bestS = s;
                best = rmse[r][s];
            }
        }
        optParents = bestR + 1;
        optChildren = bestS + 1;
    }

    private static int indexOfMin(double[] v) {
        int mi = 0;
        for (int i = 1; i < v.length; i++) if (v[i] < v[mi]) mi = i;
        return mi;
    }

    /** Leave-one-out over the nested SASA2DE models (parent tuples in rank order). */
    private void looSelect2(int[][] x, int[] yv) {
        int pairs = d * (d - 1) / 2;
        double[][] sq = new double[pairs][d];
        double[] lp = new double[c];
        double[][] sumLin = new double[d][c];
        double[] ref = new double[d];
        boolean[] started = new boolean[d];
        int[] pCnt = new int[c];
        for (int t = 0; t < numTrain; t++) {
            int[] v = x[t];
            int ty = yv[t];
            for (int col = 0; col < d; col++) {
                started[col] = false;
                Arrays.fill(sumLin[col], 0.0);
            }
            for (int fi = 1; fi < d; fi++) {
                int f = order[fi];
                for (int mi = 0; mi < fi; mi++) {
                    int m = order[mi];
                    int pIdx = fi * (fi - 1) / 2 + mi;
                    int s = 0;
                    for (int y = 0; y < c; y++) {
                        pCnt[y] = cnt2(f, v[f], m, v[m], y) - (y == ty ? 1 : 0);
                        s += pCnt[y];
                    }
                    if (s <= 0) continue;
                    for (int y = 0; y < c; y++) {
                        lp[y] = logNum(pCnt[y], 1.0 / ((double) c * nv[f] * nv[m])) - logInt[numTrain - 1];
                    }
                    for (int ci = 0; ci < d; ci++) {
                        int ch = order[ci];
                        if (ch != f && ch != m) {
                            for (int y = 0; y < c; y++) {
                                int k = cnt3(ch, v[ch], f, v[f], m, v[m], y) - (y == ty ? 1 : 0);
                                lp[y] += logM(k, pCnt[y], ch);
                            }
                        }
                        sq[pIdx][ci] += addAndError(sumLin[ci], ref, started, ci, lp, ty);
                    }
                }
            }
        }
        chooseModel(sq);
    }

    // ---------------------------------------------------------------- SA

    private void buildSA(int[][] x, int[] yv) {
        if (n != 1 && n != 2) throw new IllegalArgumentException("n must be 1 or 2");
        int s = chooseNumberAndRank(x, yv);
        int[] sel = Arrays.copyOf(saParents, s);
        saParents = sel;
        if (n == 1) {
            saChild1 = new int[s][];
            for (int p = 0; p < s; p++) saChild1[p] = new int[Math.multiplyExact(Math.multiplyExact(nv[sel[p]], totalValues), c)];
            for (int t = 0; t < numTrain; t++) {
                int[] v = x[t];
                int y = yv[t];
                for (int p = 0; p < s; p++) {
                    int a = sel[p];
                    int base = v[a] * totalValues;
                    int[] tab = saChild1[p];
                    for (int j = 0; j < d; j++) tab[(base + off[j] + v[j]) * c + y]++;
                }
            }
        } else {
            List<int[]> pairs = new ArrayList<>();
            for (int fi = 1; fi < s; fi++) {
                for (int mi = 0; mi < fi; mi++) pairs.add(new int[] {sel[fi], sel[mi]});
            }
            saPairs = pairs.toArray(new int[0][]);
            saChild2 = new int[saPairs.length][];
            for (int p = 0; p < saPairs.length; p++) {
                int f = saPairs[p][0], m = saPairs[p][1];
                saChild2[p] = new int[Math.toIntExact((long) nv[f] * nv[m] * totalValues * c)];
            }
            for (int t = 0; t < numTrain; t++) {
                int[] v = x[t];
                int y = yv[t];
                for (int p = 0; p < saPairs.length; p++) {
                    int f = saPairs[p][0], m = saPairs[p][1];
                    int base = (v[f] * nv[m] + v[m]) * totalValues;
                    int[] tab = saChild2[p];
                    for (int j = 0; j < d; j++) tab[(base + off[j] + v[j]) * c + y]++;
                }
            }
        }
    }

    /**
     * Ranks the attributes with dRank (first the attribute with the highest mutual information,
     * then the one that maximises the minimum conditional mutual information with the class given
     * each attribute already ranked) up to the number of parent attributes needed, and returns it.
     * The conditional mutual informations are computed only for the attributes already ranked.
     */
    private int chooseNumberAndRank(int[][] x, int[] yv) {
        if (ranking.equals("mi")) {
            // Validation only: Petal's "+selective +mi", the first sFixed attributes by mutual information
            saParents = rankByMI();
            return Math.min(sFixed, d);
        }
        float[] mi = mutualInformation();
        double[] minCmi = new double[d];
        Arrays.fill(minCmi, Double.POSITIVE_INFINITY);
        boolean[] used = new boolean[d];
        List<Integer> ranked = new ArrayList<>();
        int first = 0;
        for (int a = 1; a < d; a++) if (mi[a] > mi[first]) first = a;

        // Memory rule (Eq. 8 of the paper): C(s,n+1) + C(s,n)(d-s), times u^{n+1}, equals the
        // memory C(d,n) v^n of A(n-1)DE, u and v the mean number of values of the selected and of
        // all the attributes. The largest s whose memory does not exceed it is used.
        double vAll = 0;
        for (int a = 0; a < d; a++) vAll += nv[a];
        vAll /= d;
        double rhs = binom(d, n) * Math.pow(vAll, n);
        boolean fixedCount = sRule.equals("fixed") || sRule.equals("mini");
        int target = fixedCount ? Math.min(sFixed, d) : d;
        if (fixedCount && n == 2) {
            // sFixed parent tuples: the largest s with C(s,2) <= sFixed
            int s = 2;
            while (s < d && binom(s + 1, 2) <= sFixed) s++;
            target = Math.min(s, d);
        }

        int current = first;
        double sumVals = 0;
        int lastValid = n == 2 ? 2 : 1;
        while (true) {
            used[current] = true;
            ranked.add(current);
            sumVals += nv[current];
            int s = ranked.size();
            if (sRule.equals("mem")) {
                double u = sumVals / s;
                double lhs = (binom(s, n + 1) + binom(s, n) * (d - s)) * Math.pow(u, n + 1);
                if (lhs <= rhs) lastValid = s;
                else if (s > (n == 2 ? 2 : 1)) break;
            } else if (s >= target) {
                lastValid = s;
                break;
            }
            if (s == d) {
                if (!sRule.equals("mem")) lastValid = s;
                break;
            }
            // Update the minimum CMI of the remaining attributes with the newly ranked one
            updateMinCmi(x, yv, current, used, minCmi);
            int next = -1;
            for (int a = 0; a < d; a++) {
                if (!used[a] && (next < 0 || minCmi[a] > minCmi[next])) next = a;
            }
            current = next;
        }
        saParents = new int[ranked.size()];
        for (int i = 0; i < ranked.size(); i++) saParents[i] = ranked.get(i);
        order = saParents;
        return Math.min(Math.max(lastValid, n == 2 ? Math.min(2, d) : 1), ranked.size());
    }

    private static double binom(int a, int b) {
        if (b < 0 || b > a) return 0;
        double r = 1;
        for (int i = 0; i < b; i++) r = r * (a - i) / (i + 1);
        return r;
    }

    /** I(X;Y|Z) for every attribute X not yet ranked, maximum-likelihood estimate in bits. */
    private void updateMinCmi(int[][] x, int[] yv, int z, boolean[] used, double[] minCmi) {
        int nz = nv[z];
        double total = numTrain;
        int[] zy = new int[nz * c];
        int[] zc = new int[nz];
        for (int t = 0; t < numTrain; t++) {
            zy[x[t][z] * c + yv[t]]++;
            zc[x[t][z]]++;
        }
        for (int a = 0; a < d; a++) {
            if (used[a]) continue;
            int na = nv[a];
            int[] xzy = new int[na * nz * c];
            int[] xz = new int[na * nz];
            for (int t = 0; t < numTrain; t++) {
                int k = x[t][a] * nz + x[t][z];
                xzy[k * c + yv[t]]++;
                xz[k]++;
            }
            double cmi = 0;
            for (int va = 0; va < na; va++) {
                for (int vz = 0; vz < nz; vz++) {
                    int k = va * nz + vz;
                    if (xz[k] == 0) continue;
                    for (int y = 0; y < c; y++) {
                        int n3 = xzy[k * c + y];
                        if (n3 == 0) continue;
                        // p(x,y,z) log [ p(x,y,z) p(z) / (p(x,z) p(y,z)) ]
                        cmi += (n3 / total) * log2(((double) n3 * zc[vz]) / ((double) xz[k] * zy[vz * c + y]));
                    }
                }
            }
            if (cmi < minCmi[a]) minCmi[a] = cmi;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Prediction
    // ------------------------------------------------------------------------------------------

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        int[] v = values(inst);
        double[] lp;
        if (mode.equals("SASA")) lp = n == 1 ? sasa1(v) : sasa2(v);
        else lp = n == 1 ? sa1(v) : sa2(v);
        return normaliseLog(lp);
    }

    private static double[] normaliseLog(double[] lp) {
        double mx = Double.NEGATIVE_INFINITY;
        for (double z : lp) mx = Math.max(mx, z);
        double[] p = new double[lp.length];
        double tot = 0;
        for (int i = 0; i < lp.length; i++) {
            p[i] = Math.exp(lp[i] - mx);
            tot += p[i];
        }
        for (int i = 0; i < p.length; i++) p[i] /= tot;
        return p;
    }

    /** Running log-sum-exp of one component into acc. */
    private static void logAdd(double[] acc, double[] lp) {
        for (int y = 0; y < acc.length; y++) {
            double a = acc[y], b = lp[y];
            if (a == Double.NEGATIVE_INFINITY) acc[y] = b;
            else if (a > b) acc[y] = a + Math.log1p(Math.exp(b - a));
            else acc[y] = b + Math.log1p(Math.exp(a - b));
        }
    }

    private double[] negInf() {
        double[] r = new double[c];
        Arrays.fill(r, Double.NEGATIVE_INFINITY);
        return r;
    }

    private double logPrior(int y) {
        return Math.log((classCount[y] + 1.0 / c) / (numTrain + 1.0));
    }

    /** Naive Bayes with m-estimates. */
    private double[] nb(int[] v) {
        double[] lp = new double[c];
        for (int y = 0; y < c; y++) {
            lp[y] = logPrior(y);
            for (int a = 0; a < d; a++) lp[y] += logM(cnt1(a, v[a], y), classCount[y], a);
        }
        return lp;
    }

    private boolean eligible(int a, int va) {
        if (va < 0) return false;
        int s = 0;
        for (int y = 0; y < c; y++) s += xy[a][va * c + y];
        return s > 0;
    }

    private double[] sasa1(int[] v) {
        double[] acc = negInf();
        boolean any = false;
        double[] lp = new double[c];
        for (int pi = 0; pi < optParents; pi++) {
            int p = order[pi];
            if (!eligible(p, v[p])) continue;
            any = true;
            for (int y = 0; y < c; y++) {
                int py = cnt1(p, v[p], y);
                lp[y] = logNum(py, 1.0 / (c * nv[p])) - logInt[numTrain];
                for (int ci = 0; ci < optChildren; ci++) {
                    int ch = order[ci];
                    if (ch != p) lp[y] += logM(cnt2(ch, v[ch], p, v[p], y), py, ch);
                }
            }
            logAdd(acc, lp);
        }
        if (!any) return nb(v);
        return acc;
    }

    /** AODE over all attributes as parents (the fall-back of SASA2DE), NB when none is eligible. */
    private double[] aode(int[] v, int[] parents) {
        double[] acc = negInf();
        boolean any = false;
        double[] lp = new double[c];
        for (int p : parents) {
            if (!eligible(p, v[p])) continue;
            any = true;
            for (int y = 0; y < c; y++) {
                int py = cnt1(p, v[p], y);
                lp[y] = logNum(py, 1.0 / (c * nv[p])) - logInt[numTrain];
                for (int ch = 0; ch < d; ch++) {
                    if (ch != p) lp[y] += logM(childCount1(ch, v[ch], p, v[p], y), py, ch);
                }
            }
            logAdd(acc, lp);
        }
        if (!any) return nb(v);
        return acc;
    }

    /** Count of (child, parent, y), from the full tables (SASA) or the SA tables. */
    private int childCount1(int ch, int vch, int p, int vp, int y) {
        if (xxy != null) return cnt2(ch, vch, p, vp, y);
        if (vch < 0 || vp < 0) return 0;
        int idx = indexOf(saParents, p);
        return saChild1[idx][(vp * totalValues + off[ch] + vch) * c + y];
    }

    private static int indexOf(int[] a, int z) {
        for (int i = 0; i < a.length; i++) if (a[i] == z) return i;
        return -1;
    }

    private double[] sasa2(int[] v) {
        // Fall-back when no parent tuple at all is eligible
        boolean anyPair = false;
        for (int fi = 1; fi < d && !anyPair; fi++) {
            for (int mi = 0; mi < fi && !anyPair; mi++) {
                int f = order[fi], m = order[mi];
                int s = 0;
                for (int y = 0; y < c; y++) s += cnt2(f, v[f], m, v[m], y);
                if (s > 0) anyPair = true;
            }
        }
        if (!anyPair) {
            int[] all = new int[d];
            for (int a = 0; a < d; a++) all[a] = a;
            return aode(v, all);
        }
        double[] acc = negInf();
        boolean has = false;
        double[] lp = new double[c];
        outer:
        for (int fi = 1; fi < d; fi++) {
            int f = order[fi];
            for (int mi = 0; mi < fi; mi++) {
                int pIdx = fi * (fi - 1) / 2 + mi;
                if (pIdx >= optParents) break outer;
                int m = order[mi];
                int s = 0;
                for (int y = 0; y < c; y++) s += cnt2(f, v[f], m, v[m], y);
                if (s <= 0) continue;
                has = true;
                for (int y = 0; y < c; y++) {
                    int py = cnt2(f, v[f], m, v[m], y);
                    lp[y] = logNum(py, 1.0 / ((double) c * nv[f] * nv[m])) - logInt[numTrain];
                    for (int ci = 0; ci < optChildren; ci++) {
                        int ch = order[ci];
                        if (ch != f && ch != m) lp[y] += logM(cnt3(ch, v[ch], f, v[f], m, v[m], y), py, ch);
                    }
                }
                logAdd(acc, lp);
            }
        }
        if (!has) {
            double[] r = new double[c];
            for (int y = 0; y < c; y++) r[y] = Math.log((classCount[y] + 1.0 / c) / (numTrain + 1.0));
            return r;
        }
        return acc;
    }

    private double[] sa1(int[] v) {
        double[] acc = negInf();
        boolean any = false;
        double[] lp = new double[c];
        for (int pi = 0; pi < saParents.length; pi++) {
            int p = saParents[pi];
            if (!eligible(p, v[p])) continue;
            any = true;
            int vp = v[p];
            int[] tab = saChild1[pi];
            for (int y = 0; y < c; y++) {
                int py = cnt1(p, vp, y);
                lp[y] = logNum(py, 1.0 / (c * nv[p])) - logInt[numTrain];
                for (int ch = 0; ch < d; ch++) {
                    if (ch == p) continue;
                    int k = v[ch] < 0 ? 0 : tab[(vp * totalValues + off[ch] + v[ch]) * c + y];
                    lp[y] += logM(k, py, ch);
                }
            }
            logAdd(acc, lp);
        }
        if (!any) return nb(v);
        return acc;
    }

    private double[] sa2(int[] v) {
        double[] acc = negInf();
        boolean any = false;
        double[] lp = new double[c];
        for (int p = 0; p < saPairs.length; p++) {
            int f = saPairs[p][0], m = saPairs[p][1];
            if (v[f] < 0 || v[m] < 0) continue;
            int[] tab = saChild2[p];
            int base = (v[f] * nv[m] + v[m]) * totalValues;
            // Count of the parent tuple per class: the count of the parent value of f as a child
            int s = 0;
            int[] py = new int[c];
            for (int y = 0; y < c; y++) {
                py[y] = tab[(base + off[f] + v[f]) * c + y];
                s += py[y];
            }
            if (s <= 0) continue;
            any = true;
            for (int y = 0; y < c; y++) {
                lp[y] = logNum(py[y], 1.0 / ((double) c * nv[f] * nv[m])) - logInt[numTrain];
                for (int ch = 0; ch < d; ch++) {
                    if (ch == f || ch == m) continue;
                    int k = v[ch] < 0 ? 0 : tab[(base + off[ch] + v[ch]) * c + y];
                    lp[y] += logM(k, py[y], ch);
                }
            }
            logAdd(acc, lp);
        }
        if (!any) {
            // AODE with the selected attributes as parents, as Petal's A2DE falls back to
            return aodeSA(v);
        }
        return acc;
    }

    /** AODE over the selected attributes for SA2DE, with counts from the pair tables. */
    private double[] aodeSA(int[] v) {
        double[] acc = negInf();
        boolean any = false;
        double[] lp = new double[c];
        for (int p : saParents) {
            if (!eligible(p, v[p])) continue;
            any = true;
            // Pair counts (p, child, y): marginalise a pair table that contains p
            for (int y = 0; y < c; y++) {
                int py = cnt1(p, v[p], y);
                lp[y] = logNum(py, 1.0 / (c * nv[p])) - logInt[numTrain];
                for (int ch = 0; ch < d; ch++) {
                    if (ch != p) lp[y] += logM(pairCountSA(p, v[p], ch, v[ch], y), py, ch);
                }
            }
            logAdd(acc, lp);
        }
        if (!any) return nb(v);
        return acc;
    }

    /** Count of (p, child, y) from an SA2DE pair table that contains p. */
    private int pairCountSA(int p, int vp, int ch, int vch, int y) {
        if (vp < 0 || vch < 0) return 0;
        for (int q = 0; q < saPairs.length; q++) {
            int f = saPairs[q][0], m = saPairs[q][1];
            if (f != p && m != p) continue;
            int other = f == p ? m : f;
            int sum = 0;
            for (int vo = 0; vo < nv[other]; vo++) {
                int vf = f == p ? vp : vo;
                int vm = m == p ? vp : vo;
                sum += saChild2[q][((vf * nv[m] + vm) * totalValues + off[ch] + vch) * c + y];
            }
            return sum;
        }
        return 0;
    }
}
