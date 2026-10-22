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

package bayesfl.model;

import bayesfl.privacy.NumericNoiseGenerator;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

/**
 * The sufficient statistics a WDPT tree releases: synthetic-class prior counts
 * and the conditional counts of the NON-parent attributes (parent tables are
 * deterministic given the synthetic class and are reconstructed from the
 * prior). Under secure aggregation each client sends its counts plus a share
 * of the noise, and the server sums them before writing the global tables.
 * <p>
 * The attribute header (no rows) and the parent positions travel with the
 * counts, so whoever holds them can turn them into probability tables.
 * </p>
 */
public final class TableCounts {

    private final Instances header;
    private final Map<Integer, Integer> parentPos;
    private final double[] cls;
    private final double[][][] xy;

    /**
     * @param header    the attribute header of the tree's relabelled data (rows are ignored)
     * @param parentPos relabelled index of each parent attribute to its position in the combination
     * @param cls       synthetic-class counts, {@code cls[c]}
     * @param xy        conditional counts {@code xy[u][v][c]}; parent attributes are never read
     */
    public TableCounts(Instances header, Map<Integer, Integer> parentPos, double[] cls, double[][][] xy) {
        this.header = new Instances(header, 0);
        this.parentPos = parentPos;
        this.cls = cls;
        this.xy = xy;
    }

    public Instances header()                { return header; }
    public Map<Integer, Integer> parentPos() { return parentPos; }
    public double[] cls()                    { return cls; }
    public double[][][] xy()                 { return xy; }

    /**
     * Adds noise to every released count (the prior and the non-parent
     * conditionals), optionally clamping at zero. A share of secure-aggregation
     * noise must not be clamped: that would bias the sum.
     *
     * @param noise the noise generator
     * @param clamp whether to clamp the noisy counts at zero
     * @return new noisy counts
     */
    public TableCounts privatized(NumericNoiseGenerator noise, boolean clamp) {
        double[] c = noise.privatize(cls);
        if (clamp) for (int i = 0; i < c.length; i++) c[i] = Math.max(0.0, c[i]);
        double[][][] x = new double[xy.length][][];
        for (int u = 0; u < xy.length; u++) {
            x[u] = new double[xy[u].length][];
            for (int v = 0; v < xy[u].length; v++) {
                if (parentPos.containsKey(u)) {
                    x[u][v] = xy[u][v].clone();
                    continue;
                }
                x[u][v] = noise.privatize(xy[u][v]);
                if (clamp) for (int k = 0; k < x[u][v].length; k++) x[u][v][k] = Math.max(0.0, x[u][v][k]);
            }
        }
        return new TableCounts(header, parentPos, c, x);
    }

    /**
     * Element-wise sum of the clients' counts for one tree (what secure
     * aggregation reveals to the server).
     *
     * @param parts one entry per client, all for the same tree
     * @return the summed counts
     */
    public static TableCounts sum(List<TableCounts> parts) {
        TableCounts first = parts.get(0);
        double[] c = new double[first.cls.length];
        double[][][] x = new double[first.xy.length][][];
        for (int u = 0; u < x.length; u++) {
            x[u] = new double[first.xy[u].length][first.cls.length];
        }
        for (TableCounts p : parts) {
            for (int i = 0; i < c.length; i++) c[i] += p.cls[i];
            for (int u = 0; u < x.length; u++)
                for (int v = 0; v < x[u].length; v++)
                    for (int k = 0; k < c.length; k++) x[u][v][k] += p.xy[u][v][k];
        }
        return new TableCounts(first.header, first.parentPos, c, x);
    }

    /**
     * Clamps every count at zero in place (post-processing of the released sum).
     *
     * @return this, for chaining
     */
    public TableCounts clampAtZero() {
        for (int i = 0; i < cls.length; i++) cls[i] = Math.max(0.0, cls[i]);
        for (double[][] row : xy)
            for (double[] cell : row)
                for (int k = 0; k < cell.length; k++) cell[k] = Math.max(0.0, cell[k]);
        return this;
    }
}
