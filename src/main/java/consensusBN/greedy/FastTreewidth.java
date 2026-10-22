package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * Array-based version of Utils.getTreeWidth: moralise, order by maximum
 * cardinality search, eliminate in reverse order, and return the size of the
 * largest elimination clique (the max-clique-size convention of TreewidthBudget).
 *
 * The procedure is the same as the Tetrad-based one; only the tie-breaking of
 * the MCS differs (lowest index here, heap order there), so the two can differ
 * by the usual heuristic slack on a few graphs. Both are upper bounds on the
 * true value. The point is speed: no graph objects, no hashing, one pass.
 */
public final class FastTreewidth {

    private FastTreewidth() {}

    /**
     * Also try a min-fill elimination order and keep the smaller width. Both are
     * upper bounds on the true value, so the minimum is still a valid certificate
     * that the DAG fits; min-fill is the tighter one on most graphs and MCS the
     * cheaper one, so the feasibility test runs MCS first. Set with -DminFill=true.
     */
    public static boolean minFill = Boolean.getBoolean("minFill");

    public static int of(Graph dag) {
        return run(dag, Integer.MAX_VALUE, null, false);
    }

    /** Whether some elimination order tried here has no clique larger than k. */
    public static boolean fits(Graph dag, int k) {
        return run(dag, k, null, true) <= k;
    }

    /** Elimination cliques larger than k (as node sets); empty iff the DAG fits k. */
    public static List<Set<Node>> cliquesAbove(Graph dag, int k) {
        List<Set<Node>> out = new ArrayList<>();
        run(dag, k, out, false);
        return out;
    }

    private static int run(Graph dag, int k, List<Set<Node>> collect, boolean stopIfFits) {
        List<Node> nodes = dag.getNodes();
        int n = nodes.size();
        if (n == 0) return 0;
        Map<Node, Integer> id = new HashMap<>(2 * n);
        for (int i = 0; i < n; i++) id.put(nodes.get(i), i);

        int words = (n + 63) >>> 6;
        long[][] adj = new long[n][words];
        for (int v = 0; v < n; v++) {
            List<Node> pa = dag.getParents(nodes.get(v));
            int[] p = new int[pa.size()];
            for (int i = 0; i < p.length; i++) p[i] = id.get(pa.get(i));
            for (int a : p) link(adj, a, v);
            for (int i = 0; i < p.length; i++)
                for (int j = i + 1; j < p.length; j++) link(adj, p[i], p[j]);
        }
        long[][] moral = null;
        if (minFill) {
            moral = new long[n][];
            for (int v = 0; v < n; v++) moral[v] = adj[v].clone();
        }

        // Maximum cardinality search, ties to the lowest index; the elimination
        // order is its reverse.
        int[] order = new int[n], weight = new int[n];
        boolean[] numbered = new boolean[n];
        for (int i = 0; i < n; i++) {
            int best = -1;
            for (int v = 0; v < n; v++)
                if (!numbered[v] && (best < 0 || weight[v] > weight[best])) best = v;
            order[n - 1 - i] = best;
            numbered[best] = true;
            long[] row = adj[best];
            for (int w = 0; w < words; w++) {
                long bits = row[w];
                while (bits != 0) {
                    int u = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    if (!numbered[u]) weight[u]++;
                }
            }
        }
        List<Set<Node>> mcsCliques = collect == null ? null : new ArrayList<>();
        int mcs = eliminate(adj, order, nodes, k, mcsCliques);
        if (!minFill || (mcs <= k && (stopIfFits || collect != null))) {
            if (collect != null) collect.addAll(mcsCliques);
            return mcs;
        }
        List<Set<Node>> fillCliques = collect == null ? null : new ArrayList<>();
        int fill = eliminate(moral, minFillOrder(moral), nodes, k, fillCliques);
        if (collect != null) collect.addAll(fill < mcs ? fillCliques : mcsCliques);
        return Math.min(mcs, fill);
    }

    /**
     * Eliminates in the given order, adding the fill to adj. The clique closed by
     * a vertex is itself plus its neighbours not yet eliminated.
     */
    private static int eliminate(long[][] adj, int[] order, List<Node> nodes, int k, List<Set<Node>> collect) {
        int n = order.length, words = (n + 63) >>> 6;
        int[] pos = new int[n];
        for (int i = 0; i < n; i++) pos[order[i]] = i;
        int maxClique = 1;
        int[] nb = new int[n];
        for (int i = 0; i < n; i++) {
            int v = order[i], c = 0;
            long[] row = adj[v];
            for (int w = 0; w < words; w++) {
                long bits = row[w];
                while (bits != 0) {
                    int u = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    if (pos[u] > i) nb[c++] = u;
                }
            }
            maxClique = Math.max(maxClique, c + 1);
            if (collect != null && c + 1 > k) {
                Set<Node> clique = new HashSet<>();
                clique.add(nodes.get(v));
                for (int a = 0; a < c; a++) clique.add(nodes.get(nb[a]));
                collect.add(clique);
            }
            for (int a = 0; a < c; a++)
                for (int b = a + 1; b < c; b++) link(adj, nb[a], nb[b]);
        }
        return maxClique;
    }

    /**
     * Min-fill elimination order: always the vertex whose elimination adds the
     * fewest edges, ties to the lowest degree and then the lowest index. The fill
     * count of a vertex is recomputed only when its neighbourhood changed.
     */
    private static int[] minFillOrder(long[][] moral) {
        int n = moral.length, words = (n + 63) >>> 6;
        long[][] adj = new long[n][];
        for (int v = 0; v < n; v++) adj[v] = moral[v].clone();
        int[] fill = new int[n], deg = new int[n], order = new int[n], nb = new int[n];
        boolean[] gone = new boolean[n], dirty = new boolean[n];
        Arrays.fill(dirty, true);
        for (int step = 0; step < n; step++) {
            int best = -1;
            for (int v = 0; v < n; v++) {
                if (gone[v]) continue;
                if (dirty[v]) {
                    int d = 0, f = 0;
                    long[] row = adj[v];
                    for (int w = 0; w < words; w++) {
                        long bits = row[w];
                        while (bits != 0) {
                            int u = (w << 6) + Long.numberOfTrailingZeros(bits);
                            bits &= bits - 1;
                            d++;
                            long[] other = adj[u];
                            for (int x = 0; x < words; x++) f += Long.bitCount(row[x] & ~other[x]);
                            f--;                          // u itself is in row and not in other
                        }
                    }
                    deg[v] = d;
                    fill[v] = f / 2;
                    dirty[v] = false;
                }
                if (best < 0 || fill[v] < fill[best] || (fill[v] == fill[best] && deg[v] < deg[best])) best = v;
            }
            order[step] = best;
            gone[best] = true;
            int c = 0;
            long[] row = adj[best];
            for (int w = 0; w < words; w++) {
                long bits = row[w];
                while (bits != 0) {
                    nb[c++] = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                }
            }
            for (int a = 0; a < c; a++) {
                adj[nb[a]][best >>> 6] &= ~(1L << (best & 63));
                dirty[nb[a]] = true;
            }
            for (int a = 0; a < c; a++)
                for (int b = a + 1; b < c; b++) {
                    int x = nb[a], y = nb[b];
                    if ((adj[x][y >>> 6] & (1L << (y & 63))) != 0) continue;
                    // A new edge lowers the fill of the common neighbours of its ends.
                    for (int w = 0; w < words; w++) {
                        long bits = adj[x][w] & adj[y][w];
                        while (bits != 0) {
                            dirty[(w << 6) + Long.numberOfTrailingZeros(bits)] = true;
                            bits &= bits - 1;
                        }
                    }
                    link(adj, x, y);
                }
            Arrays.fill(row, 0L);
        }
        return order;
    }

    private static void link(long[][] adj, int a, int b) {
        adj[a][b >>> 6] |= 1L << (b & 63);
        adj[b][a >>> 6] |= 1L << (a & 63);
    }
}
