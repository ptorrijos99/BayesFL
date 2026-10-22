package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * Total SHD between a CPDAG and a fixed set of input CPDAGs, from per-pair
 * state counts. Utils.SHDundir charges each pair by the states it has in the
 * two graphs only (absent, undirected, a->b, b->a), so counting once how many
 * inputs hold each pair in each state turns the r comparisons of every
 * evaluation into one pass over the edges of the candidate. The total is an
 * integer equal to the sum of Utils.SHDundir over the inputs, so selections
 * made with it are the same as with the direct comparison.
 */
public class CpdagShdCounts {

    // SHDundir cost between two pair states: 0 absent, 1 undirected, 2 a->b, 3 b->a (a < b)
    private static final int[][] COST = {
            {0, 1, 1, 1},
            {1, 0, 2, 2},
            {1, 2, 0, 1},
            {1, 2, 1, 0}};

    private final int r;
    private final Map<String, Integer> index = new HashMap<>();
    // Pairs adjacent in some input: cost of each state minus the cost of leaving it absent
    private final Map<Long, int[]> delta = new HashMap<>();
    // Total when the candidate is empty: every input edge is charged once
    private final int base;

    public CpdagShdCounts(List<Graph> inputs) {
        r = inputs.size();
        for (Node v : inputs.get(0).getNodes()) index.put(v.getName(), index.size());
        Map<Long, int[]> counts = new HashMap<>();
        int edges = 0;
        for (Graph g : inputs) {
            for (Edge e : g.getEdges()) {
                counts.computeIfAbsent(key(e), k -> new int[4])[state(e)]++;
                edges++;
            }
        }
        base = edges;
        for (Map.Entry<Long, int[]> en : counts.entrySet()) {
            int[] c = en.getValue();
            c[0] = r - c[1] - c[2] - c[3];
            int[] d = new int[4];
            for (int s = 1; s < 4; s++)
                for (int t = 0; t < 4; t++) d[s] += c[t] * (COST[s][t] - COST[0][t]);
            delta.put(en.getKey(), d);
        }
    }

    /** Sum over the inputs of Utils.SHDundir(cpdag, input). */
    public int total(Graph cpdag) {
        int sum = base;
        for (Edge e : cpdag.getEdges()) {
            int[] d = delta.get(key(e));
            sum += d == null ? r : d[state(e)];
        }
        return sum;
    }

    /** Mean SHD to the inputs, the same double as summing Utils.SHDundir and dividing by r. */
    public double mean(Graph cpdag) {
        return total(cpdag) / (double) r;
    }

    private long key(Edge e) {
        int a = index.get(e.getNode1().getName()), b = index.get(e.getNode2().getName());
        return (long) Math.min(a, b) * index.size() + Math.max(a, b);
    }

    private int state(Edge e) {
        if (!e.isDirected()) return 1;
        Node head = e.getEndpoint2() == Endpoint.ARROW ? e.getNode2() : e.getNode1();
        Node tail = head == e.getNode2() ? e.getNode1() : e.getNode2();
        return index.get(tail.getName()) < index.get(head.getName()) ? 2 : 3;
    }
}
