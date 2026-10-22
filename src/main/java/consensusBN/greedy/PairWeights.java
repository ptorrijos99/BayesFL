package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.util.*;

/**
 * Per-pair weights for the two data-free consensus objectives.
 *
 * Writing A = E(moral(R)) and B_i = E(moral(G_i)), the identity
 * |A n B| = |A| + |B| - 2|A ^ B| gives
 *
 *     SMHD(R,{G_i}) = (1/r) [ SUM_i |B_i| + SUM_{ab in A} (r - 2 c_ab) ]
 *
 * with c_ab the number of inputs whose moral graph contains {a,b}. The first
 * term is independent of R, so the whole objective is carried by the per-pair
 * cost weights w_ab = r - 2 c_ab, computed once here in O(r n^2). The same
 * identity holds on skeletons, which is the adjacency component of SHD.
 *
 * Weights are in COST form: lower is better, and w_ab < 0 exactly when the pair
 * is supported by a majority of the inputs.
 */
public class PairWeights {

    private final List<Node> nodes;
    private final Map<String, Integer> index = new HashMap<>();
    private final int r;
    private final int[][] cMoral;
    private final int[][] cSkel;
    private final int[][] cDir;
    private final long constMoral;
    private final long constSkel;

    public PairWeights(List<Dag> inputs) {
        this.r = inputs.size();
        this.nodes = inputs.get(0).getNodes();
        int n = nodes.size();
        for (int i = 0; i < n; i++) index.put(nodes.get(i).getName(), i);

        this.cMoral = new int[n][n];
        this.cSkel = new int[n][n];
        this.cDir = new int[n][n];
        long sm = 0, ss = 0;
        for (Dag g : inputs) {
            Graph m = Utils.moralize(g);
            for (Edge e : m.getEdges()) bump(cMoral, e);
            sm += m.getNumEdges();
            for (Edge e : g.getEdges()) {
                bump(cSkel, e);
                cDir[idx(Edges.getDirectedEdgeTail(e))][idx(Edges.getDirectedEdgeHead(e))]++;
            }
            ss += g.getNumEdges();
        }
        this.constMoral = sm;
        this.constSkel = ss;
    }

    private void bump(int[][] c, Edge e) {
        int a = idx(e.getNode1()), b = idx(e.getNode2());
        c[a][b]++;
        c[b][a]++;
    }

    public int idx(Node v) { return index.get(v.getName()); }
    public int n() { return nodes.size(); }
    public int r() { return r; }
    public List<Node> nodes() { return nodes; }

    /** Cost weight of a moral pair. Negative means a majority of inputs support it. */
    public double wMoral(int a, int b) { return r - 2.0 * cMoral[a][b]; }
    public double wSkel(int a, int b) { return r - 2.0 * cSkel[a][b]; }

    public int countMoral(int a, int b) { return cMoral[a][b]; }
    public int countSkel(int a, int b) { return cSkel[a][b]; }

    /** Number of inputs that hold the arc a->b, in that direction. */
    public int countDir(int a, int b) { return cDir[a][b]; }

    /** SUM over the edges of a MORAL graph of w_ab: the R-dependent part of SMHD. */
    public double objectiveOfMoral(Graph moralGraph) {
        double acc = 0;
        for (Edge e : moralGraph.getEdges()) acc += wMoral(idx(e.getNode1()), idx(e.getNode2()));
        return acc;
    }

    /** SUM over the edges of a DAG of the skeleton weights. */
    public double objectiveOfSkeleton(Graph dag) {
        double acc = 0;
        for (Edge e : dag.getEdges()) acc += wSkel(idx(e.getNode1()), idx(e.getNode2()));
        return acc;
    }

    public double smhdOf(double objective) { return (constMoral + objective) / r; }
    public double skelOf(double objective) { return (constSkel + objective) / r; }

    /**
     * Lower bound on the mean SMHD of ANY DAG: relax the moral graph to an
     * arbitrary undirected graph, whose optimum is the majority graph. Strict
     * relaxation, since not every undirected graph is a moral graph (C4 is not:
     * any acyclic orientation has a sink whose two parents are non-adjacent, and
     * moralisation would join them), so the difference is a real optimality gap.
     */
    public double smhdLowerBound() { return (constMoral + majoritySum(cMoral)) / (double) r; }

    /**
     * Minimum mean skeleton distance, and it is ATTAINED: any acyclic orientation
     * of the majority graph is a DAG with that skeleton. See majoritySkeletonDag().
     */
    public double skeletonOptimum() { return (constSkel + majoritySum(cSkel)) / (double) r; }

    private double majoritySum(int[][] c) {
        double acc = 0;
        for (int i = 0; i < c.length; i++)
            for (int j = i + 1; j < c.length; j++)
                if (2 * c[i][j] > r) acc += r - 2.0 * c[i][j];
        return acc;
    }

    /**
     * Pairs adjacent in at least one input: the candidate arc pool. A pair
     * adjacent in no input has c_skel = 0, so w_skel = r > 0, and it can only
     * ever help through co-parent effects. Restricting to this pool is a
     * deliberate design choice and is ablated in the experiments.
     */
    public List<int[]> supportedPairs() {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < cSkel.length; i++)
            for (int j = i + 1; j < cSkel.length; j++)
                if (cSkel[i][j] > 0) out.add(new int[]{i, j});
        return out;
    }

    /**
     * The wider pool: pairs joined in at least one input MORAL graph, which adds
     * the pairs that some input only marries as co-parents. Those pairs are the
     * ones whose moral weight can be negative although no input links them, and
     * an arc between them is the cheapest way to obtain the moral edge.
     */
    public List<int[]> moralPairs() {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < cMoral.length; i++)
            for (int j = i + 1; j < cMoral.length; j++)
                if (cMoral[i][j] > 0) out.add(new int[]{i, j});
        return out;
    }

    /** DAG attaining skeletonOptimum(): majority skeleton oriented along the node order. */
    public Dag majoritySkeletonDag() {
        Dag d = new Dag(nodes);
        for (int i = 0; i < cSkel.length; i++)
            for (int j = i + 1; j < cSkel.length; j++)
                if (2 * cSkel[i][j] > r)
                    d.addEdge(Edges.directedEdge(nodes.get(i), nodes.get(j)));
        return d;
    }
}
