package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.util.*;

/**
 * Chooses an orientation for a fixed skeleton.
 *
 * This is the algorithmic content of the orientation asymmetry: a candidate's
 * skeleton score does not depend on orientation at all, while its moral score
 * does, because every unshielded collider a-&gt;c&lt;-b forces the moral edge
 * {a,b}. So once the skeleton is fixed — and under the skeleton objective the
 * optimal skeleton is known exactly, it is the majority graph — orientation is
 * pure profit: it can only be spent well or badly, never traded against
 * adjacency.
 *
 * The good orientation is the one whose every parent set is already a clique,
 * which adds no moralisation edges at all. Such an orientation exists exactly
 * when the skeleton is chordal, and it is read off a perfect elimination
 * ordering. For non-chordal skeletons there is no zero-cost orientation, but a
 * maximum-cardinality-search ordering is the standard cheap heuristic and
 * degrades gracefully.
 *
 * Compare with orienting along an arbitrary node ordering, which is what
 * PairWeights.majoritySkeletonDag does: that is skeleton-optimal but pays for
 * every collider the ordering happens to create.
 */
public class OrientationChooser {

    /**
     * Orients the skeleton of the given graph along a maximum-cardinality-search
     * ordering: each edge points from the earlier node to the later one, so the
     * parents of a node are its earlier neighbours. On a chordal skeleton those
     * are a clique and moralisation adds nothing.
     */
    public static Dag orientByMCS(Graph skeleton, List<Node> nodes) {
        EdgeListGraph undirected = new EdgeListGraph(nodes);
        for (Edge e : skeleton.getEdges()) {
            Node a = undirected.getNode(e.getNode1().getName());
            Node b = undirected.getNode(e.getNode2().getName());
            if (!undirected.isAdjacentTo(a, b)) undirected.addUndirectedEdge(a, b);
        }

        Node[] order = Utils.getMaximumCardinalityOrdering(undirected);
        Map<String, Integer> rank = new HashMap<>();
        for (int i = 0; i < order.length; i++) rank.put(order[i].getName(), i);

        Dag out = new Dag(nodes);
        for (Edge e : undirected.getEdges()) {
            Node a = out.getNode(e.getNode1().getName());
            Node b = out.getNode(e.getNode2().getName());
            if (rank.get(a.getName()) < rank.get(b.getName())) out.addEdge(Edges.directedEdge(a, b));
            else out.addEdge(Edges.directedEdge(b, a));
        }
        return out;
    }

    /** The majority skeleton oriented by MCS: skeleton-optimal, cheaply moralised. */
    public static Dag majoritySkeletonOriented(PairWeights w) {
        return orientByMCS(w.majoritySkeletonDag(), w.nodes());
    }

    /**
     * Number of moralisation edges the orientation costs: |E(moral(g))| - |E(g)|.
     * Zero means every parent set is already a clique.
     */
    public static int moralisationCost(Dag g) {
        return Utils.moralize(g).getNumEdges() - g.getNumEdges();
    }
}
