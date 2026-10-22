package consensusBN.greedy;

import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.utils.Utils;

/**
 * Feasibility test for a treewidth budget.
 *
 * Convention, and it is not the textbook one: Utils.getTreeWidth returns the
 * MAXIMUM CLIQUE SIZE of the moralised-triangulated graph, not that minus one.
 * ConsensusUnion.applyGreedyMaxTreewidth rejects a candidate when some clique
 * has size > maxCliqueSize, so the budget bounds the max clique size. Every
 * class in this package follows that convention, because the existing result
 * files were produced under it.
 *
 * This recomputes the full moralise-triangulate on every call. That is the
 * expected bottleneck of the greedy search; whether an incremental version is
 * worth building is a question for the profile, not for a guess.
 */
public class TreewidthBudget {

    private final int maxCliqueSize;

    /**
     * Use FastTreewidth instead of Utils.getTreeWidth. Off by default so the
     * journal runs keep their convention; the memetic front study turns it on
     * and measures every reported treewidth with the same function.
     */
    public static boolean fast = false;

    public static int treewidth(Graph g) {
        return fast ? FastTreewidth.of(g) : Utils.getTreeWidth(g);
    }

    public TreewidthBudget(int maxCliqueSize) {
        this.maxCliqueSize = maxCliqueSize;
    }

    public static TreewidthBudget unbounded() {
        return new TreewidthBudget(Integer.MAX_VALUE);
    }

    public boolean fits(Graph g) {
        if (maxCliqueSize == Integer.MAX_VALUE) return true;
        return fast ? FastTreewidth.fits(g, maxCliqueSize) : treewidth(g) <= maxCliqueSize;
    }

    public int maxCliqueSize() { return maxCliqueSize; }
}
