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

    public TreewidthBudget(int maxCliqueSize) {
        this.maxCliqueSize = maxCliqueSize;
    }

    public static TreewidthBudget unbounded() {
        return new TreewidthBudget(Integer.MAX_VALUE);
    }

    public boolean fits(Graph g) {
        return maxCliqueSize == Integer.MAX_VALUE || Utils.getTreeWidth(g) <= maxCliqueSize;
    }

    public int maxCliqueSize() { return maxCliqueSize; }
}
