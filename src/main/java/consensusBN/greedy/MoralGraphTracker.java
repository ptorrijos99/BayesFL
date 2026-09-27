package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * The moral graph of a candidate DAG, maintained incrementally.
 *
 * A pair {a,b} is a moral edge for one or more independent REASONS: the arc
 * between a and b itself, plus one reason per common child z with a,b in Pa(z).
 * The objective sums w_ab over pairs with at least one reason, so a correct
 * incremental gain has to count reasons rather than presence. Removing an arc
 * whose endpoints remain co-parents of another child does NOT remove the moral
 * edge, and an implementation that assumes it does is wrong in a way that no
 * insertion-only test will catch.
 *
 * Adding x->y touches only the pair {x,y} and the co-parent pairs {x,p} for
 * p in Pa(y), because moralisation edges depend on parent sets and only Pa(y)
 * changed. That is what makes evaluation local and input-free.
 */
public class MoralGraphTracker {

    private final PairWeights w;
    private final Dag dag;
    private final int[][] reasons;
    private double objective;

    public MoralGraphTracker(PairWeights w, Dag initial) {
        this.w = w;
        this.dag = initial;
        this.reasons = new int[w.n()][w.n()];
        this.objective = 0;

        for (Edge e : initial.getEdges()) addReason(w.idx(e.getNode1()), w.idx(e.getNode2()));
        for (Node z : initial.getNodes()) {
            List<Node> pa = initial.getParents(z);
            for (int i = 0; i < pa.size(); i++)
                for (int j = i + 1; j < pa.size(); j++)
                    addReason(w.idx(pa.get(i)), w.idx(pa.get(j)));
        }
    }

    private void addReason(int a, int b) {
        if (reasons[a][b] == 0) objective += w.wMoral(a, b);
        reasons[a][b]++;
        reasons[b][a]++;
    }

    private void removeReason(int a, int b) {
        reasons[a][b]--;
        reasons[b][a]--;
        if (reasons[a][b] == 0) objective -= w.wMoral(a, b);
    }

    public double objective() { return objective; }
    public double smhd() { return w.smhdOf(objective); }
    public Dag dag() { return dag; }

    /** Change in the objective from adding x->y. Negative is an improvement. */
    public double gainAddArc(Node x, Node y) {
        int xi = w.idx(x);
        double delta = 0;
        if (reasons[xi][w.idx(y)] == 0) delta += w.wMoral(xi, w.idx(y));
        for (Node p : dag.getParents(y)) {
            if (p.equals(x)) continue;
            int pi = w.idx(p);
            if (reasons[xi][pi] == 0) delta += w.wMoral(xi, pi);
        }
        return delta;
    }

    /**
     * How many moral edges adding x->y would create. This is the move's cost in
     * the resource the treewidth budget actually rations, which is not the same
     * quantity as its gain: an arc can be strongly supported and still be
     * expensive because it joins x to a large parent set.
     */
    public int newMoralEdges(Node x, Node y) {
        int xi = w.idx(x);
        int count = 0;
        if (reasons[xi][w.idx(y)] == 0) count++;
        for (Node p : dag.getParents(y)) {
            if (p.equals(x)) continue;
            if (reasons[xi][w.idx(p)] == 0) count++;
        }
        return count;
    }

    public void addArc(Node x, Node y) {
        int xi = w.idx(x);
        List<Node> pa = new ArrayList<>(dag.getParents(y));   // snapshot BEFORE the edge exists
        dag.addEdge(Edges.directedEdge(x, y));
        addReason(xi, w.idx(y));
        for (Node p : pa) if (!p.equals(x)) addReason(xi, w.idx(p));
    }

    /** Change in the objective from removing x->y. Negative is an improvement. */
    public double gainRemoveArc(Node x, Node y) {
        int xi = w.idx(x), yi = w.idx(y);
        double delta = 0;
        if (reasons[xi][yi] == 1) delta -= w.wMoral(xi, yi);
        for (Node p : dag.getParents(y)) {
            if (p.equals(x)) continue;
            int pi = w.idx(p);
            if (reasons[xi][pi] == 1) delta -= w.wMoral(xi, pi);
        }
        return delta;
    }

    public void removeArc(Node x, Node y) {
        int xi = w.idx(x);
        List<Node> pa = new ArrayList<>(dag.getParents(y));   // snapshot BEFORE removal
        dag.removeEdge(dag.getEdge(x, y));
        removeReason(xi, w.idx(y));
        for (Node p : pa) if (!p.equals(x)) removeReason(xi, w.idx(p));
    }
}
