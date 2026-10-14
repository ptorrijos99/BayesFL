package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * MMG, the Majority Moral Greedy: a forward greedy that builds the consensus DAG
 * from the empty graph, adding at each step the arc with the most negative exact
 * gain in the decomposable objective, subject to acyclicity and a treewidth
 * budget.
 *
 * It never builds the unrestricted fusion G+. That is the point: G+ is what
 * inflates treewidth and dominates the cost of the pruning methods, and on large
 * networks it is a far worse starting point than the empty graph.
 *
 * Against the incumbent Greedy-Ea this differs in what is scored. Greedy-Ea ranks
 * candidates by marginal ARC frequency; the objective actually decomposes over
 * MORAL PAIRS, so a rare arc that creates heavily-supported co-parent pairs is
 * worth more than a frequent arc that creates none.
 */
public class MajorityGreedy {

    /** Optimise the skeleton objective (adjacency component of SHD) instead of SMHD. */
    public boolean useSkeletonObjective = false;

    /**
     * Re-evaluate only the candidates a move can actually have changed, instead
     * of the whole pool every iteration. Exact, not approximate: it reproduces
     * the full-recomputation search edge for edge. Off by default so the
     * reference path is the plain one.
     */
    public boolean fastCandidateUpdate = false;

    /**
     * Rank candidates by gain per moral edge created, instead of raw gain.
     *
     * The treewidth budget rations moral edges, not objective units, so under a
     * tight budget the right greedy is the knapsack one: a move that buys a lot
     * of objective for one moral edge should outrank a move that buys slightly
     * more for four. This matters most where raw gains tie — with fully agreeing
     * clients every supported pair has the same weight, and ranking by gain alone
     * then degenerates into pool order while still preferring the co-parent-heavy
     * arcs that exhaust the budget fastest.
     */
    public boolean costAwareRanking = false;

    /**
     * Notified after every applied move, so a caller can record the whole
     * trajectory instead of only the endpoint.
     *
     * This is what the AAAI protocol needs: rather than fixing a treewidth budget
     * in advance, run the search all the way and score every point along the way,
     * then ask whether the point chosen by a data-free criterion (distance to the
     * inputs) is also the point that would have been chosen with the gold standard
     * in hand. For a forward search distance-to-inputs falls monotonically, so the
     * data-free choice is always the endpoint — which makes the question sharp:
     * does the endpoint overfit the inputs?
     */
    public interface StepListener { void onStep(int step, Dag current); }

    public StepListener stepListener = null;

    /** Wall-clock seconds of the last search(), excluding the O(r n^2) precomputation. */
    public double executionTime;

    private final PairWeights w;

    public MajorityGreedy(List<Dag> inputs) {
        this.w = new PairWeights(inputs);
    }

    public PairWeights weights() { return w; }

    public Dag search(TreewidthBudget budget) {
        long t0 = System.currentTimeMillis();
        Dag result = useSkeletonObjective ? searchSkeleton(budget) : searchMoral(budget);
        executionTime = (System.currentTimeMillis() - t0) / 1000.0;
        return result;
    }

    /**
     * Under the skeleton objective every pair contributes independently and
     * orientation is irrelevant, so the search reduces to adding majority pairs
     * cheapest-first subject to the budget. With no budget this returns exactly
     * majoritySkeletonDag(), the proven optimum.
     */
    private Dag searchSkeleton(TreewidthBudget budget) {
        Dag r = new Dag(w.nodes());
        List<int[]> pairs = new ArrayList<>();
        for (int[] p : w.supportedPairs()) if (w.wSkel(p[0], p[1]) < 0) pairs.add(p);
        pairs.sort(Comparator.comparingDouble(p -> w.wSkel(p[0], p[1])));

        for (int[] p : pairs) {
            Node a = w.nodes().get(p[0]), b = w.nodes().get(p[1]);
            Node x = a, y = b;
            if (r.paths().existsDirectedPath(y, x)) { x = b; y = a; }
            if (r.paths().existsDirectedPath(y, x)) continue;
            r.addEdge(Edges.directedEdge(x, y));
            if (!budget.fits(r)) r.removeEdge(r.getEdge(x, y));
        }
        return r;
    }

    private Dag searchMoral(TreewidthBudget budget) {
        return fastCandidateUpdate ? searchMoralIncremental(budget) : searchMoralFull(budget);
    }

    /**
     * Same search, re-evaluating only what a move can have changed.
     *
     * Which candidates does adding x->y invalidate? gainAddArc(a,b) reads
     * reasons[a][b] and reasons[a][p] for p in Pa(b). The move changes reasons
     * only for pairs {x,y} and {x,p}, p in Pa_old(y) — every one of them
     * involving x — and changes exactly one parent set, Pa(y). So a candidate can
     * only be affected if one of its endpoints is in
     *
     *     S = {x, y} + Pa_old(y) + Children_new(x)
     *
     * (the last term because gainAddArc(a,b) reads reasons[a][x] whenever x is a
     * parent of b). Recomputing every candidate touching S is therefore a
     * superset of the affected set, which is what makes this exact rather than a
     * heuristic. Candidate order is preserved so ties break identically.
     */
    private Dag searchMoralIncremental(TreewidthBudget budget) {
        Dag r = new Dag(w.nodes());
        MoralGraphTracker t = new MoralGraphTracker(w, r);
        List<int[]> pairs = w.supportedPairs();
        int m = pairs.size();

        List<List<Integer>> byNode = new ArrayList<>(w.n());
        for (int i = 0; i < w.n(); i++) byNode.add(new ArrayList<>());
        for (int p = 0; p < m; p++) {
            byNode.get(pairs.get(p)[0]).add(p);
            byNode.get(pairs.get(p)[1]).add(p);
        }

        double[] gains = new double[2 * m];
        for (int c = 0; c < 2 * m; c++) gains[c] = recompute(c, pairs, r, t);
        int step = 0;

        while (true) {
            List<Integer> improving = new ArrayList<>();
            for (int c = 0; c < 2 * m; c++) if (gains[c] < 0) improving.add(c);
            if (improving.isEmpty()) break;
            improving.sort(Comparator.comparingDouble(c -> gains[c]));   // stable: pool order on ties

            boolean applied = false;
            for (int c : improving) {
                Node x = src(c, pairs), y = dst(c, pairs);
                if (r.paths().existsDirectedPath(y, x)) continue;

                List<Node> paOld = new ArrayList<>(r.getParents(y));
                t.addArc(x, y);
                if (!budget.fits(r)) { t.removeArc(x, y); continue; }

                Set<Integer> dirty = new HashSet<>();
                dirty.add(w.idx(x));
                dirty.add(w.idx(y));
                for (Node p : paOld) dirty.add(w.idx(p));
                for (Node ch : r.getChildren(x)) dirty.add(w.idx(ch));

                Set<Integer> touched = new HashSet<>();
                for (int s : dirty) touched.addAll(byNode.get(s));
                for (int p : touched) {
                    gains[2 * p] = recompute(2 * p, pairs, r, t);
                    gains[2 * p + 1] = recompute(2 * p + 1, pairs, r, t);
                }

                applied = true;
                if (stepListener != null) stepListener.onStep(++step, r);
                break;
            }
            if (!applied) break;
        }
        return r;
    }

    private double recompute(int c, List<int[]> pairs, Dag r, MoralGraphTracker t) {
        Node x = src(c, pairs), y = dst(c, pairs);
        if (r.isAdjacentTo(x, y)) return Double.POSITIVE_INFINITY;
        double g = t.gainAddArc(x, y);
        return g < 0 ? rank(g, t.newMoralEdges(x, y)) : g;
    }

    /** Raw gain, or gain per moral edge created when cost-aware ranking is on. */
    private double rank(double gain, int cost) {
        return costAwareRanking ? gain / Math.max(1, cost) : gain;
    }

    private Node src(int c, List<int[]> pairs) {
        int[] p = pairs.get(c >> 1);
        return w.nodes().get((c & 1) == 0 ? p[0] : p[1]);
    }

    private Node dst(int c, List<int[]> pairs) {
        int[] p = pairs.get(c >> 1);
        return w.nodes().get((c & 1) == 0 ? p[1] : p[0]);
    }

    private Dag searchMoralFull(TreewidthBudget budget) {
        Dag r = new Dag(w.nodes());
        MoralGraphTracker t = new MoralGraphTracker(w, r);
        List<int[]> pairs = w.supportedPairs();
        int stepFull = 0;

        while (true) {
            List<Candidate> cands = new ArrayList<>();
            for (int[] p : pairs) {
                Node a = w.nodes().get(p[0]), b = w.nodes().get(p[1]);
                if (r.isAdjacentTo(a, b)) continue;
                double ga = t.gainAddArc(a, b);
                if (ga < 0) cands.add(new Candidate(a, b, rank(ga, t.newMoralEdges(a, b))));
                double gb = t.gainAddArc(b, a);
                if (gb < 0) cands.add(new Candidate(b, a, rank(gb, t.newMoralEdges(b, a))));
            }
            if (cands.isEmpty()) break;
            cands.sort(Comparator.comparingDouble(c -> c.gain));

            boolean applied = false;
            for (Candidate c : cands) {
                if (r.paths().existsDirectedPath(c.y, c.x)) continue;   // would close a cycle
                t.addArc(c.x, c.y);
                if (!budget.fits(r)) { t.removeArc(c.x, c.y); continue; }
                applied = true;
                if (stepListener != null) stepListener.onStep(++stepFull, r);
                break;
            }
            if (!applied) break;
        }
        return r;
    }

    private static class Candidate {
        final Node x, y;
        final double gain;
        Candidate(Node x, Node y, double gain) { this.x = x; this.y = y; this.gain = gain; }
    }
}
