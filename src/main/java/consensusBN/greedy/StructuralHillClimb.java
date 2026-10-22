package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * Full local search on the decomposable consensus objective: add, remove and
 * reverse arcs, from any starting DAG, under a treewidth budget.
 *
 * This exists because forward-only search demonstrably stalls. On alarm with 10
 * federated clients at budget 5, MMG stops at SMHD 12.7 while MCBNC, which
 * prunes down from G+, reaches 5.5. The forward greedy is not wrong about the
 * objective, it is simply unable to undo an early commitment: an arc that paid
 * for itself when it was added can become dead weight once later arcs supply the
 * same moral edges. Removal and reversal are what let it recover.
 *
 * Every gain is exact and local, from MoralGraphTracker. Reversal is evaluated
 * compositionally: apply the removal, evaluate the insertion in the resulting
 * graph, then undo. That is exact because the tracker counts reasons, so an
 * undo restores the objective bit for bit.
 *
 * Note this is the rescued "super-greedy" of the earlier work, but scored on the
 * decomposable objective rather than by recomputing SMHD against every input at
 * every candidate, which is what made the original too slow to use standalone.
 */
public class StructuralHillClimb {

    /** Wall-clock seconds of the last search(). */
    public double executionTime;

    /** Number of moves applied by the last search(), by kind. */
    public int addsApplied, removesApplied, reversalsApplied;

    /**
     * Weight of the skeleton objective added to the moral one. The moral
     * objective cannot see an arc between two nodes that are already co-parents:
     * the pair is a moral edge either way, so adding or deleting that arc has
     * zero moral gain. Descending from G+ therefore keeps every such arc, which
     * costs nothing on SMHD and a great deal on SHD. Both objectives decompose
     * over pairs, so any positive combination keeps every gain exact and local.
     * Moral gains are integers, so any weight below 1/r makes the skeleton a
     * pure tie-breaker (lexicographic); 0 reproduces the original search.
     */
    public double skelWeight = 0.0;

    /**
     * Orders moves of equal gain by the inputs. The two objectives depend on the
     * orientation of an arc only through the pairs it marries, so when a pair is
     * first joined both orientations have the same gain, and a wrong choice can
     * leave a v-structure of the inputs turned into a fork that no single
     * improving move repairs. Among moves of equal gain, the one whose arc
     * direction is held by more inputs goes first. It never changes which moves
     * improve the objective; false reproduces the original order.
     */
    public boolean dirTieBreak = true;

    /** Inputs holding x->y minus inputs holding y->x. */
    private int dirMargin(Node x, Node y) {
        int a = w.idx(x), b = w.idx(y);
        return w.countDir(a, b) - w.countDir(b, a);
    }

    /**
     * Candidate insertions over the pairs of the input moral graphs (c^M >= 1)
     * instead of the pairs adjacent in some input (c^S >= 1). False reproduces
     * the original pool.
     */
    public boolean poolMoral = false;

    /** Only reversals: the skeleton is kept and only the orientation is searched. */
    public boolean fixedSkeleton = false;

    /**
     * Optional immorality term. With it, the search can target the distance
     * between equivalence classes (skeleton + immoralities), which is what the
     * CPDAG-SHD rewards and what the moral objective only sees through married
     * pairs. Null or weight 0 reproduces the original search.
     */
    public ImmoralityWeights imm = null;
    public double immWeight = 0.0;

    /** Weight of the moral term; 0 turns the search into a pure class-distance search. */
    public double moralWeight = 1.0;

    /** True if a directed path from x to y exists that does not use the arc x->y itself. */
    private static boolean otherDirectedPath(Dag g, Node x, Node y) {
        Deque<Node> stack = new ArrayDeque<>();
        Set<Node> seen = new HashSet<>();
        for (Node c : g.getChildren(x)) if (!c.equals(y)) { stack.push(c); seen.add(c); }
        while (!stack.isEmpty()) {
            Node v = stack.pop();
            if (v.equals(y)) return true;
            for (Node c : g.getChildren(v)) if (seen.add(c)) stack.push(c);
        }
        return false;
    }

    private double immAdd(Dag g, Node x, Node y) {
        return imm == null || immWeight == 0 ? 0 : immWeight * imm.gainAdd(g, x, y);
    }

    private double immRemove(Dag g, Node x, Node y) {
        return imm == null || immWeight == 0 ? 0 : immWeight * imm.gainRemove(g, x, y);
    }

    private final PairWeights w;

    public StructuralHillClimb(List<Dag> inputs) {
        this.w = new PairWeights(inputs);
    }

    public StructuralHillClimb(PairWeights weights) {
        this.w = weights;
    }

    public PairWeights weights() { return w; }

    /** Starts from the empty graph. */
    public Dag search(TreewidthBudget budget) {
        return search(new Dag(w.nodes()), budget);
    }

    /**
     * Hill-climbs from the given DAG, which is mutated in place and returned.
     * The caller is responsible for the initial graph respecting the budget; the
     * search preserves feasibility but does not repair it.
     */
    public Dag search(Dag initial, TreewidthBudget budget) {
        long t0 = System.currentTimeMillis();
        addsApplied = removesApplied = reversalsApplied = 0;

        Dag r = initial;
        MoralGraphTracker t = new MoralGraphTracker(w, r);
        List<int[]> pairs = poolMoral ? w.moralPairs() : w.supportedPairs();

        while (true) {
            List<Move> moves = new ArrayList<>();

            // Additions: any supported pair not currently adjacent, both orientations.
            for (int[] p : pairs) {
                Node a = w.nodes().get(p[0]), b = w.nodes().get(p[1]);
                if (fixedSkeleton || r.isAdjacentTo(a, b)) continue;
                double gs = skelWeight * w.wSkel(p[0], p[1]);
                double ga = moralWeight * t.gainAddArc(a, b) + gs + immAdd(r, a, b);
                if (ga < 0) moves.add(new Move(Kind.ADD, a, b, ga, -dirMargin(a, b)));
                double gb = moralWeight * t.gainAddArc(b, a) + gs + immAdd(r, b, a);
                if (gb < 0) moves.add(new Move(Kind.ADD, b, a, gb, -dirMargin(b, a)));
            }

            // Removals and reversals over the current arcs.
            for (Edge e : new ArrayList<>(r.getEdges())) {
                Node x = e.getNode1(), y = e.getNode2();
                if (!r.isParentOf(x, y)) { Node tmp = x; x = y; y = tmp; }

                double gr = moralWeight * t.gainRemoveArc(x, y) + immRemove(r, x, y);
                double grTotal = gr - skelWeight * w.wSkel(w.idx(x), w.idx(y));
                if (!fixedSkeleton && grTotal < 0) moves.add(new Move(Kind.REMOVE, x, y, grTotal, 0));

                // Reversal = removal then insertion the other way, evaluated exactly
                // by applying the removal and undoing it.
                // The acyclicity test is the expensive part (a graph traversal per
                // arc per iteration), so it is only paid for improving reversals.
                double gRev;
                if (imm == null || immWeight == 0) {
                    gRev = moralWeight * t.gainReverseArc(x, y);
                } else {
                    t.removeArc(x, y);
                    gRev = gr + moralWeight * t.gainAddArc(y, x) + immAdd(r, y, x);
                    t.addArc(x, y);
                }
                // Reversing x->y creates a cycle iff another directed path x ~> y exists.
                if (gRev < 0 && !otherDirectedPath(r, x, y)) moves.add(new Move(Kind.REVERSE, x, y, gRev, -dirMargin(y, x)));
            }

            if (moves.isEmpty()) break;
            if (dirTieBreak) {
                moves.sort((p, q) -> Math.abs(p.gain - q.gain) > 1e-9
                        ? Double.compare(p.gain, q.gain) : Integer.compare(p.tie, q.tie));
            } else {
                moves.sort(Comparator.comparingDouble(m -> m.gain));
            }

            boolean applied = false;
            for (Move m : moves) {
                switch (m.kind) {
                    case ADD -> {
                        if (r.paths().existsDirectedPath(m.y, m.x)) continue;
                        t.addArc(m.x, m.y);
                        if (!budget.fits(r)) { t.removeArc(m.x, m.y); continue; }
                        addsApplied++;
                        applied = true;
                    }
                    case REMOVE -> {
                        t.removeArc(m.x, m.y);
                        removesApplied++;
                        applied = true;   // removal can never break the budget
                    }
                    case REVERSE -> {
                        t.removeArc(m.x, m.y);
                        if (r.paths().existsDirectedPath(m.x, m.y)) { t.addArc(m.x, m.y); continue; }
                        t.addArc(m.y, m.x);
                        if (!budget.fits(r)) {
                            t.removeArc(m.y, m.x);
                            t.addArc(m.x, m.y);
                            continue;
                        }
                        reversalsApplied++;
                        applied = true;
                    }
                }
                if (applied) break;
            }
            if (!applied) break;
            if (onStep != null) onStep.accept(r);
        }

        executionTime = (System.currentTimeMillis() - t0) / 1000.0;
        return r;
    }

    /**
     * Called after every applied move with the current DAG (not a copy), so a
     * caller can score the whole trajectory, as MCBNC's theta sweep does, and
     * select a point of it by a data-free criterion other than the objective.
     */
    public java.util.function.Consumer<Dag> onStep = null;

    private enum Kind { ADD, REMOVE, REVERSE }

    private static class Move {
        final Kind kind;
        final Node x, y;
        final double gain;
        final int tie;
        Move(Kind kind, Node x, Node y, double gain, int tie) {
            this.kind = kind; this.x = x; this.y = y; this.gain = gain; this.tie = tie;
        }
    }
}
