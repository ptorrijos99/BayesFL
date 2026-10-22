package consensusBN.moea;

import consensusBN.greedy.*;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.util.*;

/**
 * Memetic search for the whole treewidth / fidelity front of the data-free
 * consensus in one run.
 *
 * The archive has one cell per treewidth level (max clique size, the
 * convention of TreewidthBudget) and keeps the best DAG found at exactly that
 * level, MAP-Elites style. The front is the staircase "best DAG with treewidth
 * at most k" read off the cells. Every offspring is aimed at a target level k,
 * built by a variation operator, repaired to fit k, and polished by the exact
 * local search of StructuralHillClimb under budget k, then filed under the level
 * it actually reached. Solutions therefore move between levels, which is what a
 * separate run per budget cannot do.
 *
 * Operators, all problem-specific:
 *   KICK   remove a random subset of the arcs inside the largest cliques of the
 *          triangulated moral graph (the ones the budget is paying for) and re-climb;
 *   DOWN   take an elite from a higher level and repair it into k by deleting,
 *          one at a time, the cheapest arc inside an oversized clique;
 *   UP     take the elite from a lower level and climb with the looser budget k;
 *   CROSS  keep the arcs two elites share, add each arc they disagree on with
 *          probability 1/2 (acyclic only), repair into k and climb.
 *
 * Fitness is the objective of the journal's hill climb: the moral pair weights
 * plus the skeleton weights as a lexicographic tie-break, both exact and local.
 */
public class TreewidthArchiveMA {

    public enum Op { KICK, DOWN, UP, CROSS }

    public double timeLimit = 60;          // seconds of evolution after the seeds
    public int maxEvaluations = Integer.MAX_VALUE;
    public double kickFraction = 0.5;      // fraction of the max-clique arcs a kick removes
    public EnumSet<Op> ops = EnumSet.allOf(Op.class);

    /** Snapshots of the front, for anytime curves: (seconds, level -> objective). */
    public final List<double[]> trace = new ArrayList<>();
    public final Map<Op, int[]> opStats = new EnumMap<>(Op.class);   // {tries, improvements}
    public int evaluations;

    private final PairWeights w;
    private final double lex;
    private final Random rnd;
    private final int kMin, kMax;
    private final Cell[] cells;

    /**
     * Cell elite. Scores are compared lexicographically: moral objective first,
     * skeleton objective only on ties. Summing them with a small weight is NOT
     * lexicographic over whole graphs (the skeleton sum grows with the graph),
     * even though it is per move, which is how the local search uses it.
     */
    static final class Cell {
        Dag dag;
        double moral = Double.POSITIVE_INFINITY, skel = Double.POSITIVE_INFINITY;
    }

    static boolean better(double m1, double s1, double m2, double s2) {
        return m1 < m2 - 1e-9 || (Math.abs(m1 - m2) <= 1e-9 && s1 < s2 - 1e-9);
    }

    public TreewidthArchiveMA(PairWeights w, int kMin, int kMax, long seed) {
        this.w = w;
        this.lex = 1.0 / (2 * w.r());
        this.rnd = new Random(seed);
        this.kMin = kMin;
        this.kMax = kMax;
        this.cells = new Cell[kMax + 1];
        for (int k = 0; k <= kMax; k++) cells[k] = new Cell();
        for (Op o : Op.values()) opStats.put(o, new int[2]);
    }

    /** {moral objective, skeleton objective} of g. */
    public double[] score(Dag g) {
        return new double[]{w.objectiveOfMoral(Utils.moralize(g)), w.objectiveOfSkeleton(g)};
    }

    /** Files g under its own level; true if it became the elite there. */
    public boolean offer(Dag g) {
        int tw = TreewidthBudget.treewidth(g);
        if (tw > kMax) return false;
        double[] v = score(g);
        Cell c = cells[tw];
        if (better(v[0], v[1], c.moral, c.skel)) {
            c.moral = v[0];
            c.skel = v[1];
            c.dag = new Dag(g);
            return true;
        }
        return false;
    }

    private Cell frontCell(int k) {
        Cell best = null;
        for (int j = 0; j <= Math.min(k, kMax); j++)
            if (cells[j].dag != null && (best == null || better(cells[j].moral, cells[j].skel, best.moral, best.skel)))
                best = cells[j];
        return best;
    }

    /** Best DAG with treewidth at most k (the front at k), or null. */
    public Dag frontAt(int k) {
        Cell c = frontCell(k);
        return c == null ? null : c.dag;
    }

    /** Moral objective of the front at k (the part SMHD depends on). */
    public double frontObjAt(int k) {
        Cell c = frontCell(k);
        return c == null ? Double.POSITIVE_INFINITY : c.moral;
    }

    public int kMin() { return kMin; }
    public PairWeights weights() { return w; }
    public int kMax() { return kMax; }

    private void snapshot(double t) {
        double[] row = new double[kMax + 2];
        row[0] = t;
        for (int k = kMin; k <= kMax; k++) row[k + 1] = frontObjAt(k);
        trace.add(row);
    }

    /** Runs the evolution; the archive must already hold the seeds. */
    public void run() {
        long t0 = System.currentTimeMillis();
        snapshot(0);
        List<Op> opList = new ArrayList<>(ops);
        while (evaluations < maxEvaluations) {
            double el = (System.currentTimeMillis() - t0) / 1000.0;
            if (el >= timeLimit) break;
            int k = kMin + rnd.nextInt(kMax - kMin + 1);
            Op op = opList.get(rnd.nextInt(opList.size()));
            Dag child = switch (op) {
                case KICK -> kick(k);
                case DOWN -> down(k);
                case UP -> up(k);
                case CROSS -> cross(k);
            };
            if (child == null) continue;
            TreewidthBudget tb = new TreewidthBudget(k);
            if (!tb.fits(child)) continue;
            StructuralHillClimb hc = new StructuralHillClimb(w);
            hc.skelWeight = lex;
            child = hc.search(child, tb);
            evaluations++;
            int[] s = opStats.get(op);
            s[0]++;
            if (offer(child)) {
                s[1]++;
                snapshot((System.currentTimeMillis() - t0) / 1000.0);
            }
        }
        snapshot((System.currentTimeMillis() - t0) / 1000.0);
    }

    /** Elite from the best level at most k, perturbed inside its largest cliques. */
    private Dag kick(int k) {
        Dag p = frontAt(k);
        if (p == null) return null;
        Dag g = new Dag(p);
        List<Edge> inClique = arcsInLargestCliques(g, TreewidthBudget.treewidth(g));
        if (inClique.isEmpty()) inClique = new ArrayList<>(g.getEdges());
        Collections.shuffle(inClique, rnd);
        int m = Math.max(1, (int) Math.round(kickFraction * inClique.size() * rnd.nextDouble()));
        for (int i = 0; i < m && i < inClique.size(); i++) g.removeEdge(inClique.get(i));
        return g;
    }

    /** Elite from a random higher level, repaired into k. */
    private Dag down(int k) {
        List<Integer> higher = new ArrayList<>();
        for (int j = k + 1; j <= kMax; j++) if (cells[j].dag != null) higher.add(j);
        if (higher.isEmpty()) return kick(k);
        Dag g = new Dag(cells[higher.get(rnd.nextInt(higher.size()))].dag);
        return repair(g, k);
    }

    /** Elite from a lower level, to be climbed with the looser budget. */
    private Dag up(int k) {
        List<Integer> lower = new ArrayList<>();
        for (int j = kMin; j < k; j++) if (cells[j].dag != null) lower.add(j);
        if (lower.isEmpty()) return kick(k);
        return new Dag(cells[lower.get(rnd.nextInt(lower.size()))].dag);
    }

    /** Two elites from levels near k: shared arcs kept, disputed arcs by coin flip. */
    private Dag cross(int k) {
        List<Integer> near = new ArrayList<>();
        for (int j = Math.max(kMin, k - 2); j <= Math.min(kMax, k + 2); j++) if (cells[j].dag != null) near.add(j);
        if (near.size() < 2) return kick(k);
        Collections.shuffle(near, rnd);
        Dag a = cells[near.get(0)].dag, b = cells[near.get(1)].dag;
        Dag g = new Dag(w.nodes());
        List<Edge> disputed = new ArrayList<>();
        for (Edge e : a.getEdges()) {
            Edge eb = b.getEdge(e.getNode1(), e.getNode2());
            if (eb != null && eb.equals(e)) g.addEdge(e);
            else disputed.add(e);
        }
        for (Edge e : b.getEdges()) {
            Edge ea = a.getEdge(e.getNode1(), e.getNode2());
            if (ea == null || !ea.equals(e)) disputed.add(e);
        }
        Collections.shuffle(disputed, rnd);
        for (Edge e : disputed) {
            if (!rnd.nextBoolean()) continue;
            Node x = Edges.getDirectedEdgeTail(e), y = Edges.getDirectedEdgeHead(e);
            if (g.isAdjacentTo(x, y) || g.paths().existsDirectedPath(y, x)) continue;
            g.addEdge(Edges.directedEdge(x, y));
        }
        return repair(g, k);
    }

    /**
     * Deletes the cheapest arc inside an oversized clique until the DAG fits k.
     * "Cheapest" is the exact change in the objective, so the repair loses as
     * little fidelity per step as the local weights allow.
     */
    Dag repair(Dag g, int k) {
        MoralGraphTracker t = new MoralGraphTracker(w, g);
        while (true) {
            List<Set<Node>> big = FastTreewidth.cliquesAbove(g, k);
            if (big.isEmpty()) return g;
            Edge best = null; double bv = Double.POSITIVE_INFINITY;
            for (Edge e : g.getEdges()) {
                Node x = Edges.getDirectedEdgeTail(e), y = Edges.getDirectedEdgeHead(e);
                boolean inside = false;
                for (Set<Node> c : big) if (c.contains(x) && c.contains(y)) { inside = true; break; }
                if (!inside) continue;
                double d = t.gainRemoveArc(x, y) - lex * w.wSkel(w.idx(x), w.idx(y));
                if (d < bv) { bv = d; best = e; }
            }
            if (best == null) {   // fill edges only: fall back to any arc touching a big clique
                for (Edge e : g.getEdges()) {
                    for (Set<Node> c : big)
                        if (c.contains(e.getNode1()) || c.contains(e.getNode2())) { best = e; break; }
                    if (best != null) break;
                }
                if (best == null) return g;
            }
            t.removeArc(Edges.getDirectedEdgeTail(best), Edges.getDirectedEdgeHead(best));
        }
    }

    private List<Edge> arcsInLargestCliques(Dag g, int size) {
        List<Edge> out = new ArrayList<>();
        Set<Edge> seen = new HashSet<>();
        for (Set<Node> c : FastTreewidth.cliquesAbove(g, size - 1)) {
            for (Edge e : g.getEdges())
                if (c.contains(e.getNode1()) && c.contains(e.getNode2()) && seen.add(e)) out.add(e);
        }
        return out;
    }
}
