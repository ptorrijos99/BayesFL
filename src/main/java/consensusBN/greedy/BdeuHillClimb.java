package consensusBN.greedy;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.score.BdeuScore;

import java.util.*;

/**
 * Plain score-based hill climbing over DAGs (insertion, deletion and reversal of
 * one arc, best improvement, empty start) with the BDeu score, the classical
 * "HC" of bnlearn. It is one of the client learners of the heterogeneous
 * federations, next to GES and PC, so that the inputs of a consensus do not all
 * come from the same algorithm.
 *
 * The score decomposes over families, so only the column of the child whose
 * parent set changed is rescored after a move: delta[x][c] is the gain of
 * toggling the arc x->c (adding it when absent, deleting it when present), and a
 * reversal x->y is delta[x][y] + delta[y][x] evaluated on the current graph.
 */
public class BdeuHillClimb {

    private final BdeuScore score;
    private final List<Node> nodes;
    private final int n;
    private final List<Set<Integer>> pa;
    private final double[] local;
    private final double[][] delta;
    /** Moves applied by the last search(). */
    public int moves;

    public BdeuHillClimb(DataSet data) {
        this.score = new BdeuScore(data);
        this.nodes = data.getVariables();
        this.n = nodes.size();
        this.pa = new ArrayList<>();
        for (int i = 0; i < n; i++) pa.add(new TreeSet<>());
        this.local = new double[n];
        this.delta = new double[n][n];
    }

    private double fam(int c, Set<Integer> parents) {
        int[] p = new int[parents.size()];
        int k = 0;
        for (int x : parents) p[k++] = x;
        return score.localScore(c, p);
    }

    private void rescore(int c) {
        local[c] = fam(c, pa.get(c));
        for (int x = 0; x < n; x++) {
            if (x == c) continue;
            Set<Integer> s = new TreeSet<>(pa.get(c));
            if (!s.remove(x)) s.add(x);
            delta[x][c] = fam(c, s) - local[c];
        }
    }

    public Dag search() {
        Dag g = new Dag(nodes);
        for (int c = 0; c < n; c++) rescore(c);
        moves = 0;
        while (true) {
            // candidates in decreasing gain; the first acyclic one is applied
            List<double[]> cand = new ArrayList<>();
            for (int x = 0; x < n; x++)
                for (int y = 0; y < n; y++) {
                    if (x == y) continue;
                    boolean arc = pa.get(y).contains(x);
                    if (arc) {
                        if (delta[x][y] > 1e-9) cand.add(new double[]{delta[x][y], x, y, 1});
                        double rev = delta[x][y] + delta[y][x];
                        if (rev > 1e-9) cand.add(new double[]{rev, x, y, 2});
                    } else if (!pa.get(x).contains(y) && delta[x][y] > 1e-9) {
                        cand.add(new double[]{delta[x][y], x, y, 0});
                    }
                }
            if (cand.isEmpty()) break;
            cand.sort((a, b) -> Double.compare(b[0], a[0]));
            boolean done = false;
            for (double[] m : cand) {
                int x = (int) m[1], y = (int) m[2], kind = (int) m[3];
                Node nx = nodes.get(x), ny = nodes.get(y);
                if (kind == 0) {
                    if (g.paths().existsDirectedPath(ny, nx)) continue;
                    g.addEdge(Edges.directedEdge(nx, ny));
                    pa.get(y).add(x);
                    rescore(y);
                } else if (kind == 1) {
                    g.removeEdge(nx, ny);
                    pa.get(y).remove(x);
                    rescore(y);
                } else {
                    g.removeEdge(nx, ny);
                    if (g.paths().existsDirectedPath(nx, ny)) { g.addEdge(Edges.directedEdge(nx, ny)); continue; }
                    g.addEdge(Edges.directedEdge(ny, nx));
                    pa.get(y).remove(x);
                    pa.get(x).add(y);
                    rescore(y);
                    rescore(x);
                }
                moves++;
                done = true;
                break;
            }
            if (!done) break;
        }
        return g;
    }
}
