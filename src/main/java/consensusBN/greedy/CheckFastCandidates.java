package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.util.*;

/**
 * The incremental candidate update must be an OPTIMISATION, not an
 * approximation: it has to reproduce the full-recomputation search edge for
 * edge, not merely reach a similar objective.
 *
 * Both paths break ties the same way (candidates in pool order, a->b before
 * b->a, stable sort by gain), so identical output is achievable and any
 * difference is a bug in the dirty-set rule rather than harmless nondeterminism.
 *
 * Usage: CheckFastCandidates
 */
public class CheckFastCandidates {

    public static void main(String[] args) {
        boolean ok = true;
        for (String net : new String[]{"alarm", "insurance", "hepar2", "win95pts"}) {
            for (int r : new int[]{10, 30}) {
                ok &= compare(net, r);
            }
        }
        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static boolean compare(String net, int r) {
        Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                new File("./results/Cache/cachedDAGs-" + net + ".ser"));
        if (cached.size() < r) return true;
        List<Dag> dags = new ArrayList<>();
        for (int i = 0; i < r; i++) dags.add(cached.get(i));

        boolean ok = true;
        for (int budget : new int[]{3, 5, 8, 12, Integer.MAX_VALUE}) {
            TreewidthBudget tb = budget == Integer.MAX_VALUE
                    ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);

            MajorityGreedy slow = new MajorityGreedy(dags);
            slow.fastCandidateUpdate = false;
            Dag a = slow.search(tb);

            MajorityGreedy fast = new MajorityGreedy(dags);
            fast.fastCandidateUpdate = true;
            Dag b = fast.search(tb);

            boolean same = sameEdges(a, b);
            ok &= same;
            System.out.printf(Locale.US,
                    "%-10s r=%-3d budget=%-6s slow %5d edges %8.3fs | fast %5d edges %8.3fs | speedup %5.1fx | %s%n",
                    net, r, budget == Integer.MAX_VALUE ? "none" : budget,
                    a.getNumEdges(), slow.executionTime, b.getNumEdges(), fast.executionTime,
                    fast.executionTime > 0 ? slow.executionTime / fast.executionTime : Double.NaN,
                    same ? "identical" : "FAIL-DIFFERENT");
            if (!same) {
                System.out.printf("    slow smhd=%.4f fast smhd=%.4f%n",
                        smhd(slow.weights(), a), smhd(fast.weights(), b));
            }
        }
        return ok;
    }

    static boolean sameEdges(Dag a, Dag b) {
        if (a.getNumEdges() != b.getNumEdges()) return false;
        Set<String> ea = new HashSet<>(), eb = new HashSet<>();
        for (Edge e : a.getEdges()) ea.add(key(a, e));
        for (Edge e : b.getEdges()) eb.add(key(b, e));
        return ea.equals(eb);
    }

    static String key(Dag g, Edge e) {
        Node x = e.getNode1(), y = e.getNode2();
        if (!g.isParentOf(x, y)) { Node t = x; x = y; y = t; }
        return x.getName() + "->" + y.getName();
    }

    static double smhd(PairWeights w, Dag g) {
        return w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
    }
}
