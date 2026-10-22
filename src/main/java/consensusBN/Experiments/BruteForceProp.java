package consensusBN.Experiments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.GraphNode;
import edu.cmu.tetrad.graph.Node;

import consensusBN.ConsensusUnion;
import org.albacete.simd.utils.Utils;

/**
 * Ground-truth verifier and minimal-witness finder for Proposition 1.
 * Enumerates every DAG over a small universe, computes P1 (distance to G+) and
 * P2 (mean distance to inputs) optima under a maxClique bound, and the best
 * value reachable within each search space E_a, E_c, E_b, along with a concrete
 * separating DAG for each part (3), (4), (5).
 */
public class BruteForceProp {

    static Map<String, Node> V = new LinkedHashMap<>();
    static Node n(String name) { return V.computeIfAbsent(name, GraphNode::new); }

    static Dag g(List<Node> U, String... arcs) {
        Dag d = new Dag(U);
        for (String a : arcs) {
            String[] p = a.split("->");
            d.addEdge(new Edge(n(p[0].trim()), n(p[1].trim()), Endpoint.TAIL, Endpoint.ARROW));
        }
        return d;
    }

    static String sig(Dag d) {
        TreeSet<String> s = new TreeSet<>();
        for (Edge e : d.getEdges()) s.add(e.getNode1().getName() + "->" + e.getNode2().getName());
        return s.toString();
    }

    static Set<String> arcSet(Dag d) {
        Set<String> s = new TreeSet<>();
        for (Edge e : d.getEdges()) s.add(e.getNode1().getName() + "->" + e.getNode2().getName());
        return s;
    }

    static List<Edge> allArcs(List<Node> U) {
        List<Edge> arcs = new ArrayList<>();
        for (Node a : U) for (Node b : U) if (a != b) arcs.add(new Edge(a, b, Endpoint.TAIL, Endpoint.ARROW));
        return arcs;
    }

    static List<Node> cU = null;
    static List<Dag> cDags = null;
    static int[] cTw = null;

    static List<Dag> allDags(List<Node> U) {
        if (U == cU && cDags != null) return cDags;
        List<Edge> arcs = allArcs(U);
        List<Dag> out = new ArrayList<>();
        for (long mask = 0; mask < (1L << arcs.size()); mask++) {
            Dag d = new Dag(U);
            boolean ok = true;
            for (int i = 0; i < arcs.size(); i++) {
                if ((mask & (1L << i)) != 0) {
                    Edge e = arcs.get(i);
                    if (d.getEdge(e.getNode2(), e.getNode1()) != null) { ok = false; break; }
                    try { d.addEdge(new Edge(e.getNode1(), e.getNode2(), Endpoint.TAIL, Endpoint.ARROW)); }
                    catch (Exception ex) { ok = false; break; }
                }
            }
            if (ok) out.add(d);
        }
        cU = U; cDags = out; cTw = new int[out.size()];
        for (int i = 0; i < out.size(); i++) cTw[i] = Utils.getTreeWidth(out.get(i));
        return out;
    }

    static Dag prune(Dag input, Set<String> kept) {
        Dag d = new Dag(input.getNodes());
        for (Edge e : input.getEdges()) {
            if (kept.contains(e.getNode1().getName() + "->" + e.getNode2().getName()))
                d.addEdge(new Edge(e.getNode1(), e.getNode2(), Endpoint.TAIL, Endpoint.ARROW));
        }
        return d;
    }

    static List<String> gatherArcs(List<Dag> inputs, boolean perNet) {
        List<String> l = new ArrayList<>();
        Set<String> seen = new TreeSet<>();
        for (Dag d : inputs) for (Edge e : d.getEdges()) {
            String s = e.getNode1().getName() + "->" + e.getNode2().getName();
            if (perNet) l.add(s); else if (seen.add(s)) l.add(s);
        }
        return l;
    }

    static Set<String> reach(List<Dag> inputs, boolean perNet) {
        List<String> dec = gatherArcs(inputs, perNet);
        Set<String> res = new TreeSet<>();
        int m = dec.size();
        if (m > 16) return res;
        for (long mask = 0; mask < (1L << m); mask++) {
            List<Dag> pruned = new ArrayList<>();
            if (perNet) {
                int idx = 0;
                for (Dag in : inputs) {
                    Set<String> kept = new TreeSet<>();
                    for (Edge e : in.getEdges()) {
                        if ((mask & (1L << idx)) != 0) kept.add(e.getNode1().getName() + "->" + e.getNode2().getName());
                        idx++;
                    }
                    pruned.add(prune(in, kept));
                }
            } else {
                Set<String> kept = new TreeSet<>();
                for (int i = 0; i < m; i++) if ((mask & (1L << i)) != 0) kept.add(dec.get(i));
                for (Dag in : inputs) pruned.add(prune(in, kept));
            }
            try { res.add(sig(ConsensusUnion.fusionUnion(pruned))); } catch (Exception ex) { /* skip */ }
        }
        return res;
    }

    static class R {
        int flags;
        double optP1, optP2, eaP1, eaP2, ecP2, ebP1, ebP2;
        String gpEdges;
        int gpTw;
        String w3, w3ea, w4, w5;   // witness DAG edge-strings
    }

    static R analyze(List<Node> U, List<Dag> inputs, int k) {
        R r = new R();
        Dag gp = ConsensusUnion.fusionUnion(new ArrayList<>(inputs));
        r.gpEdges = gp.getEdges().toString();
        r.gpTw = Utils.getTreeWidth(gp);
        Set<String> gpArcs = arcSet(gp);
        Set<String> reachEc = reach(inputs, false);
        Set<String> reachEb = reach(inputs, true);

        List<Dag> dags = allDags(U);
        r.optP1 = r.optP2 = r.eaP1 = r.eaP2 = r.ecP2 = r.ebP1 = r.ebP2 = 1e9;
        for (int di = 0; di < dags.size(); di++) {
            if (cTw[di] > k) continue;
            Dag d = dags.get(di);
            double p1 = Utils.SMHD(d, gp);
            double p2 = Utils.SMHD(d, inputs);
            boolean inEa = gpArcs.containsAll(arcSet(d));
            boolean inEc = reachEc.contains(sig(d));
            boolean inEb = reachEb.contains(sig(d));
            if (p1 < r.optP1) r.optP1 = p1;
            if (p2 < r.optP2) r.optP2 = p2;
            if (inEa && p1 < r.eaP1) r.eaP1 = p1;
            if (inEa && p2 < r.eaP2) r.eaP2 = p2;
            if (inEc && p2 < r.ecP2) r.ecP2 = p2;
            if (inEb && p1 < r.ebP1) r.ebP1 = p1;
            if (inEb && p2 < r.ebP2) r.ebP2 = p2;
        }
        // second pass to capture concrete witness DAGs at the optimal values
        for (int di = 0; di < dags.size(); di++) {
            if (cTw[di] > k) continue;
            Dag d = dags.get(di);
            double p1 = Utils.SMHD(d, gp);
            double p2 = Utils.SMHD(d, inputs);
            boolean inEa = gpArcs.containsAll(arcSet(d));
            boolean inEc = reachEc.contains(sig(d));
            boolean inEb = reachEb.contains(sig(d));
            if (r.w3 == null && p2 <= r.optP2 + 1e-9 && !inEa) r.w3 = d.getEdges().toString();
            if (r.w3ea == null && inEa && p2 <= r.eaP2 + 1e-9) r.w3ea = d.getEdges().toString();
            if (r.w4 == null && inEb && !inEc && p2 <= r.ebP2 + 1e-9) r.w4 = d.getEdges().toString();
            if (r.w5 == null && inEa && !inEb && p1 <= r.eaP1 + 1e-9) r.w5 = d.getEdges().toString();
        }
        if (r.eaP2 > r.optP2 + 1e-9) r.flags |= 1;   // (3)
        if (r.ebP2 + 1e-9 < r.ecP2) r.flags |= 2;    // (4)
        if (r.eaP1 + 1e-9 < r.ebP1) r.flags |= 4;    // (5)
        return r;
    }

    static void print(String title, List<Node> U, List<Dag> inputs, int k) {
        R r = analyze(U, inputs, k);
        System.out.println("\n==== " + title + "  (k=" + k + ") ====");
        for (Dag in : inputs) System.out.println("   input = " + in.getEdges());
        System.out.println("   G+ = " + r.gpEdges + "   maxClique=" + r.gpTw);
        System.out.printf("   P1: global=%.2f E_a=%.2f E_b=%.2f  |  P2: global=%.2f E_a=%.2f E_c=%.2f E_b=%.2f%n",
                r.optP1, r.eaP1, r.ebP1, r.optP2, r.eaP2, r.ecP2, r.ebP2);
        System.out.println("   (3)=" + ((r.flags & 1) != 0) + " (4)=" + ((r.flags & 2) != 0) + " (5)=" + ((r.flags & 4) != 0));
        if ((r.flags & 1) != 0) System.out.println("   (3) P2-opt DAG unreachable under E_a: " + r.w3 + "   (E_a best P2 DAG: " + r.w3ea + ")");
        if ((r.flags & 2) != 0) System.out.println("   (4) P2-opt DAG in E_b not E_c: " + r.w4);
        if ((r.flags & 4) != 0) System.out.println("   (5) P1-opt DAG in E_a not E_b: " + r.w5);
    }

    /** Print the best feasible E_b pruning (per-network) and best E_c pruning (shared) for a P2 witness. */
    static void diagnose4(String title, List<Node> U, List<Dag> inputs, int k) {
        System.out.println("\n==== DIAGNOSE " + title + "  (k=" + k + ") ====");
        // E_b: per-network arc decisions
        List<int[]> netArcCount = new ArrayList<>();
        int total = 0;
        List<List<String>> perNetArcs = new ArrayList<>();
        for (Dag in : inputs) {
            List<String> a = new ArrayList<>();
            for (Edge e : in.getEdges()) a.add(e.getNode1().getName() + "->" + e.getNode2().getName());
            perNetArcs.add(a); total += a.size();
        }
        double bestP2 = 1e9; String bestDesc = "";
        for (long mask = 0; mask < (1L << total); mask++) {
            List<Dag> pruned = new ArrayList<>();
            StringBuilder desc = new StringBuilder();
            int idx = 0;
            for (int ni = 0; ni < inputs.size(); ni++) {
                Set<String> kept = new TreeSet<>();
                for (String s : perNetArcs.get(ni)) { if ((mask & (1L << idx)) != 0) kept.add(s); idx++; }
                pruned.add(prune(inputs.get(ni), kept));
                desc.append(" N").append(ni).append("=").append(kept);
            }
            Dag fused;
            try { fused = ConsensusUnion.fusionUnion(pruned); } catch (Exception ex) { continue; }
            if (Utils.getTreeWidth(fused) > k) continue;
            double p2 = Utils.SMHD(fused, inputs);
            if (p2 < bestP2 - 1e-9) { bestP2 = p2; bestDesc = "keep{" + desc + " } -> fuse " + fused.getEdges() + "  P2=" + p2; }
        }
        System.out.println("E_b best feasible P2 = " + bestP2 + "\n   " + bestDesc);

        // E_c: one decision per unique arc
        List<String> uniq = new ArrayList<>(new TreeSet<>(gatherArcs(inputs, false)));
        double bestC = 1e9; String bestCDesc = "";
        for (long mask = 0; mask < (1L << uniq.size()); mask++) {
            Set<String> kept = new TreeSet<>();
            for (int i = 0; i < uniq.size(); i++) if ((mask & (1L << i)) != 0) kept.add(uniq.get(i));
            List<Dag> pruned = new ArrayList<>();
            for (Dag in : inputs) pruned.add(prune(in, kept));
            Dag fused;
            try { fused = ConsensusUnion.fusionUnion(pruned); } catch (Exception ex) { continue; }
            if (Utils.getTreeWidth(fused) > k) continue;
            double p2 = Utils.SMHD(fused, inputs);
            if (p2 < bestC - 1e-9) { bestC = p2; bestCDesc = "keep" + kept + " -> fuse " + fused.getEdges() + "  P2=" + p2; }
        }
        System.out.println("E_c best feasible P2 = " + bestC + "\n   " + bestCDesc);
    }

    static java.util.Random RND = new java.util.Random(11);
    static Dag randomDag(List<Node> U, int maxArcs) {
        List<Edge> arcs = allArcs(U);
        java.util.Collections.shuffle(arcs, RND);
        Dag d = new Dag(U);
        int added = 0;
        for (Edge e : arcs) {
            if (added >= maxArcs) break;
            if (d.getEdge(e.getNode1(), e.getNode2()) != null || d.getEdge(e.getNode2(), e.getNode1()) != null) continue;
            try { d.addEdge(new Edge(e.getNode1(), e.getNode2(), Endpoint.TAIL, Endpoint.ARROW)); added++; } catch (Exception ex) {}
        }
        return d;
    }

    static int totalArcs(List<Dag> in) { int s = 0; for (Dag d : in) s += d.getNumEdges(); return s; }

    /** Find the minimal-arc witness (by total input arcs) for a target flag among random instances. */
    static void findMinimal(String label, List<Node> U, int targetFlag, int maxInputs, int maxArcsPer, int kMax, int trials) {
        List<Dag> best = null; int bestArcs = 999, bestK = 0;
        for (int t = 0; t < trials; t++) {
            int r = 2 + RND.nextInt(maxInputs - 1);
            List<Dag> inputs = new ArrayList<>();
            for (int i = 0; i < r; i++) inputs.add(randomDag(U, 1 + RND.nextInt(maxArcsPer)));
            if (totalArcs(inputs) >= bestArcs) continue;
            for (int k = 2; k <= kMax; k++) {
                R res;
                try { res = analyze(U, inputs, k); } catch (Exception ex) { continue; }
                if ((res.flags & targetFlag) == targetFlag && totalArcs(inputs) < bestArcs) {
                    best = inputs; bestArcs = totalArcs(inputs); bestK = k; break;
                }
            }
        }
        System.out.println("\n######## MINIMAL " + label + " (totalArcs=" + bestArcs + ") ########");
        if (best != null) print("minimal " + label, U, best, bestK);
        else System.out.println("   none found");
    }

    public static void main(String[] args) {
        List<Node> U3 = Arrays.asList(n("A"), n("B"), n("C"));
        List<Node> U4 = Arrays.asList(n("A"), n("B"), n("C"), n("D"));

        // Final chosen witnesses for the paper (match the appendix figures)
        print("(3) 2x collider {A->B,C->B} + 1x chain {A->B,B->C}", U3,
                Arrays.asList(g(U3, "A->B", "C->B"), g(U3, "A->B", "C->B"), g(U3, "A->B", "B->C")), 2);

        // Diagnose the current (4) instance: show the E_b pruning that wins
        diagnose4("(4) current: 2x {A,B,C->D} + 1x {D->A,B,C}", U4,
                Arrays.asList(g(U4, "A->D", "B->D", "C->D"), g(U4, "A->D", "B->D", "C->D"),
                        g(U4, "D->A", "D->B", "D->C")), 3);
        diagnose4("(4) minimal 2-input", U4,
                Arrays.asList(g(U4, "C->A", "B->A", "D->C"), g(U4, "A->C", "B->A", "D->C")), 2);

        // Re-inspect (5): find the best E_b consensus by P1 (distance to G+)
        {
            List<Dag> inputs = Arrays.asList(g(U4, "D->A", "C->A"), g(U4, "A->C", "B->C", "D->A"));
            Dag gp = ConsensusUnion.fusionUnion(new ArrayList<>(inputs));
            Set<String> reachEb = reach(inputs, true);
            List<Dag> dags = allDags(U4);
            double best = 1e9; String bestDag = "";
            for (int i = 0; i < dags.size(); i++) {
                if (cTw[i] > 2) continue;
                if (!reachEb.contains(sig(dags.get(i)))) continue;
                double p1 = Utils.SMHD(dags.get(i), gp);
                if (p1 < best) { best = p1; bestDag = dags.get(i).getEdges().toString(); }
            }
            System.out.println("\n==== (5) best E_b consensus by P1 ====");
            System.out.println("   G+ = " + gp.getEdges());
            System.out.println("   best E_b feasible DAG (P1=" + best + ") = " + bestDag);
        }

        print("(4) 2x collider {A,B,C->D} + 1x diverging {D->A,B,C}", U4,
                Arrays.asList(g(U4, "A->D", "B->D", "C->D"), g(U4, "A->D", "B->D", "C->D"),
                        g(U4, "D->A", "D->B", "D->C")), 3);

        print("(5) minimal 2-input", U4,
                Arrays.asList(g(U4, "D->A", "C->A"), g(U4, "A->C", "B->C", "D->A")), 2);
    }
}
