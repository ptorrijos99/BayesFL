package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * Numerical validation of the closed-form decomposition of the two data-free
 * consensus objectives, which the whole greedy framework rests on.
 *
 * Writing A = E(moral(R)) and B_i = E(moral(G_i)), and using
 * |A n B| = |A| + |B| - 2|A ^ B|, the mean SMHD of a candidate R to the r
 * input DAGs is
 *
 *     SMHD(R,{G_i}) = (1/r) [ SUM_i |B_i|  +  SUM_{ab in A} (r - 2 c_ab) ]
 *
 * where c_ab counts the inputs whose MORAL graph contains the pair {a,b}.
 * The first term does not depend on R, so minimising SMHD is exactly
 * maximising SUM_{ab in moral(R)} (2 c_ab - r): a per-pair weight computable
 * once in O(r n^2). The same identity holds for the skeleton with c_ab counted
 * on the SKELETONS instead of the moral graphs, which is the adjacency part
 * of SHD.
 *
 * This harness checks three claims:
 *   1. the closed form equals Utils.SMHD on arbitrary candidates,
 *   2. the same for the skeleton objective,
 *   3. the gain of inserting an arc is local: it only involves the pair {x,y}
 *      and the co-parent pairs {x,p} for p in Pa(y), so no candidate evaluation
 *      needs to touch the input DAGs.
 *
 * Usage: DiagDecomposition [net] [nDags] [seed]
 */
public class DiagDecomposition {

    static Map<String, Integer> idx = new HashMap<>();

    public static void main(String[] args) throws Exception {
        String net   = args.length > 0 ? args[0] : "alarm.0";
        int    nDags = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        int    seed  = args.length > 2 ? Integer.parseInt(args[2]) : 0;
        String PATH  = "./";

        String netName = net.split("\\.")[0];
        DataSet data = readData(PATH + "res/networks/BBDD/" + netName + "/" + net + ".csv");
        BIFReader reader = new BIFReader();
        reader.processFile(PATH + "res/networks/" + netName + ".xbif");

        RandomBN rbn = new RandomBN(reader, data, seed, nDags, 5);
        rbn.generate();
        List<Dag> dags = rbn.setOfRandomDags;
        int r = dags.size();

        List<Node> nodes = dags.get(0).getNodes();
        int n = nodes.size();
        for (int i = 0; i < n; i++) idx.put(nodes.get(i).getName(), i);

        // ---- per-pair counts, computed once: O(r n^2) ----
        int[][] cMoral = new int[n][n];
        int[][] cSkel  = new int[n][n];
        long sumMoral = 0, sumSkel = 0;
        for (Dag g : dags) {
            Graph m = Utils.moralize(g);
            for (Edge e : m.getEdges()) bump(cMoral, e);
            sumMoral += m.getNumEdges();
            for (Edge e : g.getEdges()) bump(cSkel, e);
            sumSkel += g.getNumEdges();
        }

        System.out.printf(Locale.US, "net=%s r=%d n=%d%n", net, r, n);
        System.out.printf(Locale.US, "moral-majority pairs (c>r/2): %d | skeleton-majority pairs: %d%n",
                countMajority(cMoral, r), countMajority(cSkel, r));

        // ---- bounds ----
        // Every candidate contributes SUM_{ab in A} (r - 2 c_ab) with A an undirected
        // graph, so relaxing A to ANY undirected graph gives the minimum at
        // A* = {ab : 2 c_ab > r}, the majority graph. For the SKELETON objective the
        // relaxation is exact: any orientation of A* consistent with a node ordering
        // is a DAG whose skeleton is A*, so the bound is ATTAINED. For SMHD it is a
        // strict relaxation, because not every undirected graph is the moral graph of
        // a DAG (C4 is not: any acyclic orientation has a sink whose two parents are
        // non-adjacent, which moralisation would join). So it is a lower bound and the
        // difference to it is a genuine optimality gap.
        double smhdLB = (sumMoral + majoritySum(cMoral, r)) / (double) r;
        double skelOPT = (sumSkel + majoritySum(cSkel, r)) / (double) r;
        System.out.printf(Locale.US, "SMHD lower bound (moral relaxation): %.4f | SKEL optimum (attained): %.4f%n%n",
                smhdLB, skelOPT);

        // ---- candidates ----
        Dag gPlus = ConsensusUnion.fusionUnion(dags);
        List<String> names = new ArrayList<>();
        List<Dag> cands = new ArrayList<>();

        names.add("empty");        cands.add(new Dag(nodes));
        names.add("G+");           cands.add(gPlus);
        for (String tw : new String[]{"3", "5", "8"}) {
            names.add("greedyEa-tw" + tw);
            cands.add(ConsensusUnion.fusionUnion(dags, "MaxTreewidth", tw));
        }
        Random rnd = new Random(seed);
        for (double frac : new double[]{0.25, 0.5, 0.75}) {
            names.add(String.format(Locale.US, "randsub-%.2f", frac));
            cands.add(randomSubgraph(gPlus, frac, rnd));
        }

        System.out.printf("%-16s %10s %12s %6s %8s   %10s %12s %6s %8s%n",
                "candidate", "SMHD dir", "SMHD closed", "ok", "gapLB", "SKEL dir", "SKEL closed", "ok", "gapOPT");
        boolean allOk = true;
        for (int i = 0; i < cands.size(); i++) {
            Dag R = cands.get(i);
            double sd = Utils.SMHD(R, dags);
            double sc = closed(Utils.moralize(R), cMoral, sumMoral, r);
            double kd = skelDirect(R, dags);
            double kc = closed(R, cSkel, sumSkel, r);
            boolean ok1 = Math.abs(sd - sc) < 1e-9, ok2 = Math.abs(kd - kc) < 1e-9;
            allOk &= ok1 && ok2;
            System.out.printf(Locale.US, "%-16s %10.4f %12.4f %6s %7.1f%%   %10.4f %12.4f %6s %7.1f%%%n",
                    names.get(i), sd, sc, ok1 ? "OK" : "FAIL", 100.0 * (sd - smhdLB) / smhdLB,
                    kd, kc, ok2 ? "OK" : "FAIL", 100.0 * (kd - skelOPT) / skelOPT);
        }

        // ---- the attained skeleton optimum, and what it costs in treewidth ----
        // Orienting the majority skeleton along the node ordering yields a DAG whose
        // skeleton is exactly the majority graph, so it attains skelOPT. Whether it is
        // USABLE is a separate question: its treewidth decides whether the
        // unconstrained optimum is anywhere near the tractable regime.
        Dag majDag = new Dag(nodes);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (2 * cSkel[i][j] > r) majDag.addEdge(Edges.directedEdge(nodes.get(i), nodes.get(j)));
        System.out.printf(Locale.US,
                "majority-skeleton DAG: edges=%d tw=%d SKEL=%.4f (opt %.4f) SMHD=%.4f (LB %.4f)%n",
                majDag.getNumEdges(), Utils.getTreeWidth(majDag), skelDirect(majDag, dags), skelOPT,
                Utils.SMHD(majDag, dags), smhdLB);
        System.out.printf(Locale.US, "input DAG treewidths: min=%d max=%d | tw(G+)=%d%n",
                dags.stream().mapToInt(Utils::getTreeWidth).min().orElse(-1),
                dags.stream().mapToInt(Utils::getTreeWidth).max().orElse(-1),
                Utils.getTreeWidth(gPlus));

        // ---- claim 3: locality of the insert gain ----
        System.out.println();
        System.out.println("Insert-gain locality (predicted from {x,y} + co-parent pairs vs. recomputed):");
        Dag R = randomSubgraph(gPlus, 0.5, rnd);
        int checked = 0, mismatches = 0;
        List<Edge> pool = new ArrayList<>(gPlus.getEdges());
        Collections.shuffle(pool, rnd);
        for (Edge e : pool) {
            if (checked >= 40) break;
            Node x = e.getNode1(), y = e.getNode2();
            if (R.isAdjacentTo(x, y)) continue;
            if (R.paths().existsDirectedPath(y, x)) continue;   // would create a cycle

            double before = objective(Utils.moralize(R), cMoral, r);
            double predicted = insertGain(R, x, y, cMoral, r);
            R.addEdge(Edges.directedEdge(x, y));
            double after = objective(Utils.moralize(R), cMoral, r);
            R.removeEdge(R.getEdge(x, y));

            checked++;
            if (Math.abs((after - before) - predicted) > 1e-9) {
                mismatches++;
                System.out.printf(Locale.US, "  MISMATCH %s->%s actual=%.4f predicted=%.4f%n",
                        x.getName(), y.getName(), after - before, predicted);
            }
        }
        System.out.printf("  checked=%d mismatches=%d%n", checked, mismatches);
        allOk &= (mismatches == 0);

        System.out.println();
        System.out.println(allOk ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    }

    /** SUM_{ab in E(g)} (r - 2 c_ab), the only R-dependent part of the objective. */
    static double objective(Graph g, int[][] c, int r) {
        double acc = 0;
        for (Edge e : g.getEdges()) {
            int a = idx.get(e.getNode1().getName()), b = idx.get(e.getNode2().getName());
            acc += r - 2.0 * c[a][b];
        }
        return acc;
    }

    static double closed(Graph g, int[][] c, long constTerm, int r) {
        return (constTerm + objective(g, c, r)) / r;
    }

    /** Gain of adding x->y: the pair {x,y} plus co-parent pairs {x,p}, p in Pa(y). */
    static double insertGain(Dag R, Node x, Node y, int[][] c, int r) {
        Graph m = Utils.moralize(R);
        double acc = 0;
        if (!m.isAdjacentTo(x, y)) acc += weight(x, y, c, r);
        for (Node p : R.getParents(y)) {
            if (p.equals(x)) continue;
            if (!m.isAdjacentTo(x, p)) acc += weight(x, p, c, r);
        }
        return acc;
    }

    static double weight(Node u, Node v, int[][] c, int r) {
        return r - 2.0 * c[idx.get(u.getName())][idx.get(v.getName())];
    }

    static void bump(int[][] c, Edge e) {
        int a = idx.get(e.getNode1().getName()), b = idx.get(e.getNode2().getName());
        c[a][b]++; c[b][a]++;
    }

    /** SUM over the majority graph {ab : 2 c_ab > r} of (r - 2 c_ab): the relaxed minimum. */
    static double majoritySum(int[][] c, int r) {
        double acc = 0;
        for (int i = 0; i < c.length; i++)
            for (int j = i + 1; j < c.length; j++)
                if (2 * c[i][j] > r) acc += r - 2.0 * c[i][j];
        return acc;
    }

    static int countMajority(int[][] c, int r) {
        int k = 0;
        for (int i = 0; i < c.length; i++)
            for (int j = i + 1; j < c.length; j++)
                if (2 * c[i][j] > r) k++;
        return k;
    }

    static double skelDirect(Dag R, List<Dag> dags) {
        double s = 0;
        for (Dag g : dags) s += Utils.SMHDwithoutMoralize(R, g);
        return s / dags.size();
    }

    static Dag randomSubgraph(Dag gPlus, double keepFrac, Random rnd) {
        Dag R = new Dag(gPlus);
        for (Edge e : new ArrayList<>(gPlus.getEdges()))
            if (rnd.nextDouble() > keepFrac) R.removeEdge(e);
        return R;
    }
}
