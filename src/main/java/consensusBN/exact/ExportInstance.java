package consensusBN.exact;

import consensusBN.greedy.*;
import consensusBN.moea.TreewidthArchiveMA;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.PrintWriter;
import java.util.*;

/**
 * Dumps one consensus instance (network, r, replicate) as JSON for the exact
 * solver in notebooks/ExactConsensus: the input DAGs and their CPDAGs, the gold
 * standard, and the heuristic fronts (PHC per budget, optionally TAMA) with the
 * metrics FrontRun reports, which serve both as warm starts and as a check that
 * the Python side measures the same distances.
 *
 * Replicates and budgets follow FrontRun: clients rep*r..(rep+1)*r-1, levels
 * k = 2..kMax under the max-clique-size convention of TreewidthBudget.
 *
 * Usage: ExportInstance <cacheDir> <net> <r> <rep> <outJson> [tamaSeconds] [kCap] [seed]
 */
public class ExportInstance {

    public static void main(String[] args) throws Exception {
        String dir = args[0], net = args[1];
        int r = Integer.parseInt(args[2]);
        int rep = Integer.parseInt(args[3]);
        String outPath = args[4];
        double tamaSeconds = args.length > 5 ? Double.parseDouble(args[5]) : 0;
        int kCap = args.length > 6 ? Integer.parseInt(args[6]) : 20;
        long seed = args.length > 7 ? Long.parseLong(args[7]) : 1;
        System.setProperty("cacheDir", dir);
        TreewidthBudget.fast = true;

        Map<Integer, Dag> cached = CacheDir.read(net);
        List<Dag> dags = new ArrayList<>();
        for (int i = rep * r; i < (rep + 1) * r; i++) dags.add(cached.get(i));
        PairWeights w = new PairWeights(dags);
        List<Graph> inCpdags = new ArrayList<>();
        for (Dag d : dags) inCpdags.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));
        Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());
        double lex = 1.0 / (2 * w.r());

        List<Dag> anchors = new ArrayList<>();
        for (Dag start : List.of(new Dag(w.nodes()), OrientationChooser.majoritySkeletonOriented(w))) {
            StructuralHillClimb s = new StructuralHillClimb(w);
            s.skelWeight = lex;
            anchors.add(s.search(start, TreewidthBudget.unbounded()));
        }
        int kMin = 2, kMax = kMin;
        for (Dag a : anchors) kMax = Math.max(kMax, TreewidthBudget.treewidth(a));
        kMax = Math.min(kMax, kCap);

        Dag[] phc = new Dag[kMax + 1];
        for (int k = kMin; k <= kMax; k++) {
            StructuralHillClimb s = new StructuralHillClimb(w);
            s.skelWeight = lex;
            phc[k] = s.search(new Dag(w.nodes()), new TreewidthBudget(k));
        }
        Dag kruskal = JournalRun.kruskalForest(w);

        Dag[] tama = null;
        if (tamaSeconds > 0) {
            TreewidthArchiveMA ma = new TreewidthArchiveMA(w, kMin, kMax, seed);
            ma.timeLimit = tamaSeconds;
            for (int k = kMin; k <= kMax; k++) ma.offer(phc[k]);
            ma.offer(kruskal);
            for (Dag a : anchors) ma.offer(a);
            ma.run();
            tama = new Dag[kMax + 1];
            for (int k = kMin; k <= kMax; k++) tama[k] = ma.frontAt(k);
        }

        File outFile = new File(outPath);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();
        try (PrintWriter out = new PrintWriter(outFile, "UTF-8")) {
            out.println("{");
            out.printf(Locale.US, "\"net\":\"%s\",\"r\":%d,\"rep\":%d,\"n\":%d,\"kMax\":%d,%n", net, r, rep, w.n(), kMax);
            out.printf(Locale.US, "\"smhdLB\":%.6f,\"skelOpt\":%.6f,%n", w.smhdLowerBound(), w.skeletonOptimum());
            StringBuilder names = new StringBuilder();
            for (Node v : w.nodes()) names.append(names.length() > 0 ? "," : "").append('"').append(v.getName()).append('"');
            out.println("\"nodes\":[" + names + "],");
            out.println("\"gold\":" + arcs(w, gold) + ",");
            out.println("\"inputs\":[");
            for (int i = 0; i < dags.size(); i++)
                out.println(arcs(w, dags.get(i)) + (i + 1 < dags.size() ? "," : ""));
            out.println("],");
            out.println("\"inputCpdags\":[");
            for (int i = 0; i < inCpdags.size(); i++)
                out.println(pdag(w, inCpdags.get(i)) + (i + 1 < inCpdags.size() ? "," : ""));
            out.println("],");
            out.println("\"solutions\":[");
            List<String> sols = new ArrayList<>();
            sols.add(solution(w, "Kruskal", 2, kruskal, inCpdags));
            for (int i = 0; i < anchors.size(); i++) sols.add(solution(w, "Anchor" + i, -1, anchors.get(i), inCpdags));
            for (int k = kMin; k <= kMax; k++) sols.add(solution(w, "PHC", k, phc[k], inCpdags));
            if (tama != null)
                for (int k = kMin; k <= kMax; k++)
                    if (tama[k] != null) sols.add(solution(w, "TAMA", k, tama[k], inCpdags));
            out.println(String.join(",\n", sols));
            out.println("]");
            out.println("}");
        }
        System.out.printf("%s r=%d rep=%d n=%d kMax=%d -> %s%n", net, r, rep, w.n(), kMax, outPath);
    }

    static String arcs(PairWeights w, Graph g) {
        StringBuilder sb = new StringBuilder("[");
        for (Edge e : g.getEdges()) {
            if (sb.length() > 1) sb.append(',');
            sb.append('[').append(w.idx(Edges.getDirectedEdgeTail(e))).append(',')
                    .append(w.idx(Edges.getDirectedEdgeHead(e))).append(']');
        }
        return sb.append(']').toString();
    }

    static String pdag(PairWeights w, Graph g) {
        StringBuilder d = new StringBuilder("["), u = new StringBuilder("[");
        for (Edge e : g.getEdges()) {
            if (e.isDirected()) {
                if (d.length() > 1) d.append(',');
                d.append('[').append(w.idx(Edges.getDirectedEdgeTail(e))).append(',')
                        .append(w.idx(Edges.getDirectedEdgeHead(e))).append(']');
            } else {
                if (u.length() > 1) u.append(',');
                u.append('[').append(w.idx(e.getNode1())).append(',').append(w.idx(e.getNode2())).append(']');
            }
        }
        return "{\"directed\":" + d + "],\"undirected\":" + u + "]}";
    }

    static String solution(PairWeights w, String method, int k, Dag g, List<Graph> inCpdags) {
        Graph c = GraphTransforms.dagToCpdag(new EdgeListGraph(g));
        double shd = 0;
        for (Graph x : inCpdags) shd += Utils.SHDundir(c, x);
        return String.format(Locale.US,
                "{\"method\":\"%s\",\"k\":%d,\"tw\":%d,\"smhdInputs\":%.6f,\"skelInputs\":%.6f,\"shdInputs\":%.6f,\"arcs\":%s}",
                method, k, TreewidthBudget.treewidth(g),
                w.smhdOf(w.objectiveOfMoral(Utils.moralize(g))), w.skelOf(w.objectiveOfSkeleton(g)),
                shd / inCpdags.size(), arcs(w, g));
    }
}
