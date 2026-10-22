package consensusBN.exact;

import consensusBN.greedy.PairWeights;
import consensusBN.greedy.TreewidthBudget;
import consensusBN.moea.TreewidthArchiveMA;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * One round of TAMA from a given archive, for the alternation with the local
 * exact improvement of notebooks/ExactConsensus (alt_front.py). Works on node
 * indices only, so it needs no cache and no network file.
 *
 * Input (text): first line "n r kMin kMax nSeeds", then r lines with the input
 * DAGs and nSeeds lines with the seeds, each a blank-separated list of arcs
 * "a>b" (an empty line is the empty DAG). Output: one line per level of the
 * final front, "k tw smhd arcs".
 *
 * Usage: SeededTama <in.txt> <out.txt> <seconds> [seed] [ops]     (ops: e.g. KICK or KICK,UP,DOWN; default all)
 */
public class SeededTama {

    public static void main(String[] args) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get(args[0]));
        double seconds = Double.parseDouble(args[2]);
        long seed = args.length > 3 ? Long.parseLong(args[3]) : 1;
        TreewidthBudget.fast = true;
        String[] h = lines.get(0).trim().split(" ");
        int n = Integer.parseInt(h[0]), r = Integer.parseInt(h[1]);
        int kMin = Integer.parseInt(h[2]), kMax = Integer.parseInt(h[3]), nSeeds = Integer.parseInt(h[4]);
        List<Node> nodes = new ArrayList<>();
        for (int i = 0; i < n; i++) nodes.add(new GraphNode("v" + i));

        List<Dag> inputs = new ArrayList<>();
        for (int i = 0; i < r; i++) inputs.add(parse(nodes, lines.get(1 + i)));
        PairWeights w = new PairWeights(inputs);
        TreewidthArchiveMA ma = new TreewidthArchiveMA(w, kMin, kMax, seed);
        ma.timeLimit = seconds;
        if (args.length > 4) {
            ma.ops = EnumSet.noneOf(TreewidthArchiveMA.Op.class);
            for (String o : args[4].split(",")) ma.ops.add(TreewidthArchiveMA.Op.valueOf(o));
        }

        // The caller may kill the process at its deadline (one evaluation can be long on large
        // networks), so the front is also written every two seconds, without width and SMHD.
        Thread dump = new Thread(() -> {
            try {
                while (true) {
                    Thread.sleep(2000);
                    write(ma, w, kMin, kMax, args[1], false);
                }
            } catch (Exception e) { /* Interrupted at the end of the run */ }
        });
        dump.setDaemon(true);
        dump.start();
        for (int i = 0; i < nSeeds; i++) ma.offer(parse(w.nodes(), lines.get(1 + r + i)));
        ma.run();
        dump.interrupt();
        dump.join();
        write(ma, w, kMin, kMax, args[1], true);
    }

    static void write(TreewidthArchiveMA ma, PairWeights w, int kMin, int kMax, String file, boolean full) throws Exception {
        try (PrintWriter out = new PrintWriter(file + ".tmp", "UTF-8")) {
            for (int k = kMin; k <= kMax; k++) {
                Dag g = ma.frontAt(k);
                if (g == null) continue;
                StringBuilder sb = new StringBuilder();
                for (Edge e : g.getEdges())
                    sb.append(' ').append(Edges.getDirectedEdgeTail(e).getName().substring(1)).append('>')
                            .append(Edges.getDirectedEdgeHead(e).getName().substring(1));
                out.printf(Locale.US, "%d %d %.6f%s%n", k, full ? TreewidthBudget.treewidth(g) : 0,
                        full ? w.smhdOf(w.objectiveOfMoral(Utils.moralize(g))) : 0.0, sb);
            }
        }
        Files.move(Paths.get(file + ".tmp"), Paths.get(file), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                java.nio.file.StandardCopyOption.ATOMIC_MOVE);
    }

    static Dag parse(List<Node> nodes, String line) {
        Dag g = new Dag(nodes);
        for (String t : line.trim().split(" ")) {
            if (t.isEmpty()) continue;
            int c = t.indexOf('>');
            g.addEdge(Edges.directedEdge(nodes.get(Integer.parseInt(t.substring(0, c))),
                    nodes.get(Integer.parseInt(t.substring(c + 1)))));
        }
        return g;
    }
}
