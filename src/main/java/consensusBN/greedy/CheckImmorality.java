package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;

import java.util.*;

/**
 * Checks the incremental immorality gains against full recomputation.
 *
 * Applies random insertions and deletions to a DAG and compares, for every
 * move, ImmoralityWeights.gainAdd/gainRemove divided by r with the change in
 * ImmoralityWeights.distance. Any mismatch is a bug in the local rule.
 *
 * Usage: CheckImmorality <cacheDir> <net> <r> [moves]
 */
public class CheckImmorality {

    public static void main(String[] args) {
        System.setProperty("cacheDir", args[0]);
        List<Dag> dags = CacheDir.clients(args[1], Integer.parseInt(args[2]));
        int moves = args.length > 3 ? Integer.parseInt(args[3]) : 2000;
        PairWeights w = new PairWeights(dags);
        ImmoralityWeights imm = new ImmoralityWeights(dags, w);
        int r = dags.size();

        Random rnd = new Random(7);
        Dag g = new Dag(dags.get(0));
        List<Node> nodes = g.getNodes();
        int mismatches = 0, checked = 0;
        for (int k = 0; k < moves; k++) {
            Node x = nodes.get(rnd.nextInt(nodes.size())), y = nodes.get(rnd.nextInt(nodes.size()));
            if (x.equals(y)) continue;
            double before = imm.distance(g);
            double gain;
            if (g.isParentOf(x, y)) {
                gain = imm.gainRemove(g, x, y);
                g.removeEdge(g.getEdge(x, y));
            } else if (!g.isAdjacentTo(x, y) && !g.paths().existsDirectedPath(y, x)) {
                gain = imm.gainAdd(g, x, y);
                g.addEdge(Edges.directedEdge(x, y));
            } else continue;
            double after = imm.distance(g);
            checked++;
            if (Math.abs((after - before) - gain / r) > 1e-9) {
                mismatches++;
                if (mismatches < 5)
                    System.out.printf("MISMATCH %s->%s predicted %.4f actual %.4f%n", x, y, gain / r, after - before);
            }
        }
        System.out.printf("%s: %d moves checked, %d mismatches%n", args[1], checked, mismatches);
    }
}
