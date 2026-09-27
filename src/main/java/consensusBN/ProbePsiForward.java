package consensusBN;

import consensusBN.Experiments.ExperimentMinCutGES;
import consensusBN.Experiments.RandomBN;
import consensusBN.greedy.PairWeights;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * Can the min-cut criticality score drive a FORWARD phase?
 *
 * The spec flags this as the main risk of the CPDAG track. MCBNC uses Psi to
 * decide what to DELETE from G+, where the conditioning set comes from the
 * current dense graph. A forward phase starting from the empty graph has an
 * EMPTY conditioning set, and then Psi(x->y) is just the average min-cut between
 * x and y in the moralised ancestral graphs of the inputs — a count of
 * edge-disjoint paths, which rewards well-connected nodes whether or not the
 * specific pair is supported.
 *
 * Two things fall out of the definition before any measurement, and both matter:
 *   1. with an empty conditioning set Psi(x->y) == Psi(y->x), because the
 *      conditioned graph is the same object, so Psi cannot choose an orientation
 *      at the point where a forward search has to;
 *   2. Psi is bounded below by the number of inputs containing the direct edge,
 *      so it inherits majority support but adds path count on top.
 *
 * This probe measures whether the added path count helps or hurts, by ranking
 * candidate pairs by Psi and by plain majority support and scoring both against
 * the gold-standard skeleton. Lives in package consensusBN because the
 * Ford-Fulkerson helpers in MinCutTreeWidthUnion are package-private.
 *
 * Writes results/greedy/probe_psi.csv. Usage: ProbePsiForward [net] [r]
 */
public class ProbePsiForward {

    public static void main(String[] args) throws Exception {
        String net = args.length > 0 ? args[0] : "alarm";
        int r = args.length > 1 ? Integer.parseInt(args[1]) : 10;

        Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                new File("./results/Cache/cachedDAGs-" + net + ".ser"));
        List<Dag> dags = new ArrayList<>();
        for (int i = 0; i < r; i++) dags.add(cached.get(i));

        PairWeights w = new PairWeights(dags);
        Dag gold = loadGold(net);

        // Instance only needed for the package-private Ford-Fulkerson helpers.
        MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 1);

        // Degree in the majority skeleton, to test the hub concern directly.
        Dag maj = w.majoritySkeletonDag();

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/probe_psi.csv"));
        out.println("net,r,a,b,psi,psiReverse,cSkel,cMoral,inGold,degA,degB");

        int asymmetric = 0, total = 0;
        for (int[] p : w.supportedPairs()) {
            Node a = w.nodes().get(p[0]), b = w.nodes().get(p[1]);

            double psiAB = psi(mc, dags, a, b);
            double psiBA = psi(mc, dags, b, a);
            if (Math.abs(psiAB - psiBA) > 1e-9) asymmetric++;
            total++;

            boolean inGold = gold.isAdjacentTo(nodeIn(gold, a), nodeIn(gold, b));
            out.printf(Locale.US, "%s,%d,%s,%s,%.6f,%.6f,%d,%d,%b,%d,%d%n",
                    net, r, a.getName(), b.getName(), psiAB, psiBA,
                    w.countSkel(p[0], p[1]), w.countMoral(p[0], p[1]), inGold,
                    maj.getAdjacentNodes(maj.getNode(a.getName())).size(),
                    maj.getAdjacentNodes(maj.getNode(b.getName())).size());
        }
        out.close();

        System.out.printf("%s r=%d: %d candidate pairs, %d with Psi(a,b) != Psi(b,a) at the empty graph%n",
                net, r, total, asymmetric);
        System.out.println("wrote results/greedy/probe_psi.csv");
    }

    /** Criticality with an EMPTY conditioning set: what a forward phase would see first. */
    static double psi(MinCutTreeWidthUnion mc, List<Dag> dags, Node x, Node y) {
        double acc = 0;
        for (Dag g : dags) {
            Node gx = nodeIn(g, x), gy = nodeIn(g, y);
            Graph aux = mc.constructConditionedGraph(g, gy, gx, new LinkedList<>());
            Map<Node, List<MinCutTreeWidthUnion.EdgeFordFulkerson>> adj = mc.fordFulkerson(aux, gy, gx);
            acc += mc.findMinCut(adj, gy).size();
        }
        return acc / dags.size();
    }

    static Node nodeIn(Graph g, Node n) {
        Node r = g.getNode(n.getName());
        return r != null ? r : n;
    }

    static Dag loadGold(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbnGold = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
    }
}
