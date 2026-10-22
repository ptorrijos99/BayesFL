package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.Pc;
import edu.cmu.tetrad.search.test.IndTestChiSquare;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.util.*;

/**
 * Client DAGs learned with other algorithms than GES, for the heterogeneous
 * federations: PC (chi-square test, alpha 0.05, depth at most 5, CPDAG extended
 * to a DAG) and BDeu hill climbing. The "mix" cache takes client i from GES, PC
 * and HC in turn (i mod 3), reading the GES DAG from the existing cache, so a
 * federation of r clients mixes the three learners in equal parts.
 *
 * Output: results/Cache-<alg>-n<rows>/cachedDAGs-<net>.ser, in the format of the
 * GES caches, so JournalRun reads it unchanged with -DcacheDir pointing there.
 *
 * Usage: BuildAltCache <net> <pc|hc|mix> <nClients> <sampleSize>
 */
public class BuildAltCache {

    public static void main(String[] args) throws Exception {
        String net = args[0], alg = args[1];
        int nClients = Integer.parseInt(args[2]);
        int sampleSize = Integer.parseInt(args[3]);

        File cacheFile = cacheFile(alg, sampleSize, net);
        cacheFile.getParentFile().mkdirs();
        Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(cacheFile);
        Map<String, Map<Integer, Dag>> parts = new HashMap<>();
        if (alg.equals("mix")) {
            parts.put("ges", ExperimentMinCutGES.readCache(new File("./results/Cache-n" + sampleSize + "/cachedDAGs-" + net + ".ser")));
            parts.put("pc", ExperimentMinCutGES.readCache(cacheFile("pc", sampleSize, net)));
            parts.put("hc", ExperimentMinCutGES.readCache(cacheFile("hc", sampleSize, net)));
        }
        String[] turn = {"ges", "pc", "hc"};

        for (int i = 0; i < nClients; i++) {
            if (cached.containsKey(i)) continue;
            long t = System.currentTimeMillis();
            Dag dag;
            if (alg.equals("mix")) {
                Dag d = parts.get(turn[i % 3]).get(i);
                if (d == null) throw new IllegalStateException("missing " + turn[i % 3] + " client " + i + " of " + net);
                dag = d;
            } else {
                DataSet data = read(net, i, sampleSize);
                dag = alg.equals("pc") ? pc(data) : new BdeuHillClimb(data).search();
            }
            cached.put(i, dag);
            System.out.printf(Locale.US, "%s %s n=%d client %d: %d edges, %.2fs%n", net, alg, sampleSize, i,
                    dag.getNumEdges(), (System.currentTimeMillis() - t) / 1000.0);
            ExperimentMinCutGES.writeCache(cacheFile, cached);
        }
        ExperimentMinCutGES.writeCache(cacheFile, cached);
    }

    static File cacheFile(String alg, int sampleSize, String net) {
        return new File("./results/Cache-" + alg + "-n" + sampleSize + "/cachedDAGs-" + net + ".ser");
    }

    /** First sampleSize rows of client i (all of them for 0). */
    static DataSet read(String net, int i, int sampleSize) {
        DataSet data = Utils.readData("./res/networks/BBDD/" + net + "/" + net + "." + i + ".csv");
        if (sampleSize > 0 && sampleSize < data.getNumRows()) {
            int[] rows = new int[sampleSize];
            for (int k = 0; k < sampleSize; k++) rows[k] = k;
            int[] cols = new int[data.getNumColumns()];
            for (int k = 0; k < cols.length; k++) cols[k] = k;
            data = data.subsetRowsColumns(rows, cols);
        }
        return data;
    }

    /**
     * PC, then any DAG of the CPDAG. Conflicting orientations (bidirected edges)
     * and edges the extension cannot orient are made undirected first, and an
     * edge that would still close a cycle is dropped, so the result is always a DAG.
     */
    static Dag pc(DataSet data) {
        Pc pc = new Pc(new IndTestChiSquare(data, 0.05));
        pc.setDepth(5);
        pc.setGuaranteeCpdag(true);
        Graph g = pc.search();
        Graph clean = new EdgeListGraph(g.getNodes());
        for (Edge e : g.getEdges()) {
            if (Edges.isDirectedEdge(e)) clean.addEdge(e);
            else clean.addUndirectedEdge(e.getNode1(), e.getNode2());
        }
        Graph ext;
        try {
            ext = GraphTransforms.dagFromCpdag(clean);
        } catch (RuntimeException ex) {
            ext = clean;
        }
        Dag d = new Dag(g.getNodes());
        for (Edge e : ext.getEdges()) {
            Node a = e.getNode1(), b = e.getNode2();
            if (Edges.isDirectedEdge(e)) { a = Edges.getDirectedEdgeTail(e); b = Edges.getDirectedEdgeHead(e); }
            if (d.paths().existsDirectedPath(b, a)) { Node t = a; a = b; b = t; }
            if (!d.paths().existsDirectedPath(b, a)) d.addEdge(Edges.directedEdge(a, b));
        }
        return d;
    }
}
