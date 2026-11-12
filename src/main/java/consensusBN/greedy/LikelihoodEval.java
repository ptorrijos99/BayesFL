package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;

/**
 * Probabilistic evaluation of a consensus structure. Once the structure is fixed,
 * its parameters can be learned without moving any data, since the counts of
 * each family are sums over the clients; this class does exactly that on the
 * first N rows of the r clients of the federation and reports the mean
 * log-likelihood per row of held-out data, client 99, which no federation of
 * r = 10 or r = 30 uses (replicates take clients 0..89).
 *
 * Parameters are posterior means with one pseudo-count per parent configuration
 * spread evenly over the states, theta_ijk = (n_ijk + 1/K_i) / (n_ij + 1), so an
 * unseen configuration gets the uniform distribution.
 *
 * Structures: every input (mean, as clientMean), the medoid, the averaged network,
 * PHC-Sel under both protocols, the averaged network and PHC-Sel under the
 * treewidth bounds b = 2..5 (max clique size, i.e. k = 1..4), the gold standard
 * and the empty graph, all fitted the same way.
 */
public class LikelihoodEval {

    static final String HEADER = "sampleSize,net,source,n,r,rep,noiseSkel,smhdLB,skelOpt,method,param,"
            + "edges,tw,llTest,time";

    static final int TEST_CLIENT = 99;

    final List<Node> nodes;
    final int[] card;
    final short[][] train, test;      // [row][var], var in the order of nodes

    LikelihoodEval(List<Node> nodes, List<String> trainFiles, int rowsPerFile, String testFile) throws Exception {
        this.nodes = nodes;
        int n = nodes.size();
        List<Map<String, Integer>> code = new ArrayList<>();
        for (int i = 0; i < n; i++) code.add(new HashMap<>());
        List<short[]> tr = new ArrayList<>();
        for (String f : trainFiles) read(f, rowsPerFile, code, tr);
        List<short[]> te = new ArrayList<>();
        read(testFile, Integer.MAX_VALUE, code, te);
        train = tr.toArray(new short[0][]);
        test = te.toArray(new short[0][]);
        card = new int[n];
        for (int i = 0; i < n; i++) card[i] = Math.max(1, code.get(i).size());
    }

    private void read(String path, int maxRows, List<Map<String, Integer>> code, List<short[]> rows) throws Exception {
        Map<String, Integer> pos = new HashMap<>();
        for (int i = 0; i < nodes.size(); i++) pos.put(nodes.get(i).getName(), i);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(Utils.openData(path)))) {
            String[] head = br.readLine().trim().split(",");
            int[] col = new int[head.length];
            for (int k = 0; k < head.length; k++) col[k] = pos.getOrDefault(head[k].trim(), -1);
            String line;
            int count = 0;
            while (count < maxRows && (line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] v = line.split(",", -1);
                short[] row = new short[nodes.size()];
                for (int k = 0; k < head.length; k++) {
                    if (col[k] < 0) continue;
                    Map<String, Integer> m = code.get(col[k]);
                    String s = v[k].trim().toLowerCase(Locale.ROOT);
                    Integer c = m.get(s);
                    if (c == null) { c = m.size(); m.put(s, c); }
                    row[col[k]] = (short) (int) c;
                }
                rows.add(row);
                count++;
            }
        }
    }

    /** Mean log-likelihood per test row of structure g with parameters fitted on the training rows. */
    double logLik(Graph g) {
        Map<String, Integer> pos = new HashMap<>();
        for (int i = 0; i < nodes.size(); i++) pos.put(nodes.get(i).getName(), i);
        double total = 0;
        for (int i = 0; i < nodes.size(); i++) {
            Node v = g.getNode(nodes.get(i).getName());
            List<Node> pa = v == null ? List.of() : g.getParents(v);
            int[] p = new int[pa.size()];
            for (int k = 0; k < p.length; k++) p[k] = pos.get(pa.get(k).getName());
            int K = card[i];
            Map<Object, int[]> counts = new HashMap<>();
            for (short[] row : train) {
                int[] c = counts.computeIfAbsent(key(row, p), z -> new int[K + 1]);
                c[row[i]]++;
                c[K]++;
            }
            for (short[] row : test) {
                int[] c = counts.get(key(row, p));
                double nk = c == null ? 0 : c[row[i]], nj = c == null ? 0 : c[K];
                total += Math.log((nk + 1.0 / K) / (nj + 1.0));
            }
        }
        return total / test.length;
    }

    private Object key(short[] row, int[] p) {
        long k = 0;
        boolean fits = true;
        for (int q : p) {
            if (k > Long.MAX_VALUE / (card[q] + 1)) { fits = false; break; }
            k = k * card[q] + row[q];
        }
        if (fits) return k;
        short[] s = new short[p.length];
        for (int q = 0; q < p.length; q++) s[q] = row[p[q]];
        return Arrays.toString(s);
    }

    /** JournalRun group "ll": every structure of the class comment, one row each. */
    static void run(List<Dag> dags, PairWeights w, PrintWriter out, String prefix, String net, int r, int rep,
                    int rowsPerClient, double wTime) throws Exception {
        if ((rep + 1) * r > TEST_CLIENT) return;     // the test client would be in the federation
        List<String> files = new ArrayList<>();
        for (int i = rep * r; i < (rep + 1) * r; i++) files.add(csv(net, i));
        LikelihoodEval ev = new LikelihoodEval(w.nodes(), files, rowsPerClient, csv(net, TEST_CLIENT));
        JournalRun.Row row = (method, param, g, time) -> {
            out.println(prefix + "," + method + "," + param + "," + String.format(Locale.US, "%d,%d,%.6f,%.4f",
                    g.getNumEdges(), Utils.getTreeWidth(g), ev.logLik(g), time));
            out.flush();
        };

        double acc = 0, edges = 0, tw = 0;
        for (Dag d : dags) { acc += ev.logLik(d); edges += d.getNumEdges(); tw += Utils.getTreeWidth(d); }
        out.println(prefix + ",clientMean,-," + String.format(Locale.US, "%.4f,%.4f,%.6f,0", edges / r, tw / r, acc / r));
        out.flush();

        row.emit("gold", "-", GoldStandard.loadAlignedTo(net, w.nodes()), 0);
        row.emit("empty", "-", new Dag(w.nodes()), 0);
        long t = System.currentTimeMillis();
        row.emit("medoid", "-", JournalRun.medoidMoral(w, dags), wTime + (System.currentTimeMillis() - t) / 1000.0);
        t = System.currentTimeMillis();
        row.emit("avgNet", "-", new AveragedNetwork().build(w), wTime + (System.currentTimeMillis() - t) / 1000.0);
        JournalRun.phcSelMoral(w, dags, row, "HC-best-free3-pool", wTime);
        t = System.currentTimeMillis();
        java.util.function.ToDoubleFunction<Dag> shdIn = JournalRun.shdToInputs(dags);
        Dag medS = JournalRun.medoidBy(dags, shdIn);
        double tPre = (System.currentTimeMillis() - t) / 1000.0;
        Object[] res = JournalRun.traj3(w, medS, true, shdIn);
        row.emit("HC-best-free3-pool@traj", "-", (Dag) res[1], wTime + tPre + (Double) res[2]);

        double lex = 1.0 / (2 * w.r());
        Dag med = JournalRun.onNodes(w, JournalRun.medoidMoral(w, dags));
        Dag maj = OrientationChooser.majoritySkeletonOriented(w);
        for (int b = 2; b <= 5; b++) {
            TreewidthBudget tb = new TreewidthBudget(b);
            t = System.currentTimeMillis();
            row.emit("avgNet", "b" + b, new AveragedNetwork().buildBounded(w, tb), wTime + (System.currentTimeMillis() - t) / 1000.0);
            Map<String, Dag> hc = new LinkedHashMap<>();
            Map<String, Double> times = new LinkedHashMap<>();
            times.put("HC-empty", JournalRun.climbCfg(w, new Dag(w.nodes()), lex, true, tb, null, "HC-empty", "-", wTime, hc));
            t = System.currentTimeMillis();
            Dag pm = JournalRun.pruneToBudget(w, new Dag(maj), tb, lex);
            double tp = (System.currentTimeMillis() - t) / 1000.0;
            times.put("HC-maj", JournalRun.climbCfg(w, pm, lex, true, tb, null, "HC-maj", "-", wTime + tp, hc));
            t = System.currentTimeMillis();
            Dag pd = JournalRun.pruneToBudget(w, new Dag(med), tb, lex);
            tp = (System.currentTimeMillis() - t) / 1000.0;
            times.put("HC-medoid", JournalRun.climbCfg(w, pd, lex, true, tb, null, "HC-medoid", "-", wTime + tp, hc));
            String bestK = null; double bestV = Double.POSITIVE_INFINITY;
            for (Map.Entry<String, Dag> e : hc.entrySet()) {
                double v = w.smhdOf(w.objectiveOfMoral(Utils.moralize(e.getValue())))
                        + w.skelOf(w.objectiveOfSkeleton(e.getValue())) * 1e-6;
                if (v < bestV) { bestV = v; bestK = e.getKey(); }
            }
            double total = -wTime * (times.size() - 1);
            for (double x : times.values()) total += x;
            row.emit("HC-best-free3-pool", "b" + b, hc.get(bestK), total);
        }
    }

    static String csv(String net, int i) {
        return "./res/networks/BBDD/" + net + "/" + net + "." + i + ".csv";
    }
}
