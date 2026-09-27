package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.util.*;

/**
 * Reports what the federated GES cache actually holds, so the federated
 * experiments can be sized without guessing: how many clients per network, how
 * many variables, and the treewidth spread of the inputs.
 *
 * Usage: ProbeGesCache
 */
public class ProbeGesCache {

    public static void main(String[] args) {
        File dir = new File("./results/Cache");
        File[] files = dir.listFiles((d, name) -> name.startsWith("cachedDAGs-") && name.endsWith(".ser"));
        if (files == null) { System.out.println("no cache dir"); return; }

        // Only the base-network caches (cachedDAGs-alarm.ser), not the per-seed ones
        // (cachedDAGs-alarm.0.ser), which belong to a different experiment.
        List<File> base = new ArrayList<>();
        for (File f : files) {
            String stem = f.getName().replace("cachedDAGs-", "").replace(".ser", "");
            if (!stem.matches(".*\\.\\d+$")) base.add(f);
        }
        base.sort(Comparator.comparing(File::getName));

        System.out.printf("%-14s %8s %6s %8s %10s %10s%n", "net", "clients", "n", "meanEdg", "tw min-max", "meanTW");
        for (File f : base) {
            String net = f.getName().replace("cachedDAGs-", "").replace(".ser", "");
            Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(f);
            if (cached.isEmpty()) { System.out.printf("%-14s %8s%n", net, "EMPTY"); continue; }

            List<Integer> keys = new ArrayList<>(cached.keySet());
            Collections.sort(keys);
            int n = cached.get(keys.get(0)).getNodes().size();

            double meanEdges = 0, meanTw = 0;
            int minTw = Integer.MAX_VALUE, maxTw = 0;
            for (int k : keys) {
                Dag d = cached.get(k);
                meanEdges += d.getNumEdges();
                int tw = Utils.getTreeWidth(d);
                meanTw += tw;
                minTw = Math.min(minTw, tw);
                maxTw = Math.max(maxTw, tw);
            }
            meanEdges /= keys.size();
            meanTw /= keys.size();

            System.out.printf(Locale.US, "%-14s %8d %6d %8.1f %10s %10.1f%n",
                    net, keys.size(), n, meanEdges, minTw + "-" + maxTw, meanTw);
        }
    }
}
