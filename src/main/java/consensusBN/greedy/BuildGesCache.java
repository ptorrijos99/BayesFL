package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;

import edu.cmu.tetrad.graph.EdgeListGraph;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Builds a GES cache at a chosen sample size, so the difficulty of the fusion
 * problem can be varied.
 *
 * The existing cache was learned from the full 5000 rows per client, which on
 * the small networks is enough for GES to recover almost the whole structure.
 * That is why `child` and `mildew` show no method beating the average client:
 * there is nothing for a consensus to repair. Shrinking the sample degrades the
 * inputs and puts the fusion back in the regime the paper is about.
 *
 * Each client CSV is an i.i.d. sample, so a prefix of it is also an i.i.d.
 * sample; taking the first `sampleSize` rows is therefore a valid smaller
 * dataset and avoids re-sampling from the network. It is also deterministic,
 * which keeps the caches reproducible.
 *
 * Output goes to results/Cache-n<sampleSize>/cachedDAGs-<net>.ser, in the same
 * format ExperimentMinCutGES.readCache expects, so every existing harness can
 * read it by pointing at the other directory.
 *
 * Usage: BuildGesCache <net> <nClients> <sampleSize> [relearn|-] [client|merge]
 *   relearn: comma-separated clients removed from the cache before the run.
 *   client: learn only this client and write it to its own file in parts/, so
 *     the clients of a slow network can run as parallel jobs without sharing
 *     the cache file.
 *   merge: move the files in parts/ into the cache and learn nothing.
 *   Without either, the files in parts/ are merged first and the missing
 *   clients are then learned one after another.
 *   sampleSize 0 or 5000 means "use the whole file".
 */
public class BuildGesCache {

    public static void main(String[] args) throws Exception {
        String net = args[0];
        int nClients = Integer.parseInt(args[1]);
        int sampleSize = Integer.parseInt(args[2]);

        String dir = "./results/Cache-n" + sampleSize + "/";
        new File(dir).mkdirs();
        File cacheFile = new File(dir + "cachedDAGs-" + net + ".ser");
        Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(cacheFile);
        // optional comma-separated clients to learn again, e.g. after their dataset was repaired;
        // their old DAGs are kept only to report whether the new one differs
        Map<Integer, Dag> old = new HashMap<>();
        if (args.length > 3 && !args[3].equals("-"))
            for (String s : args[3].split(",")) {
                int c = Integer.parseInt(s.trim());
                Dag d = cached.remove(c);
                if (d != null) old.put(c, d);
            }
        String mode = args.length > 4 ? args[4] : "";
        File partsDir = new File(dir + "parts");
        String partPrefix = "cachedDAGs-" + net + ".c";

        if (mode.isEmpty() || mode.equals("merge")) {
            File[] parts = partsDir.listFiles((d, f) -> f.startsWith(partPrefix) && f.endsWith(".ser"));
            int merged = 0;
            if (parts != null)
                for (File f : parts)
                    for (Map.Entry<Integer, Dag> e : ExperimentMinCutGES.readCache(f).entrySet()) {
                        cached.put(e.getKey(), e.getValue());   // a part is newer than the cache
                        merged++;
                    }
            if (merged > 0 || mode.equals("merge")) {
                save(cacheFile, cached);
                if (parts != null) for (File f : parts) f.delete();
            }
            List<Integer> missing = new ArrayList<>();
            for (int i = 0; i < nClients; i++) if (!cached.containsKey(i)) missing.add(i);
            System.out.printf("%s n=%d: %d clients merged from parts, %d in cache, missing %s%n",
                    net, sampleSize, merged, cached.size(), missing);
            if (mode.equals("merge")) return;
        }
        int only = mode.isEmpty() ? -1 : Integer.parseInt(mode);
        File partFile = new File(partsDir, partPrefix + only + ".ser");
        if (only >= 0) {
            if (cached.containsKey(only) || partFile.exists()) {
                System.out.printf("%s n=%d client %d: already learned%n", net, sampleSize, only);
                return;
            }
            partsDir.mkdirs();
        }

        int learned = 0;
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < nClients; i++) {
            if (only >= 0 && i != only) continue;
            if (cached.containsKey(i)) continue;   // resumable: long builds get interrupted
            DataSet data = Utils.readData("./res/networks/BBDD/" + net + "/" + net + "." + i + ".csv");
            if (sampleSize > 0 && sampleSize < data.getNumRows()) {
                int[] rows = new int[sampleSize];
                for (int k = 0; k < sampleSize; k++) rows[k] = k;
                int[] cols = new int[data.getNumColumns()];
                for (int k = 0; k < cols.length; k++) cols[k] = k;
                data = data.subsetRowsColumns(rows, cols);
            }
            long ti = System.currentTimeMillis();
            BNBuilder algorithm = new GES_BNBuilder(data, true);
            Dag dag = new Dag(algorithm.search());
            cached.put(i, dag);
            learned++;
            String vsOld = !old.containsKey(i) ? ""
                    : new EdgeListGraph(old.get(i)).equals(new EdgeListGraph(dag)) ? ", same as the old DAG"
                    : ", CHANGED from the old DAG (" + old.get(i).getNumEdges() + " edges)";
            System.out.printf(Locale.US, "%s n=%d client %d: %d vars, %d rows, %d edges, tw %d, %.2fs%s%n",
                    net, sampleSize, i, data.getNumColumns(), data.getNumRows(),
                    dag.getNumEdges(), Utils.getTreeWidth(dag),
                    (System.currentTimeMillis() - ti) / 1000.0, vsOld);

            if (only >= 0) {
                save(partFile, Map.of(i, dag));
                return;
            }
            // Flush after EVERY client. On munin a single client costs >10 minutes,
            // so batching the writes risks losing hours to one crash or OOM kill.
            // Serialising the whole map each time costs seconds against that.
            save(cacheFile, cached);
        }
        save(cacheFile, cached);
        System.out.printf(Locale.US, "wrote %s (%d clients, %d newly learned, %.1fs total)%n",
                cacheFile.getPath(), cached.size(), learned,
                (System.currentTimeMillis() - t0) / 1000.0);
    }

    /**
     * Writes the map as the whole cache. ExperimentMinCutGES.writeCache merges with
     * putIfAbsent, so an entry already in the file would win over a client learned
     * again; here the map replaces the file, through a temporary file and a rename.
     */
    static void save(File f, Map<Integer, Dag> m) throws IOException {
        File tmp = new File(f.getPath() + ".tmp");
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(tmp)))) {
            oos.writeObject(new HashMap<>(m));
        }
        Files.move(tmp.toPath(), f.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
}
