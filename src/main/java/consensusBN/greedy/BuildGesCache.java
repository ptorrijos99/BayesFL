package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;

import java.io.File;
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
 * Usage: BuildGesCache <net> <nClients> <sampleSize>
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

        int learned = 0;
        long t0 = System.currentTimeMillis();
        for (int i = 0; i < nClients; i++) {
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
            System.out.printf(Locale.US, "%s n=%d client %d: %d vars, %d rows, %d edges, tw %d, %.2fs%n",
                    net, sampleSize, i, data.getNumColumns(), data.getNumRows(),
                    dag.getNumEdges(), Utils.getTreeWidth(dag),
                    (System.currentTimeMillis() - ti) / 1000.0);

            // Flush after EVERY client. On munin a single client costs >10 minutes,
            // so batching the writes risks losing hours to one crash or OOM kill.
            // Serialising the whole map each time costs seconds against that.
            ExperimentMinCutGES.writeCache(cacheFile, cached);
        }
        ExperimentMinCutGES.writeCache(cacheFile, cached);
        System.out.printf(Locale.US, "wrote %s (%d clients, %d newly learned, %.1fs total)%n",
                cacheFile.getPath(), cached.size(), learned,
                (System.currentTimeMillis() - t0) / 1000.0);
    }
}
