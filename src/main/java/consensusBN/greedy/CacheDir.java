package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.Dag;

import java.io.File;
import java.util.*;

/**
 * Which GES cache to read.
 *
 * The sample size used to learn the client DAGs is an experimental axis, not a
 * constant: with 5000 rows GES already recovers most of the structure on the
 * small networks, so there is nothing left for a consensus to repair and those
 * networks look — misleadingly — like cases where fusion does not help. Caches
 * learned from 500 and 1000 rows put the problem back in the regime the paper is
 * about.
 *
 * Select with -DcacheDir=results/Cache-n500 (default: results/Cache, the
 * original 5000-row cache). Every probe and pilot reads through here so a whole
 * sweep can be re-pointed without touching its code.
 */
public class CacheDir {

    public static String path() {
        return System.getProperty("cacheDir", "./results/Cache");
    }

    /** Short tag for the CSV, so rows from different sample sizes stay separable. */
    public static String tag() {
        String p = path().replace('\\', '/');
        if (p.endsWith("/")) p = p.substring(0, p.length() - 1);
        String base = p.substring(p.lastIndexOf('/') + 1);
        // Cache-n1000, and the other learners' Cache-mix-n1000: the rows are what follows "-n"
        int k = base.lastIndexOf("-n");
        return base.startsWith("Cache") && k >= 0 ? base.substring(k + 2) : "5000";
    }

    public static Map<Integer, Dag> read(String net) {
        return ExperimentMinCutGES.readCache(new File(path() + "/cachedDAGs-" + net + ".ser"));
    }

    /** First r clients, or null when the cache does not hold that many. */
    public static List<Dag> clients(String net, int r) {
        Map<Integer, Dag> cached = read(net);
        if (cached.size() < r) return null;
        List<Dag> dags = new ArrayList<>();
        for (int i = 0; i < r; i++) {
            Dag d = cached.get(i);
            if (d == null) return null;
            dags.add(d);
        }
        return dags;
    }
}
