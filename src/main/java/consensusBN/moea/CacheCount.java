package consensusBN.moea;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.Dag;

import java.io.File;
import java.util.Map;

/**
 * Number of learned clients (consecutive from 0) in a GES cache, per network.
 * Usage: CacheCount <cacheDir> <net>...
 */
public class CacheCount {
    public static void main(String[] args) {
        for (int i = 1; i < args.length; i++) {
            File f = new File(args[0] + "/cachedDAGs-" + args[i] + ".ser");
            Map<Integer, Dag> c = f.exists() ? ExperimentMinCutGES.readCache(f) : Map.of();
            int k = 0;
            while (c.containsKey(k)) k++;
            System.out.println(args[i] + " " + k);
        }
    }
}
