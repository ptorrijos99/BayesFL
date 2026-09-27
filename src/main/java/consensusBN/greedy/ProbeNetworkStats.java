package consensusBN.greedy;

import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * Structural profile of every gold-standard network available, so the imported
 * bnRep networks can be compared against the bnlearn ones on the properties that
 * actually drive this study: size, density, and the treewidth of the true
 * structure.
 *
 * Density matters because the consensus objective is about moral pairs, and
 * treewidth because the whole tractability argument is stated in those terms — a
 * benchmark suite whose gold standards are all trivially thin would make the
 * treewidth results vacuous.
 *
 * Usage: ProbeNetworkStats
 */
public class ProbeNetworkStats {

    public static void main(String[] args) {
        File dir = new File("./res/networks");
        File[] files = dir.listFiles((d, n) -> n.endsWith(".xbif"));
        if (files == null) { System.out.println("no networks"); return; }
        Arrays.sort(files, Comparator.comparing(File::getName));

        try (PrintWriter out = new PrintWriter(new FileWriter("results/greedy/network_stats.csv"))) {
            out.println("net,source,nodes,arcs,density,goldTw,maxParents");
            System.out.printf("%-30s %-8s %6s %6s %8s %7s %7s%n",
                    "net", "source", "nodes", "arcs", "arcs/n", "goldTw", "maxPa");
            for (File f : files) {
                String net = f.getName().replace(".xbif", "");
                String source = net.startsWith("bnrep_") ? "bnRep" : "bnlearn";
                try {
                    Dag g = GoldStandard.load(net);
                    int n = g.getNumNodes(), a = g.getNumEdges();
                    int maxPa = 0;
                    for (edu.cmu.tetrad.graph.Node v : g.getNodes())
                        maxPa = Math.max(maxPa, g.getParents(v).size());
                    int tw = Utils.getTreeWidth(g);
                    out.printf(Locale.US, "%s,%s,%d,%d,%.3f,%d,%d%n", net, source, n, a, a / (double) n, tw, maxPa);
                    System.out.printf(Locale.US, "%-30s %-8s %6d %6d %8.2f %7d %7d%n",
                            net, source, n, a, a / (double) n, tw, maxPa);
                } catch (Exception e) {
                    System.out.printf("%-30s %-8s  FAILED %s%n", net, source, e.getMessage());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("\nwrote results/greedy/network_stats.csv");
    }
}
