package consensusBN.greedy;

import consensusBN.Experiments.RandomBN;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import weka.classifiers.bayes.net.BIFReader;

import java.util.Locale;

import static org.albacete.simd.utils.Utils.readData;

/**
 * The structure-only loader must agree, arc for arc, with the existing
 * BIFReader route on every network where that route works. If it does not, the
 * gold standards of all previous results would silently change meaning.
 *
 * The new loader additionally has to work on networks where the old one throws,
 * which is the whole reason it exists.
 *
 * Usage: CheckGoldStandard
 */
public class CheckGoldStandard {

    public static void main(String[] args) {
        String[] legacy = {"alarm", "insurance", "child", "water", "mildew", "barley",
                           "hailfinder", "hepar2", "win95pts", "pathfinder", "andes",
                           "asia", "sachs", "diabetes", "pigs", "link"};
        boolean ok = true;

        System.out.printf("%-14s %6s %6s %8s %8s %s%n",
                "net", "nodes", "arcs", "legacy", "match", "note");
        for (String net : legacy) {
            Dag mine;
            try {
                mine = GoldStandard.load(net);
            } catch (Exception e) {
                ok = false;
                System.out.printf("%-14s %6s %6s %8s %8s %s%n", net, "-", "-", "-", "FAIL", e.getMessage());
                continue;
            }
            Dag old = null;
            String note = "";
            try {
                old = legacyLoad(net);
            } catch (Throwable t) {
                note = "legacy route throws: " + t.getClass().getSimpleName();
            }

            String match;
            if (old == null) {
                match = "n/a";
            } else {
                // Compare by NAME: the two graphs hold different Node objects, so
                // Tetrad's adjacency lookups cannot be used across them.
                boolean same = old.getNumNodes() == mine.getNumNodes()
                        && arcNames(old).equals(arcNames(mine));
                match = same ? "same" : "DIFFER";
                ok &= same;
            }
            System.out.printf(Locale.US, "%-14s %6d %6d %8s %8s %s%n",
                    net, mine.getNumNodes(), mine.getNumEdges(),
                    old == null ? "-" : String.valueOf(old.getNumEdges()), match, note);
        }

        // The case the new loader exists for: an imported network whose sampled
        // CSV need not contain every category.
        try {
            Dag g = GoldStandard.load("bnrep_pneumonia");
            System.out.printf(Locale.US, "%-14s %6d %6d %8s %8s %s%n",
                    "bnrep_pneumonia", g.getNumNodes(), g.getNumEdges(), "-", "ok", "imported");
            ok &= g.getNumNodes() == 62;
        } catch (Exception e) {
            System.out.println("bnrep_pneumonia FAILED: " + e.getMessage());
            ok = false;
        }

        // The check that was missing: arc-for-arc equality says nothing about
        // whether the gold can actually be COMPARED to a client DAG. Tetrad
        // indexes by Node instance, so an unaligned gold throws inside SMHD.
        System.out.println();
        for (String net : new String[]{"alarm", "hepar2", "bnrep_pneumonia"}) {
            try {
                java.util.List<Dag> dags = CacheDir.clients(net, 3);
                if (dags == null) { System.out.println("  usable-in-SMHD " + net + ": no cache, skipped"); continue; }
                Dag g = GoldStandard.loadAlignedTo(net, dags.get(0).getNodes());
                double d = org.albacete.simd.utils.Utils.SMHD(dags.get(0), g);
                System.out.printf(Locale.US, "  usable-in-SMHD %-16s ok (SMHD %.1f)%n", net, d);
            } catch (Exception e) {
                ok = false;
                System.out.println("  usable-in-SMHD " + net + ": FAIL " + e);
            }
        }

        System.out.println();
        System.out.println(ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    static java.util.Set<String> arcNames(Dag g) {
        java.util.Set<String> out = new java.util.HashSet<>();
        for (edu.cmu.tetrad.graph.Edge e : g.getEdges()) {
            edu.cmu.tetrad.graph.Node a = e.getNode1(), b = e.getNode2();
            if (!g.isParentOf(a, b)) { edu.cmu.tetrad.graph.Node t = a; a = b; b = t; }
            out.add(a.getName() + "->" + b.getName());
        }
        return out;
    }

    static Dag legacyLoad(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbn = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbn.originalBayesIm.getBayesPm().getDag());
    }
}
