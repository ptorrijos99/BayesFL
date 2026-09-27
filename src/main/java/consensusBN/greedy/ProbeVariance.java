package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * Replicates every cell over DISJOINT subsets of clients, so the differences
 * between methods can be compared against their own sampling variability.
 *
 * Everything measured so far uses clients 0..r-1 of each cache — a single draw.
 * With 100 cached clients, r=10 admits 10 disjoint replicates and r=30 admits 3,
 * at no extra GES cost. This matters because the gaps that decide the paper are
 * narrow: mean ranks of 2.28 (MCBNC) against 2.95 (HC-empty), and means of 55.5
 * against 55.6. Without a spread those numbers cannot be read.
 *
 * Disjoint rather than bootstrap: the replicates are then genuinely independent
 * draws of the federation, which is the quantity of interest, rather than
 * resamples of one fixed federation.
 *
 * Usage: ProbeVariance <cacheDir> <outCsv> <r> [nets]
 */
public class ProbeVariance {

    public static void main(String[] args) throws Exception {
        String dir = args[0], outPath = args[1];
        int r = Integer.parseInt(args[2]);
        // Cap on network size. Everything here builds G+ and searches from it,
        // which does not terminate in useful time past a few hundred variables —
        // three runs burned a day stuck on link (724 vars) before this existed.
        // Large networks are measured by ProbeScaling under an external timeout.
        int maxNodes = Integer.getInteger("maxNodes", 250);
        System.setProperty("cacheDir", dir);
        String tag = CacheDir.tag();

        List<String> nets = new ArrayList<>();
        if (args.length > 3) nets.addAll(Arrays.asList(args[3].split(",")));
        else {
            File nd = new File("./res/networks");
            File[] fs = nd.listFiles((d, n) -> n.endsWith(".xbif"));
            if (fs != null) for (File f : fs) nets.add(f.getName().replace(".xbif", ""));
            Collections.sort(nets);
        }

        new File(outPath).getParentFile().mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter(outPath));
        out.println("sampleSize,net,source,n,r,rep,noiseSkel,meanInputGold,"
                + "majMCSgold,mmgGold,hcEmptyGold,hcGplusGold,gplusGold");

        for (String net : nets) {
            try { if (GoldStandard.load(net).getNumNodes() > maxNodes) continue; }
            catch (Exception e) { continue; }
            Map<Integer, Dag> cached = CacheDir.read(net);
            int reps = cached.size() / r;
            if (reps < 2) continue;

            for (int rep = 0; rep < reps; rep++) {
                List<Dag> dags = new ArrayList<>();
                boolean complete = true;
                for (int i = rep * r; i < (rep + 1) * r; i++) {
                    Dag d = cached.get(i);
                    if (d == null) { complete = false; break; }
                    dags.add(d);
                }
                if (!complete) continue;

                try {
                    PairWeights w = new PairWeights(dags);
                    Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());

                    double meanGold = 0;
                    for (Dag d : dags) meanGold += Utils.SMHD(d, gold);
                    meanGold /= r;

                    int supported = w.supportedPairs().size(), maj = 0;
                    for (int i = 0; i < w.n(); i++)
                        for (int j = i + 1; j < w.n(); j++)
                            if (2 * w.countSkel(i, j) > r) maj++;

                    Dag gPlus = new Dag(ConsensusUnion.fusionUnion(dags));
                    Dag majMCS = OrientationChooser.majoritySkeletonOriented(w);
                    MajorityGreedy m = new MajorityGreedy(dags);
                    m.fastCandidateUpdate = true;
                    Dag mmg = m.search(TreewidthBudget.unbounded());
                    Dag hc = new StructuralHillClimb(w).search(new Dag(w.nodes()), TreewidthBudget.unbounded());
                    Dag hcG = new StructuralHillClimb(w).search(new Dag(gPlus), TreewidthBudget.unbounded());

                    out.printf(Locale.US, "%s,%s,%s,%d,%d,%d,%.4f,%.4f,%d,%d,%d,%d,%d%n",
                            tag, net, net.startsWith("bnrep_") ? "bnRep" : "bnlearn", w.n(), r, rep,
                            1.0 - maj / (double) supported, meanGold,
                            Utils.SMHD(majMCS, gold), Utils.SMHD(mmg, gold),
                            Utils.SMHD(hc, gold), Utils.SMHD(hcG, gold), Utils.SMHD(gPlus, gold));
                    out.flush();
                } catch (Exception e) {
                    System.out.printf("%s %s rep %d SKIP %s%n", tag, net, rep, e);
                }
            }
            System.out.printf("%-6s %-28s r=%-4d %d replicates%n", tag, net, r, reps);
        }
        out.close();
        System.out.println("wrote " + outPath);
    }
}
