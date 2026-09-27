package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * One row per (network, r, sampleSize) with everything the analysis needs, so
 * the questions can be answered from a single table instead of joining half a
 * dozen partial sweeps.
 *
 * Includes the covariates that might explain when consensus helps — gold
 * density and treewidth, input quality, union noise, tw(G+) — alongside what the
 * cheap methods actually achieve. Whether density predicts anything is left to
 * the data; the point of recording it is to stop guessing.
 *
 * Budgeted comparisons are deliberately NOT here: this is the unconstrained
 * view, which is the protocol the AAAI paper uses. The budget sweep is separate.
 *
 * Usage: ProbeMasterTable <cacheDir> <outCsv> [rs] [nets]
 *   nets defaults to every network that has both an .xbif and a cache entry.
 */
public class ProbeMasterTable {

    public static void main(String[] args) throws Exception {
        String dir = args[0];
        String outPath = args[1];
        String[] rs = (args.length > 2 ? args[2] : "10,30,100").split(",");
        // Cap on network size. Everything here builds G+ and searches from it,
        // which does not terminate in useful time past a few hundred variables —
        // three runs burned a day stuck on link (724 vars) before this existed.
        // Large networks are measured by ProbeScaling under an external timeout.
        int maxNodes = Integer.getInteger("maxNodes", 250);
        System.setProperty("cacheDir", dir);
        String tag = CacheDir.tag();

        List<String> nets = new ArrayList<>();
        if (args.length > 3) {
            nets.addAll(Arrays.asList(args[3].split(",")));
        } else {
            File nd = new File("./res/networks");
            File[] fs = nd.listFiles((d, n) -> n.endsWith(".xbif"));
            if (fs != null) for (File f : fs) nets.add(f.getName().replace(".xbif", ""));
            Collections.sort(nets);
        }

        new File(outPath).getParentFile().mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter(outPath));
        out.println("sampleSize,net,source,n,r,goldArcs,goldDensity,goldTw,"
                + "meanInputGold,meanInputEdges,meanInputTw,supportedPairs,majorityPairs,noiseSkel,"
                + "twGPlus,gplusEdges,gplusGold,gplusInputs,"
                + "majMCSgold,majMCSinputs,mmgGold,mmgInputs,hcEmptyGold,hcEmptyInputs,"
                + "hcGplusGold,hcGplusInputs,bestGold,gainOverMeanInput,hcTime");

        for (String net : nets) {
            Dag goldRaw;
            try { goldRaw = GoldStandard.load(net); }
            catch (Exception e) { continue; }
            if (goldRaw.getNumNodes() > maxNodes) continue;

            for (String rStr : rs) {
                int r = Integer.parseInt(rStr);
                List<Dag> dags = CacheDir.clients(net, r);
                if (dags == null) continue;

                try {
                    PairWeights w = new PairWeights(dags);
                    Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());

                    double gAcc = 0, eAcc = 0, twAcc = 0;
                    for (Dag d : dags) {
                        gAcc += Utils.SMHD(d, gold);
                        eAcc += d.getNumEdges();
                        twAcc += Utils.getTreeWidth(d);
                    }
                    double meanGold = gAcc / r;

                    int supported = w.supportedPairs().size(), maj = 0;
                    for (int i = 0; i < w.n(); i++)
                        for (int j = i + 1; j < w.n(); j++)
                            if (2 * w.countSkel(i, j) > r) maj++;
                    double noise = 1.0 - maj / (double) supported;

                    Dag gPlus = new Dag(ConsensusUnion.fusionUnion(dags));
                    int twGPlus = Utils.getTreeWidth(gPlus);

                    Dag majMCS = OrientationChooser.majoritySkeletonOriented(w);
                    MajorityGreedy m = new MajorityGreedy(dags);
                    m.fastCandidateUpdate = true;
                    Dag mmg = m.search(TreewidthBudget.unbounded());
                    StructuralHillClimb hcE = new StructuralHillClimb(w);
                    Dag hc = hcE.search(new Dag(w.nodes()), TreewidthBudget.unbounded());
                    Dag hcG = new StructuralHillClimb(w).search(new Dag(gPlus), TreewidthBudget.unbounded());

                    double best = Math.min((double) Utils.SMHD(majMCS, gold),
                            Math.min((double) Utils.SMHD(mmg, gold),
                                    Math.min((double) Utils.SMHD(hc, gold), (double) Utils.SMHD(hcG, gold))));

                    out.printf(Locale.US,
                            "%s,%s,%s,%d,%d,%d,%.4f,%d,"
                            + "%.4f,%.2f,%.2f,%d,%d,%.4f,"
                            + "%d,%d,%.2f,%.4f,"
                            + "%.2f,%.4f,%.2f,%.4f,%.2f,%.4f,"
                            + "%.2f,%.4f,%.2f,%.4f,%.4f%n",
                            tag, net, net.startsWith("bnrep_") ? "bnRep" : "bnlearn", w.n(), r,
                            goldRaw.getNumEdges(), goldRaw.getNumEdges() / (double) w.n(),
                            Utils.getTreeWidth(goldRaw),
                            meanGold, eAcc / r, twAcc / r, supported, maj, noise,
                            twGPlus, gPlus.getNumEdges(), (double) Utils.SMHD(gPlus, gold), smhdIn(w, gPlus),
                            (double) (double) Utils.SMHD(majMCS, gold), smhdIn(w, majMCS),
                            (double) (double) Utils.SMHD(mmg, gold), smhdIn(w, mmg),
                            (double) (double) Utils.SMHD(hc, gold), smhdIn(w, hc),
                            (double) (double) Utils.SMHD(hcG, gold), smhdIn(w, hcG),
                            best, meanGold == 0 ? 0 : (meanGold - best) / meanGold, hcE.executionTime);
                    out.flush();
                    System.out.printf(Locale.US, "%-6s %-28s r=%-4d noise %.3f meanIn %8.2f best %8.2f gain %+6.1f%%%n",
                            tag, net, r, noise, meanGold, best,
                            meanGold == 0 ? 0 : 100 * (meanGold - best) / meanGold);
                } catch (Exception e) {
                    System.out.printf("%-6s %-28s r=%-4d SKIP %s%n", tag, net, r, e);
                }
            }
        }
        out.close();
        System.out.println("wrote " + outPath);
    }

    static double smhdIn(PairWeights w, Dag g) {
        return w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
    }
}
