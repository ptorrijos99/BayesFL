package consensusBN.greedy;

import consensusBN.Experiments.RandomBN;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * Does shrinking the GES sample actually put the fusion back in the regime the
 * paper is about?
 *
 * The claim to test is that with 5000 rows the client DAGs are already close to
 * the gold standard, so a consensus has little to add — which is what `child`
 * and `mildew` looked like. With fewer rows the clients should get worse and
 * disagree more, the union should get noisier, and the gain of the consensus
 * over the average client should grow.
 *
 * Reports per (net, r, sampleSize): how good the average client is, how noisy
 * the union is, and what the cheap consensus methods achieve — so the whole
 * sample-size axis can be read off one table before committing to a full sweep.
 *
 * Usage: ProbeSampleSize [nets] [rs] [cacheDirs]
 *   e.g. ProbeSampleSize alarm,child 10,100 results/Cache-n500,results/Cache
 */
public class ProbeSampleSize {

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0]
                : "child,mildew,asia,sachs,water,insurance,alarm,barley,hailfinder,hepar2,win95pts").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");
        String[] dirs = (args.length > 2 ? args[2]
                : "results/Cache-n500,results/Cache-n1000,results/Cache").split(",");

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/sample_size.csv"));
        out.println("net,n,r,sampleSize,meanInputGold,meanInputEdges,meanInputTw,supportedPairs,"
                + "majorityPairs,noiseSkel,majMCSgold,mmgGold,hcEmptyGold,gainOverMeanInput");

        for (String dir : dirs) {
            System.setProperty("cacheDir", dir);
            String tag = CacheDir.tag();
            for (String net : nets) {
                for (String rStr : rs) {
                    int r = Integer.parseInt(rStr);
                    List<Dag> dags = CacheDir.clients(net, r);
                    if (dags == null) { System.out.println("skip " + tag + " " + net + " r=" + r); continue; }

                    PairWeights w = new PairWeights(dags);
                    Dag gold;
                    try { gold = GoldStandard.loadAlignedTo(net, w.nodes()); }
                    catch (Exception e) { System.out.println("skip " + net + ": " + e.getMessage()); continue; }
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

                    Dag majMCS = OrientationChooser.majoritySkeletonOriented(w);
                    MajorityGreedy m = new MajorityGreedy(dags);
                    m.fastCandidateUpdate = true;
                    Dag mmg = m.search(TreewidthBudget.unbounded());
                    Dag hc = new StructuralHillClimb(w).search(new Dag(w.nodes()), TreewidthBudget.unbounded());

                    double gMaj = Utils.SMHD(majMCS, gold), gMmg = Utils.SMHD(mmg, gold), gHc = Utils.SMHD(hc, gold);
                    double best = Math.min(gMaj, Math.min(gMmg, gHc));

                    out.printf(Locale.US, "%s,%d,%d,%s,%.4f,%.2f,%.2f,%d,%d,%.4f,%.2f,%.2f,%.2f,%.4f%n",
                            net, w.n(), r, tag, meanGold, eAcc / r, twAcc / r, supported, maj,
                            1.0 - maj / (double) supported, gMaj, gMmg, gHc,
                            (meanGold - best) / meanGold);
                    System.out.printf(Locale.US,
                            "%-11s n=%-5s r=%-4d meanInput %8.2f | noise %.3f | best consensus %8.2f | gain %+6.1f%%%n",
                            net, tag, r, meanGold, 1.0 - maj / (double) supported, best,
                            100 * (meanGold - best) / meanGold);
                }
            }
            out.flush();
        }
        out.close();
        System.out.println("wrote results/greedy/sample_size.csv");
    }

}
