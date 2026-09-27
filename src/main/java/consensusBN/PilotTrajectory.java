package consensusBN;

import consensusBN.Experiments.ExperimentMinCutGES;
import consensusBN.Experiments.RandomBN;
import consensusBN.greedy.MajorityGreedy;
import consensusBN.greedy.PairWeights;
import consensusBN.greedy.TreewidthBudget;
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
 * The AAAI evaluation protocol, applied to the new methods.
 *
 * That paper did NOT fix a treewidth budget. It ran MCBNC all the way from the
 * unrestricted fusion down towards the empty graph, recorded every point of the
 * trajectory, and then made the case that a purely structural, data-free
 * criterion — distance to the INPUT graphs — selects almost the same point as the
 * criterion you could only use with the answer in hand, distance to the GOLD
 * STANDARD, and that the selected point beats the average input.
 *
 * This harness reproduces that comparison for both families, so the new methods
 * are judged the way the previous paper judged its own:
 *
 *   - MCBNC sweeps its criticality threshold (experiments_perc with an infinite
 *     percentage runs the whole trajectory, snapshotting whenever the threshold
 *     rises).
 *   - MMG sweeps forward from the empty graph, one snapshot per applied move.
 *
 * The asymmetry to watch: for a forward search SMHD-to-inputs falls monotonically
 * by construction, so the data-free criterion always picks the endpoint. That
 * makes the question sharper rather than easier — if SMHD-to-gold turns upward
 * before the search stops, running to convergence overfits the inputs and the
 * data-free rule cannot see it. This measures that regret directly.
 *
 * Columns: net, r, method, step, threshold, edges, tw, smhdInputs, smhdGold,
 *          shdGold, meanInputSmhdGold
 * Lives in package consensusBN for the package-private MinCut internals.
 *
 * Usage: PilotTrajectory [nets] [rs]
 */
public class PilotTrajectory {

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0]
                : "alarm,insurance,child,water,mildew,hailfinder,barley,hepar2").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/trajectory.csv"));
        out.println("net,n,r,method,step,threshold,edges,tw,smhdInputs,smhdGold,shdGold,meanInputSmhdGold");

        for (String net : nets) {
            Map<Integer, Dag> cached = ExperimentMinCutGES.readCache(
                    new File("./results/Cache/cachedDAGs-" + net + ".ser"));
            if (cached.isEmpty()) { System.out.println("skip " + net); continue; }
            Dag gold = loadGold(net);

            for (String rStr : rs) {
                int r = Integer.parseInt(rStr);
                if (cached.size() < r) continue;
                List<Dag> dags = new ArrayList<>();
                for (int i = 0; i < r; i++) dags.add(cached.get(i));

                PairWeights w = new PairWeights(dags);
                double meanGoldAcc = 0;
                for (Dag d : dags) meanGoldAcc += Utils.SMHD(d, gold);
                final double meanInputGold = meanGoldAcc / r;

                System.out.printf(Locale.US, "%n=== %s n=%d r=%d (mean input SMHD to gold %.2f) ===%n",
                        net, w.n(), r, meanInputGold);

                // ---- MMG: forward, one snapshot per applied move ----
                MajorityGreedy mmg = new MajorityGreedy(dags);
                mmg.fastCandidateUpdate = true;
                int[] count = {0};
                mmg.stepListener = (step, cur) ->
                        emit(out, net, w, r, "MMG", step, Double.NaN, cur, gold, meanInputGold, count);
                emit(out, net, w, r, "MMG", 0, Double.NaN, new Dag(w.nodes()), gold, meanInputGold, count);
                mmg.search(TreewidthBudget.unbounded());
                System.out.printf("  MMG: %d trajectory points%n", count[0]);

                // ---- MCBNC: the AAAI sweep, fusion down towards empty ----
                MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, 2, Double.POSITIVE_INFINITY);
                mc.experiments_perc = true;
                mc.fusion();
                int[] c2 = {0};
                for (int i = 0; i < mc.outputExperimentDAGs.size(); i++) {
                    emit(out, net, w, r, "MCBNC", i,
                            mc.outputExperimentPercentages.get(i), mc.outputExperimentDAGs.get(i),
                            gold, meanInputGold, c2);
                }
                System.out.printf("  MCBNC: %d trajectory points%n", c2[0]);
                out.flush();
            }
        }
        out.close();
        System.out.println("wrote results/greedy/trajectory.csv");
    }

    static void emit(PrintWriter out, String net, PairWeights w, int r, String method,
                     int step, double threshold, Dag g, Dag gold, double meanInputGold, int[] count) {
        double smhdInputs = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
        double smhdGold = Utils.SMHD(g, gold);
        int shdGold = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)),
                                     GraphTransforms.dagToCpdag(new EdgeListGraph(gold)));
        out.printf(Locale.US, "%s,%d,%d,%s,%d,%s,%d,%d,%.4f,%.4f,%d,%.4f%n",
                net, w.n(), r, method, step,
                Double.isNaN(threshold) ? "" : String.format(Locale.US, "%.6f", threshold),
                g.getNumEdges(), Utils.getTreeWidth(g), smhdInputs, smhdGold, shdGold, meanInputGold);
        count[0]++;
    }

    static Dag loadGold(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbnGold = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
    }
}
