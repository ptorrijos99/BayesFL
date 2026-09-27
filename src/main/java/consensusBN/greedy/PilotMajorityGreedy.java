package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.Experiments.RandomBN;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

import static org.albacete.simd.utils.Utils.readData;

/**
 * Pilot comparison of MMG against the incumbent greedies and the computable
 * bound, on perturbed inputs, locally.
 *
 * Columns: net, r, seed, budget, algorithm, edges, tw, smhd, skel, time,
 *          smhdLowerBound, skeletonOptimum, gapPct
 * where gapPct is 100*(smhd - smhdLowerBound)/smhdLowerBound. Note the gap is
 * against an UNCONSTRAINED bound, so part of it is the price of the treewidth
 * budget rather than suboptimality; the budget=none rows are what separate the
 * two.
 *
 * Usage: PilotMajorityGreedy [nets] [rs] [seeds]
 *   e.g. PilotMajorityGreedy alarm.0,insurance.0,hepar2.0 5,10,20 0,1,2
 */
public class PilotMajorityGreedy {

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0] : "alarm.0,insurance.0,hepar2.0").split(",");
        String[] rs = (args.length > 1 ? args[1] : "5,10,20").split(",");
        String[] seeds = (args.length > 2 ? args[2] : "0,1,2").split(",");
        int[] budgets = {3, 5, 8, 12, Integer.MAX_VALUE};

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/pilot_mmg.csv"));
        out.println("net,r,seed,budget,algorithm,edges,tw,smhd,skel,time,smhdLowerBound,skeletonOptimum,gapPct");

        for (String net : nets) {
            String netName = net.split("\\.")[0];
            DataSet data = readData("./res/networks/BBDD/" + netName + "/" + net + ".csv");
            BIFReader reader = new BIFReader();
            reader.processFile("./res/networks/" + netName + ".xbif");

            for (String rStr : rs) {
                for (String seedStr : seeds) {
                    int r = Integer.parseInt(rStr), seed = Integer.parseInt(seedStr);
                    RandomBN rbn = new RandomBN(reader, data, seed, r, 5);
                    rbn.generate();
                    List<Dag> dags = rbn.setOfRandomDags;
                    PairWeights w = new PairWeights(dags);

                    for (int budget : budgets) {
                        TreewidthBudget tb = budget == Integer.MAX_VALUE
                                ? TreewidthBudget.unbounded() : new TreewidthBudget(budget);
                        String bLabel = budget == Integer.MAX_VALUE ? "none" : String.valueOf(budget);

                        MajorityGreedy mmg = new MajorityGreedy(dags);
                        emit(out, net, r, seed, bLabel, "MMG-moral", mmg.search(tb), mmg.executionTime, dags, w);

                        MajorityGreedy mmgS = new MajorityGreedy(dags);
                        mmgS.useSkeletonObjective = true;
                        emit(out, net, r, seed, bLabel, "MMG-skel", mmgS.search(tb), mmgS.executionTime, dags, w);

                        if (budget != Integer.MAX_VALUE) {
                            long t0 = System.currentTimeMillis();
                            Dag ea = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(budget));
                            emit(out, net, r, seed, bLabel, "Greedy-Ea", ea,
                                    (System.currentTimeMillis() - t0) / 1000.0, dags, w);

                            t0 = System.currentTimeMillis();
                            MinCutTreeWidthUnion mc = new MinCutTreeWidthUnion(dags, 10, budget);
                            Dag mcDag = mc.fusion();
                            emit(out, net, r, seed, bLabel, "MCBNC", mcDag,
                                    (System.currentTimeMillis() - t0) / 1000.0, dags, w);
                        }
                    }

                    emit(out, net, r, seed, "none", "majoritySkeletonDag",
                            w.majoritySkeletonDag(), 0.0, dags, w);
                    emit(out, net, r, seed, "none", "G+", ConsensusUnion.fusionUnion(dags), 0.0, dags, w);
                    emit(out, net, r, seed, "none", "empty", new Dag(w.nodes()), 0.0, dags, w);

                    out.flush();
                    System.out.printf("done %s r=%d seed=%d%n", net, r, seed);
                }
            }
        }
        out.close();
        System.out.println("wrote results/greedy/pilot_mmg.csv");
    }

    static void emit(PrintWriter out, String net, int r, int seed, String budget, String alg,
                     Dag g, double time, List<Dag> dags, PairWeights w) {
        double smhd = Utils.SMHD(g, dags);
        double skel = 0;
        for (Dag d : dags) skel += Utils.SMHDwithoutMoralize(g, d);
        skel /= dags.size();
        double lb = w.smhdLowerBound();
        out.printf(Locale.US, "%s,%d,%d,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f%n",
                net, r, seed, budget, alg, g.getNumEdges(), Utils.getTreeWidth(g),
                smhd, skel, time, lb, w.skeletonOptimum(), 100.0 * (smhd - lb) / lb);
    }
}
