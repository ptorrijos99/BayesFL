package consensusBN.greedy;

import consensusBN.ConsensusUnion;
import consensusBN.Experiments.ExperimentMinCutGES;
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
 * Quantifies the initialisation sensitivity of the local search across networks,
 * on federated inputs.
 *
 * The question this answers: forward-from-empty and descend-from-G+ each win in
 * some regime, so how often, by how much, and does it track anything observable
 * without data (tw(G+) against the budget)? That decides whether the paper can
 * offer a rule for choosing the starting point, which is the practical form of
 * the FES+BES argument.
 *
 * Rows are one per (net, r, budget, algorithm). "feasibleGPlus" records whether
 * G+ respected the budget at all, since when it does not, descending is simply
 * unavailable.
 *
 * Usage: PilotHillClimb [nets] [rs]
 */
public class PilotHillClimb {

    public static void main(String[] args) throws Exception {
        String[] nets = (args.length > 0 ? args[0]
                : "alarm,insurance,hepar2,child,water,mildew,barley,hailfinder,win95pts,pathfinder").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");
        int[] budgets = {3, 5, 8, 12, 16};

        new File("results/greedy").mkdirs();
        PrintWriter out = new PrintWriter(new FileWriter("results/greedy/pilot_hillclimb.csv"));
        out.println("net,n,r,budget,algorithm,edges,tw,smhd,skel,smhdGold,shdGold,time,"
                + "smhdLowerBound,twGPlus,feasibleGPlus");

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
                Dag gPlus = new Dag(ConsensusUnion.fusionUnion(dags));
                int twGPlus = Utils.getTreeWidth(gPlus);

                System.out.printf("%n=== %s n=%d r=%d tw(G+)=%d ===%n", net, w.n(), r, twGPlus);

                for (int budget : budgets) {
                    TreewidthBudget tb = new TreewidthBudget(budget);
                    boolean feasibleGPlus = tb.fits(gPlus);

                    MajorityGreedy mmg = new MajorityGreedy(dags);
                    mmg.fastCandidateUpdate = true;
                    Dag mmgDag = mmg.search(tb);
                    emit(out, net, r, budget, "MMG", mmgDag, mmg.executionTime, w, gold, twGPlus, feasibleGPlus);

                    StructuralHillClimb h1 = new StructuralHillClimb(w);
                    emit(out, net, r, budget, "HC-empty", h1.search(new Dag(w.nodes()), tb),
                            h1.executionTime, w, gold, twGPlus, feasibleGPlus);

                    StructuralHillClimb h2 = new StructuralHillClimb(w);
                    emit(out, net, r, budget, "HC-mmg", h2.search(new Dag(mmgDag), tb),
                            h2.executionTime, w, gold, twGPlus, feasibleGPlus);

                    if (feasibleGPlus) {
                        StructuralHillClimb h3 = new StructuralHillClimb(w);
                        emit(out, net, r, budget, "HC-gplus", h3.search(new Dag(gPlus), tb),
                                h3.executionTime, w, gold, twGPlus, true);
                    }

                    long t0 = System.currentTimeMillis();
                    Dag ea = ConsensusUnion.fusionUnion(dags, "MaxTreewidth", String.valueOf(budget));
                    emit(out, net, r, budget, "Greedy-Ea", ea,
                            (System.currentTimeMillis() - t0) / 1000.0, w, gold, twGPlus, feasibleGPlus);

                    // MCBNC deliberately omitted: this pilot is about initialisation
                    // sensitivity, its numbers are already in pilot_federated_small.csv,
                    // and it has no time cap here — it stalled this harness for hours on
                    // win95pts r=100, where the federated pilot had already clocked it
                    // over the 150 s cap.
                }
                out.flush();
            }
        }
        out.close();
        System.out.println("wrote results/greedy/pilot_hillclimb.csv");
    }

    static void emit(PrintWriter out, String net, int r, int budget, String alg, Dag g,
                     double time, PairWeights w, Dag gold, int twGPlus, boolean feasibleGPlus) {
        double smhd = w.smhdOf(w.objectiveOfMoral(Utils.moralize(g)));
        double skel = w.skelOf(w.objectiveOfSkeleton(g));
        double smhdGold = Utils.SMHD(g, gold);
        int shdGold = Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)),
                                     GraphTransforms.dagToCpdag(new EdgeListGraph(gold)));
        out.printf(Locale.US, "%s,%d,%d,%d,%s,%d,%d,%.4f,%.4f,%.4f,%d,%.4f,%.4f,%d,%s%n",
                net, w.n(), r, budget, alg, g.getNumEdges(), Utils.getTreeWidth(g),
                smhd, skel, smhdGold, shdGold, time, w.smhdLowerBound(), twGPlus, feasibleGPlus);
        System.out.printf(Locale.US, "  budget=%-3d %-10s edges=%5d tw=%3d smhd=%9.2f gold=%8.2f %7.2fs%n",
                budget, alg, g.getNumEdges(), Utils.getTreeWidth(g), smhd, smhdGold, time);
    }

    static Dag loadGold(String net) throws Exception {
        DataSet seedData = readData("./res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        reader.processFile("./res/networks/" + net + ".xbif");
        RandomBN rbnGold = new RandomBN(reader, seedData, 0, 1, 5);
        return new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
    }
}
