package consensusBN.moea;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import consensusBN.greedy.*;
import edu.cmu.tetrad.graph.*;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

/**
 * The genetic algorithm of the previous consensus papers (GA-E_c, fitness
 * against the inputs, greedy warm start, population 100, 100 generations),
 * run once per budget on a subset of levels. Same replicate definition and
 * output schema as FrontRun, so its rows join the main results.
 *
 * Usage: GaRun <cacheDir> <net> <r> <outCsv> <rep> <seed> <k1,k2,...>
 */
public class GaRun {

    public static void main(String[] args) throws Exception {
        String dir = args[0], net = args[1];
        int r = Integer.parseInt(args[2]);
        String outPath = args[3];
        int rep = Integer.parseInt(args[4]);
        long seed = Long.parseLong(args[5]);
        System.setProperty("cacheDir", dir);
        TreewidthBudget.fast = true;
        String tag = CacheDir.tag();

        Map<Integer, Dag> cached = CacheDir.read(net);
        List<Dag> dags = new ArrayList<>();
        for (int i = rep * r; i < (rep + 1) * r; i++) dags.add(cached.get(i));
        PairWeights w = new PairWeights(dags);
        Dag gold = GoldStandard.loadAlignedTo(net, w.nodes());
        Graph goldCpdag = GraphTransforms.dagToCpdag(new EdgeListGraph(gold));

        File outFile = new File(outPath);
        if (outFile.getParentFile() != null) outFile.getParentFile().mkdirs();
        boolean fresh = !outFile.exists();
        PrintWriter out = new PrintWriter(new FileWriter(outFile, true));
        if (fresh) out.println(FrontRun.HEADER + ",budget,shdInputs");
        List<Graph> inCpdags = new ArrayList<>();
        for (Dag d : dags) inCpdags.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));

        ConsensusUnion.metricSMHD = true;
        ConsensusUnion.metricAgainstOriginalDAGs = true;
        for (String ks : args[6].split(",")) {
            int k = Integer.parseInt(ks);
            long t = System.currentTimeMillis();
            GeneticTreeWidthUnion ga = new GeneticTreeWidthUnion(dags, (int) seed, k);
            ga.populationSize = 100;
            ga.numIterations = 100;
            ga.candidatesFromInitialDAGs = true;
            ga.repeatCandidates = false;
            ga.useGreedyWarmstart = true;
            Dag g = ga.fusionUnion();
            double time = FrontRun.secs(t);
            // Filed at its measured width, like every other method; the budget it
            // was run with is kept in the last column.
            int level = TreewidthBudget.treewidth(g);
            out.printf(Locale.US, "%s,%s,%d,%d,%d,%d,-1,%.4f,GA-Ec,%d,%s,%d,%.4f%n", tag, net, w.n(), r, rep, seed,
                    w.smhdLowerBound(), level, FrontRun.metrics(w, g, gold, goldCpdag, time), k,
                    FrontRun.shdToInputs(g, inCpdags));
            out.flush();
            System.out.printf("%s k=%d level=%d time=%.1f%n", net, k, level, time);
        }
        out.close();
    }
}
