package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Robustness of structural consensus (P2) against fusion approximation (P1)
 * when a fraction of the input networks is adversarial.
 *
 * Of the r input DAGs, r-k are obtained by controlled perturbation of the gold
 * standard, as in the main experiments, and k are dense DAGs drawn independently
 * of the gold structure over the same variables. Sweeping k tests the claim that
 * P2 degrades more gracefully than P1, because it weights dependencies by their
 * support across inputs instead of preserving every asserted independence.
 *
 * Quality is measured against the gold standard, which the adversarial clients do
 * not share, so recovery of the true structure is what is being scored.
 *
 * Usage: AdversarialRobustness <net> [seed] [r] [outCsv]
 */
public class AdversarialRobustness {

    static final String PATH = System.getProperty("user.dir") + "/";

    /** Fractions of adversarial clients. Beyond 0.5 the majority structure is undefined. */
    static final double[] K_FRACTIONS = {0.0, 0.1, 0.2, 0.3, 0.4};

    /** Density of an adversarial DAG relative to the gold standard edge count. */
    static final double ADVERSARIAL_DENSITY = 1.5;

    /** true = all adversarial clients share one wrong structure (coordinated attack). */
    static boolean COORDINATED = true;

    /** true = clean clients are GES-learned (federated); false = perturbations of the gold. */
    static boolean useGES = true;

    static final int POP_SIZE = 100;
    static final int N_ITERATIONS = 100;

    public static void main(String[] args) throws IOException {
        String net  = args.length > 0 ? args[0] : "alarm";
        int seed    = args.length > 1 ? Integer.parseInt(args[1]) : 42;
        int r       = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        String out  = args.length > 3 ? args[3]
                : PATH + "results/adversarial/" + net + "_seed" + seed + "_r" + r + ".csv";

        new File(out).getParentFile().mkdirs();

        // --- Gold standard and clean input DAGs -------------------------------
        DataSet data = Utils.readData(PATH + "res/networks/BBDD/" + net + "/" + net + ".0.csv");
        BIFReader reader = new BIFReader();
        try { reader.processFile(PATH + "res/networks/xbif/" + net + ".xbif"); }
        catch (Exception e) { throw new RuntimeException(e); }

        RandomBN rbnGold = new RandomBN(reader, data, seed, 1, 5);
        Dag goldDag = new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());
        int twGold = Utils.getTreeWidth(goldDag);

        // Clean clients. The federated source (GES-learned structures) is the
        // meaningful setting for this test: perturbed inputs already fuse into a
        // graph far from the gold standard, which masks the effect of an adversary.
        List<Dag> cleanDags;
        if (useGES) {
            cleanDags = new ArrayList<>(ExperimentsJournal.loadGESDags(net + ".0", net, r));
        } else {
            RandomBN rbn = new RandomBN(reader, data, seed, r, 5);
            rbn.generate();
            cleanDags = new ArrayList<>(rbn.setOfRandomDags);
        }

        List<Node> nodes = goldDag.getNodes();
        int goldEdges = goldDag.getNumEdges();
        Graph cpdagGold = GraphTransforms.dagToCpdag(new EdgeListGraph(goldDag));

        // Treewidth budgets: exactly enough for the truth, and two slacker settings.
        int[] limits = { twGold, (int) Math.round(1.5 * twGold), 2 * twGold };

        System.out.println("net=" + net + "  r=" + r + "  seed=" + seed
                + "  tw(gold)=" + twGold + "  |E(gold)|=" + goldEdges);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(out))) {
            bw.write("net,seed,r,k,kFrac,limitTW,twGold,twUnion,unionEdges,"
                   + "unionSMHDgold,unionSHDgold,objective,algo,smhdGold,shdGold,tw,edges,time\n");

            for (double kFrac : K_FRACTIONS) {
                int k = (int) Math.round(kFrac * r);

                // Replace the last k clean clients by adversarial ones. The seed is
                // offset per client so the corrupted set is reproducible but distinct.
                List<Dag> dags = new ArrayList<>(cleanDags.subList(0, r - k));
                for (int i = 0; i < k; i++) {
                    // Coordinated attack: every adversarial client is drawn from the
                    // same seed, so they assert one consistent wrong structure. This is
                    // the case the majority argument is about. Independent draws behave
                    // as unstructured noise and cancel out across clients.
                    long advSeed = COORDINATED ? seed * 1000L : seed * 1000L + i;
                    dags.add(randomDenseDag(nodes,
                            (int) Math.round(ADVERSARIAL_DENSITY * goldEdges),
                            new Random(advSeed)));
                }

                // Unrestricted fusion of the contaminated set, as a reference point.
                ConsensusUnion.metricSMHD = true;
                ConsensusUnion.metricAgainstOriginalDAGs = false;
                Dag union = ConsensusUnion.fusionUnion(dags);
                int twUnion = Utils.getTreeWidth(union);
                int unionSmhd = Utils.SMHD(union, goldDag);
                int unionShd = Utils.SHDundir(
                        GraphTransforms.dagToCpdag(new EdgeListGraph(union)), cpdagGold);

                System.out.printf("  k=%d (%.0f%%)  tw(G+)=%d  SMHD(G+,gold)=%d%n",
                        k, 100 * kFrac, twUnion, unionSmhd);

                for (int t : limits) {
                    for (boolean againstOriginals : new boolean[]{false, true}) {
                        String objective = againstOriginals ? "P2" : "P1";

                        ConsensusUnion.metricSMHD = true;
                        ConsensusUnion.metricAgainstOriginalDAGs = againstOriginals;

                        GeneticTreeWidthUnion ga = new GeneticTreeWidthUnion(dags, seed);
                        ga.populationSize = POP_SIZE;
                        ga.numIterations  = N_ITERATIONS;
                        ga.maxTreewidth   = t;
                        ga.candidatesFromInitialDAGs = true;    // E_c
                        ga.repeatCandidates = false;
                        ga.useGreedyWarmstart = true;

                        double start = System.currentTimeMillis();
                        Dag result = ga.fusionUnion();
                        double time = (System.currentTimeMillis() - start) / 1000.0;

                        // The greedy the run was warm-started from is recorded too, so
                        // that the guarantee "a warm-started GA never returns worse than
                        // its greedy" can be checked from the CSV rather than assumed.
                        Dag greedy = ga.greedyDag;

                        for (Dag[] pair : new Dag[][]{{result, null}, {greedy, null}}) {
                            Dag g = pair[0];
                            if (g == null) continue;
                            boolean isGA = (g == result);
                            bw.write(String.format(Locale.US,
                                    "%s,%d,%d,%d,%.1f,%d,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%.2f%n",
                                    net, seed, r, k, kFrac, t, twGold, twUnion, union.getNumEdges(),
                                    unionSmhd, unionShd, objective, isGA ? "geneticEc" : "greedyEc",
                                    Utils.SMHD(g, goldDag),
                                    Utils.SHDundir(GraphTransforms.dagToCpdag(new EdgeListGraph(g)), cpdagGold),
                                    Utils.getTreeWidth(g), g.getNumEdges(),
                                    isGA ? time : ga.executionTimeGreedy));
                        }
                        bw.flush();
                    }
                }
            }
        }
        System.out.println("Written: " + out);
    }

    /**
     * A DAG over the given nodes, drawn independently of the gold structure.
     * A random permutation fixes a topological order, so acyclicity holds by
     * construction, and edges are sampled uniformly among the forward pairs
     * until the target edge count is reached.
     */
    static Dag randomDenseDag(List<Node> nodes, int targetEdges, Random rnd) {
        List<Node> order = new ArrayList<>(nodes);
        Collections.shuffle(order, rnd);

        Dag dag = new Dag(nodes);
        int n = order.size();
        int maxPairs = n * (n - 1) / 2;
        int target = Math.min(targetEdges, maxPairs);

        int added = 0, guard = 0;
        while (added < target && guard++ < 100 * target) {
            int i = rnd.nextInt(n), j = rnd.nextInt(n);
            if (i == j) continue;
            if (i > j) { int tmp = i; i = j; j = tmp; }
            Node from = order.get(i), to = order.get(j);
            if (dag.isAdjacentTo(from, to)) continue;
            dag.addDirectedEdge(from, to);
            added++;
        }
        return dag;
    }
}
