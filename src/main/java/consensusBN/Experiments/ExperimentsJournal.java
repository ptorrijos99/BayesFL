package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphTransforms;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.net.InetAddress;
import java.util.*;

import static org.albacete.simd.utils.Utils.*;

public class ExperimentsJournal {

    public static String PATH = "./";
    public static boolean verbose = false;

    // -------------------------------------------------------------------------
    // Inner classes
    // -------------------------------------------------------------------------

    /** Holds a named GeneticTreeWidthUnion config (instantiated once per experiment). */
    static class GeneticConfig {
        final String greedyName;
        final String geneticName;
        final GeneticTreeWidthUnion obj;

        GeneticConfig(String greedyName, String geneticName, GeneticTreeWidthUnion obj) {
            this.greedyName = greedyName;
            this.geneticName = geneticName;
            this.obj = obj;
        }
    }

    /** One algorithm result for a given treewidth. */
    static class AlgorithmResult {
        final String name;
        final Dag dag;
        final double time;

        AlgorithmResult(String name, Dag dag, double time) {
            this.name = name;
            this.dag = dag;
            this.time = time;
        }
    }

    /**
     * Pre-computed greedy snapshots for Eb (with repetition) and Ec (without repetition).
     * Key = treewidth limit. Built once before the TW loop to avoid recomputing per iteration.
     */
    static class GreedyCache {
        final Map<Integer, List<Dag>> ebDags  = new HashMap<>();
        final Map<Integer, Double>    ebTimes = new HashMap<>();
        final Map<Integer, List<Dag>> ecDags  = new HashMap<>();
        final Map<Integer, Double>    ecTimes = new HashMap<>();
    }

    // -------------------------------------------------------------------------
    // Genetic configs
    // -------------------------------------------------------------------------

    /**
     * Instantiates all 4 GeneticTreeWidthUnion variants once.
     * maxTreewidth is set per iteration in the treewidth loop.
     */
    static List<GeneticConfig> buildGeneticConfigs(List<Dag> dags, int seed,
                                                    int popSize, int nIterations,
                                                    boolean useGreedyWarmstart) {
        List<GeneticConfig> configs = new ArrayList<>();

        // E_a: candidates NOT from input DAGs (CEC baseline)
        GeneticTreeWidthUnion ea = new GeneticTreeWidthUnion(dags, seed);
        ea.populationSize = popSize;
        ea.numIterations  = nIterations;
        ea.candidatesFromInitialDAGs = false;
        ea.repeatCandidates = false;
        ea.useGreedyWarmstart = useGreedyWarmstart;
        configs.add(new GeneticConfig("greedyEa", "geneticEa", ea));

        // E_b: candidates from input DAGs, with repetition (GECCO)
        GeneticTreeWidthUnion eb = new GeneticTreeWidthUnion(dags, seed);
        eb.populationSize = popSize;
        eb.numIterations  = nIterations;
        eb.candidatesFromInitialDAGs = true;
        eb.repeatCandidates = true;
        eb.useGreedyWarmstart = useGreedyWarmstart;
        configs.add(new GeneticConfig("greedyEb", "geneticEb", eb));

        // E_c: candidates from input DAGs, without repetition (GECCO)
        GeneticTreeWidthUnion ec = new GeneticTreeWidthUnion(dags, seed);
        ec.populationSize = popSize;
        ec.numIterations  = nIterations;
        ec.candidatesFromInitialDAGs = true;
        ec.repeatCandidates = false;
        ec.useGreedyWarmstart = useGreedyWarmstart;
        configs.add(new GeneticConfig("greedyEc", "geneticEc", ec));

        // E_b + MinCut BES (AAAI greedy as BES phase)
        GeneticTreeWidthUnion ebMC = new GeneticTreeWidthUnion(dags, seed);
        ebMC.populationSize = popSize;
        ebMC.numIterations  = nIterations;
        ebMC.candidatesFromInitialDAGs = true;
        ebMC.repeatCandidates = true;
        ebMC.useMinCutBES = true;
        ebMC.useGreedyWarmstart = useGreedyWarmstart;
        configs.add(new GeneticConfig("greedyEbMinCut", "geneticEbMinCut", ebMC));

        return configs;
    }

    /**
     * Pre-computes originalDAGsGreedyTreewidthBefore (Eb) and WoRepeat (Ec) for every
     * treewidth value from 1 to maxTw, timing each call individually.
     * This replaces O(2 × nTwIterations) expensive calls during the TW loop with a single
     * upfront pass (same total work minus the redundant TW-1 call per iteration).
     */
    static GreedyCache buildGreedyCache(List<Dag> dags, List<Node> alpha, int maxTw) {
        GreedyCache cache = new GreedyCache();
        if (verbose) System.out.println("Pre-computing greedy caches for TW 1.." + maxTw + "...");
        for (int tw = 1; tw <= maxTw; tw++) {
            double t;

            t = System.currentTimeMillis();
            cache.ebDags.put(tw, ConsensusUnion.originalDAGsGreedyTreewidthBefore(dags, alpha, "" + tw));
            cache.ebTimes.put(tw, (System.currentTimeMillis() - t) / 1000);

            t = System.currentTimeMillis();
            cache.ecDags.put(tw, ConsensusUnion.originalDAGsGreedyTreewidthBeforeWoRepeat(dags, alpha, "" + tw));
            cache.ecTimes.put(tw, (System.currentTimeMillis() - t) / 1000);
        }
        if (verbose) System.out.println("Greedy caches done.");
        return cache;
    }

    // -------------------------------------------------------------------------
    // Node benchmark helpers
    // -------------------------------------------------------------------------

    /** Returns the SLURM node name (SLURMD_NODENAME env var), falling back to the short hostname. */
    static String getNodeName() {
        String slurm = System.getenv("SLURMD_NODENAME");
        if (slurm != null && !slurm.isEmpty()) return slurm;
        try {
            return InetAddress.getLocalHost().getHostName().split("\\.")[0];
        } catch (Exception e) {
            return "unknown";
        }
    }

    /**
     * Reads the benchmark factor (median matrix-multiply time in seconds) from
     * results/calibration/<nodeName>.txt.  Returns 1.0 if the file is missing,
     * so raw times are preserved unchanged on uncalibrated nodes.
     */
    static double loadBenchmarkFactor(String nodeName) {
        File f = new File(PATH + "results/calibration/" + nodeName + ".txt");
        if (!f.exists()) {
            System.out.println("Warning: no calibration file for node '" + nodeName + "'. Times will not be normalised.");
            return 1.0;
        }
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            return Double.parseDouble(br.readLine().trim());
        } catch (Exception e) {
            System.out.println("Warning: could not read calibration file for node '" + nodeName + "': " + e);
            return 1.0;
        }
    }

    // -------------------------------------------------------------------------
    // main()
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        String net;
        int nClients, popSize, nIterations, seed;
        double twLimit;
        boolean useGES, optimizeSMHD, optimizeAgainstOriginals, useGreedyWarmstart;

        if (args.length < 2) {
            // Local test run
            net = "alarm.0";
            verbose = true;
            nClients = 10;
            popSize = 100;
            nIterations = 100;
            twLimit = 2;
            seed = 0;
            useGES = false;
            optimizeSMHD = true;              // SMHD (not FSim)
            optimizeAgainstOriginals = true;  // vs input DAGs (not vs G+)
            useGreedyWarmstart = true;
        } else {
            int index = Integer.parseInt(args[0]);
            String paramsFile = args[1];
            String[] params = null;
            try (BufferedReader br = new BufferedReader(new FileReader(paramsFile))) {
                for (int i = 0; i < index; i++) br.readLine();
                params = br.readLine().split(" ");
            } catch (Exception e) { System.out.println(e); }

            net                      = params[0];
            nClients                 = Integer.parseInt(params[1]);
            popSize                  = Integer.parseInt(params[2]);
            nIterations              = Integer.parseInt(params[3]);
            twLimit                  = Double.parseDouble(params[4]);
            seed                     = Integer.parseInt(params[5]);
            useGES                   = params.length > 6 && Boolean.parseBoolean(params[6]);
            // params[7]: optimizeSMHD             (default true  = SMHD, false = FSim)
            // params[8]: optimizeAgainstOriginals  (default true  = vs input DAGs, false = vs G+)
            // params[9]: useGreedyWarmstart        (default true  = warmstart from greedy, false = ablation)
            optimizeSMHD             = params.length <= 7 || Boolean.parseBoolean(params[7]);
            optimizeAgainstOriginals = params.length <= 8 || Boolean.parseBoolean(params[8]);
            useGreedyWarmstart       = params.length <= 9 || Boolean.parseBoolean(params[9]);
        }

        String gesTag      = useGES ? "GES" : "";
        String metricTag   = optimizeAgainstOriginals ? "" : "vsUnion";
        String warmstartTag = useGreedyWarmstart ? "" : "noWS";
        String tag = "_Journal" + gesTag + metricTag + warmstartTag + "_";
        String savePath = "./results/Server/" + net + tag
                + nClients + "_" + popSize + "_" + nIterations
                + "_" + seed + "_" + twLimit + ".csv";

        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, useGES,
                         optimizeSMHD, optimizeAgainstOriginals, useGreedyWarmstart, savePath);
    }

    /**
     * @param optimizeSMHD              true = SMHD fitness; false = fusionSimilarity
     * @param optimizeAgainstOriginals  true = measure against input DAGs; false = against G+
     * @param useGreedyWarmstart        true = greedy injected into population[0/1]; false = ablation (all random init)
     */
    public static void launchExperiment(String net, int nDags, int popSize,
                                        int nIterations, double twLimit,
                                        int seed, boolean useGES,
                                        boolean optimizeSMHD, boolean optimizeAgainstOriginals,
                                        boolean useGreedyWarmstart,
                                        String savePath) {
        // Early exit: check CSV before running any expensive computation
        if (isAlreadyComplete(savePath)) {
            System.out.println("Already complete (skipping): " + savePath);
            return;
        }

        ConsensusUnion.metricSMHD = optimizeSMHD;
        ConsensusUnion.metricAgainstOriginalDAGs = optimizeAgainstOriginals;
        new File("./results/Server/").mkdirs();

        String nodeName       = getNodeName();
        double benchmarkFactor = loadBenchmarkFactor(nodeName);
        if (verbose) System.out.println("Node: " + nodeName + "  benchmarkFactor: " + benchmarkFactor);

        boolean realNetwork = net.contains(".");
        int originalTw = -1;
        Dag goldDag = null;
        DataSet seedData = null;

        // --- Load gold standard (real networks) and input DAGs ---
        List<Dag> dags;

        if (realNetwork) {
            String netName = net.split("\\.")[0];
            seedData = readData(PATH + "res/networks/BBDD/" + netName + "/" + net + ".csv");
            BIFReader reader = new BIFReader();
            try { reader.processFile(PATH + "res/networks/" + netName + ".xbif"); }
            catch (Exception e) { throw new RuntimeException(e); }

            // Gold standard from the xbif (always needed regardless of DAG source)
            RandomBN rbnGold = new RandomBN(reader, seedData, seed, 1, twLimit);
            originalTw = getTreeWidth(rbnGold.originalBayesIm.getDag());
            goldDag    = new Dag(rbnGold.originalBayesIm.getBayesPm().getDag());

            if (useGES) {
                dags = loadGESDags(net, netName, nDags);
            } else {
                RandomBN randomBN = new RandomBN(reader, seedData, seed, nDags, twLimit);
                randomBN.generate();
                dags = randomBN.setOfRandomDags;
            }
        } else {
            RandomBN randomBN = new RandomBN(seed, Integer.parseInt(net), nDags);
            randomBN.generate();
            dags = randomBN.setOfRandomDags;
        }

        // --- Treewidth stats of input DAGs ---
        int maxTW = 0, minTW = Integer.MAX_VALUE;
        double meanTW = 0;
        for (Dag d : dags) {
            int tw = getTreeWidth(d);
            maxTW  = Math.max(maxTW, tw);
            minTW  = Math.min(minTW, tw);
            meanTW += tw;
        }
        meanTW /= dags.size();

        // --- Parent stats of input DAGs ---
        double meanParents = 0;
        int    maxParents  = 0;
        for (Dag d : dags) {
            meanParents += Experiments.meanParents(d);
            maxParents   = Math.max(maxParents, Experiments.maxParents(d));
        }
        meanParents /= dags.size();

        // --- Moralized input DAGs (for SMHD) ---
        List<Graph> moralizedDags = new ArrayList<>();
        for (Dag d : dags) moralizedDags.add(Utils.moralize(d));

        // --- CPDAG of input DAGs (for CPDAG-SHD) ---
        List<Graph> cpdagDags = new ArrayList<>();
        for (Dag d : dags) cpdagDags.add(GraphTransforms.dagToCpdag(new EdgeListGraph(d)));

        // --- Build genetic configs once (outside the treewidth loop) ---
        List<GeneticConfig> geneticConfigs = buildGeneticConfigs(dags, seed, popSize, nIterations, useGreedyWarmstart);

        // --- G+: reuse fusionUnion field from first genetic config ---
        Dag   fusionUnion      = geneticConfigs.get(0).obj.fusionUnion;
        int   unionTw          = getTreeWidth(fusionUnion);
        Graph moralFusionUnion = Utils.moralize(fusionUnion);
        Graph cpdagFusionUnion = GraphTransforms.dagToCpdag(new EdgeListGraph(fusionUnion));

        if (verbose) System.out.println("G+ treewidth: " + unionTw);

        // --- Pre-compute CPDAG and BDeu for gold standard and reference structures (real networks only) ---
        Graph  cpdagGold     = null;
        double goldBDeu      = -1;
        double unionBDeu     = -1;
        double meanInputBDeu = -1;
        if (realNetwork) {
            cpdagGold = GraphTransforms.dagToCpdag(new EdgeListGraph(goldDag));
            goldBDeu  = Experiments.getBDeuScore(goldDag, seedData);
            unionBDeu = Experiments.getBDeuScore(fusionUnion, seedData);
            double bdeuSum = 0;
            for (Dag d : dags) bdeuSum += Experiments.getBDeuScore(d, seedData);
            meanInputBDeu = bdeuSum / dags.size();
            if (verbose) System.out.println("goldBDeu=" + goldBDeu + "  unionBDeu=" + unionBDeu + "  meanInputBDeu=" + meanInputBDeu);
        }

        // --- Pre-run MinCut ONCE, capturing snapshots at every treewidth ---
        // Constructor deep-copies dags, so originals are safe.
        MinCutTreeWidthUnion preRunMC = new MinCutTreeWidthUnion(new ArrayList<>(dags), 10, 1);
        preRunMC.experiments_tw = true;
        if (verbose) System.out.println("Running MinCut pre-run...");
        preRunMC.fusion();
        if (verbose) System.out.println("MinCut pre-run done: " + preRunMC.outputExperimentDAGs.size() + " snapshots.");

        // --- Pre-compute Eb/Ec greedy caches for all TW values ---
        List<Node> alpha = geneticConfigs.get(0).obj.getAlpha();
        GreedyCache greedyCache = buildGreedyCache(dags, alpha, unionTw - 1);

        // --- Resume support ---
        int startTw = resumeTw(savePath);
        if (unionTw <= startTw) {
            System.out.println("Already completed up to unionTw=" + unionTw);
            return;
        }

        // --- Treewidth loop ---
        for (int tw = startTw; tw < unionTw; tw++) {
            if (verbose) System.out.println("\n--- Treewidth: " + tw + " ---");

            List<AlgorithmResult> results = runAlgorithms(
                    geneticConfigs, dags, tw, moralFusionUnion, moralizedDags, preRunMC, greedyCache);

            saveRound(net, nDags, popSize, nIterations, twLimit, seed, useGES, useGreedyWarmstart,
                      optimizeSMHD, optimizeAgainstOriginals,
                      originalTw, unionTw, minTW, meanTW, maxTW, tw,
                      meanParents, maxParents,
                      fusionUnion, moralFusionUnion, cpdagFusionUnion,
                      dags, moralizedDags, cpdagDags,
                      goldDag, cpdagGold,
                      unionBDeu, goldBDeu, meanInputBDeu, seedData,
                      results, nodeName, benchmarkFactor, savePath);

            saveConvergence(geneticConfigs, tw, savePath,
                            net, nDags, popSize, nIterations, seed, useGES,
                            optimizeSMHD, optimizeAgainstOriginals, useGreedyWarmstart);
        }
    }

    // -------------------------------------------------------------------------
    // Convergence CSV output
    // -------------------------------------------------------------------------

    /**
     * Appends one row per (algo, iteration) for the given treewidth to a separate convergence CSV.
     * Format: bbdd,nDags,popSize,nIterations,seed,useGES,optimizeSMHD,optimizeAgainstOriginals,useGreedyWarmstart,limitTW,algo,iter,bestFitness
     * Only genetic algorithms are included (greedy and minCut have no iteration curve).
     */
    static void saveConvergence(List<GeneticConfig> geneticConfigs, int tw, String savePath,
                                 String net, int nDags, int popSize, int nIterations,
                                 int seed, boolean useGES, boolean optimizeSMHD,
                                 boolean optimizeAgainstOriginals, boolean useGreedyWarmstart) {
        // Place convergence files in a dedicated subfolder to keep them separate from main CSVs
        File mainFile = new File(savePath);
        String convergenceDir = mainFile.getParent() + "/convergence/";
        new File(convergenceDir).mkdirs();
        String convergencePath = convergenceDir + mainFile.getName().replace(".csv", "_convergence.csv");
        File file = new File(convergencePath);
        String prefix = net + "," + nDags + "," + popSize + "," + nIterations + ","
                      + seed + "," + useGES + "," + optimizeSMHD + "," + optimizeAgainstOriginals + ","
                      + useGreedyWarmstart + "," + tw + ",";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, true))) {
            if (file.length() == 0) {
                bw.write("bbdd,nDags,popSize,nIterations,seed,useGES,optimizeSMHD,optimizeAgainstOriginals,useGreedyWarmstart,limitTW,algo,iter,bestFitness\n");
            }
            for (GeneticConfig cfg : geneticConfigs) {
                List<Double> curve = cfg.obj.convergenceCurve;
                for (int iter = 0; iter < curve.size(); iter++) {
                    bw.write(prefix + cfg.geneticName + "," + iter + "," + curve.get(iter) + "\n");
                }
            }
            bw.flush();
        } catch (IOException e) { System.out.println(e); }
    }

    // -------------------------------------------------------------------------
    // GES DAG loading
    // -------------------------------------------------------------------------

    /**
     * Loads GES-learned DAGs for each client (0..nDags-1) using the same
     * file-locking cache as ExperimentMinCutGES.
     */
    static List<Dag> loadGESDags(String net, String netName, int nDags) {
        String cachePath = PATH + "results/Cache/";
        new File(cachePath).mkdirs();
        File cacheFile = new File(cachePath + "cachedDAGs-" + net + ".ser");
        Map<Integer, Dag> cachedDags = ExperimentMinCutGES.readCache(cacheFile);

        List<Dag> dags = new ArrayList<>();
        for (int i = 0; i < nDags; i++) {
            if (cachedDags.containsKey(i)) {
                Dag d = cachedDags.get(i);
                dags.add(d);
                if (verbose) System.out.println("DAG " + i + " from cache (TW=" + getTreeWidth(d) + ")");
            } else {
                DataSet dataTemp = readData(PATH + "res/networks/BBDD/" + netName + "/" + netName + "." + i + ".csv");
                BNBuilder algorithm = new GES_BNBuilder(dataTemp, true);
                Dag temp = new Dag(algorithm.search());
                cachedDags.put(i, temp);
                dags.add(temp);
                if (verbose) System.out.println("DAG " + i + " learned with GES (TW=" + getTreeWidth(temp) + ")");
            }
        }

        ExperimentMinCutGES.writeCache(cacheFile, cachedDags);
        return dags;
    }

    // -------------------------------------------------------------------------
    // runAlgorithms()
    // -------------------------------------------------------------------------

    /**
     * Runs all algorithms for a given treewidth.
     * Returns a flat list: [greedyEa, greedyEb, greedyEc,
     *                        minCut,
     *                        geneticEa, geneticEb, geneticEc, geneticEbMinCut]
     *
     * greedyEbMinCut is omitted — it IS the MinCut result.
     * geneticEbMinCut starts from the best MinCut snapshot as warmstart.
     * MinCut result: best SMHDoriginals among snapshots with actual TW ≤ tw.
     */
    static List<AlgorithmResult> runAlgorithms(List<GeneticConfig> geneticConfigs,
                                                List<Dag> dags, int tw,
                                                Graph moralFusionUnion,
                                                List<Graph> moralizedDags,
                                                MinCutTreeWidthUnion preRunMC,
                                                GreedyCache greedyCache) {
        List<AlgorithmResult> results = new ArrayList<>();

        // --- Best MinCut snapshot index for this treewidth ---
        int mcBestIdx = bestMinCutIndex(preRunMC, tw, moralizedDags);
        Dag    mcBestDag  = mcBestIdx >= 0 ? preRunMC.outputExperimentDAGs.get(mcBestIdx)  : preRunMC.outputDag;
        double mcBestTime = mcBestIdx >= 0 ? preRunMC.outputExperimentTimes.get(mcBestIdx) : 0;

        // --- Genetic configs: greedy results (skip greedyEbMinCut — same as minCut) ---
        for (GeneticConfig cfg : geneticConfigs) {
            cfg.obj.maxTreewidth = tw;
            if (cfg.obj.useMinCutBES && mcBestIdx >= 0) {
                // EbMinCut: warmstart from the best MinCut snapshot
                cfg.obj.precomputedMinCut = preRunMC;
                cfg.obj.precomputedMinCutBestIdx = mcBestIdx;
            } else if (cfg.obj.candidatesFromInitialDAGs && !cfg.obj.useMinCutBES) {
                // Eb or Ec: inject pre-computed greedy cache (avoids re-running per TW)
                boolean repeat = cfg.obj.repeatCandidates;
                cfg.obj.cachedGreedyDags   = repeat ? greedyCache.ebDags.get(tw)    : greedyCache.ecDags.get(tw);
                cfg.obj.cachedGreedyDagsM1 = repeat ? greedyCache.ebDags.get(tw -1) : greedyCache.ecDags.get(tw - 1);
                cfg.obj.cachedGreedyTime   = repeat ? greedyCache.ebTimes.get(tw)   : greedyCache.ecTimes.get(tw);
            }
            cfg.obj.fusionUnion();
            if (!cfg.obj.useMinCutBES) {
                results.add(new AlgorithmResult(cfg.greedyName, cfg.obj.greedyDag, cfg.obj.executionTimeGreedy));
            }
        }

        // --- MinCut: best snapshot with TW ≤ tw, minimum SMHDoriginals ---
        results.add(new AlgorithmResult("minCut", mcBestDag, mcBestTime));

        // --- Genetic configs: genetic results ---
        for (GeneticConfig cfg : geneticConfigs) {
            results.add(new AlgorithmResult(cfg.geneticName, cfg.obj.bestDag, cfg.obj.executionTime));
        }

        if (verbose) {
            for (AlgorithmResult r : results) {
                System.out.printf("%-20s TW=%d  edges=%d  SMHDfusion=%.1f  SMHDoriginals=%.1f  time=%.2fs%n",
                    r.name,
                    getTreeWidth(r.dag),
                    r.dag.getNumEdges(),
                    (double) Utils.SMHDwithoutMoralize(moralFusionUnion, Utils.moralize(r.dag)),
                    Utils.SMHDwithoutMoralize(Utils.moralize(r.dag), moralizedDags),
                    r.time);
            }
        }

        return results;
    }

    /**
     * Returns the index of the best MinCut snapshot: among all pre-run snapshots
     * with actual TW ≤ maxTw, the one with minimum SMHDoriginals.
     * Falls back to the last snapshot index if none satisfies the TW constraint.
     * Returns -1 only if there are no snapshots at all.
     */
    static int bestMinCutIndex(MinCutTreeWidthUnion preRunMC, int maxTw, List<Graph> moralizedDags) {
        List<Dag> snapshots = preRunMC.outputExperimentDAGs;

        int    bestIdx  = -1;
        double bestSMHD = Double.MAX_VALUE;

        for (int i = 0; i < snapshots.size(); i++) {
            if (getTreeWidth(snapshots.get(i)) <= maxTw) {
                double smhd = Utils.SMHDwithoutMoralize(Utils.moralize(snapshots.get(i)), moralizedDags);
                if (smhd < bestSMHD) {
                    bestSMHD = smhd;
                    bestIdx  = i;
                }
            }
        }

        // Fallback: last snapshot (lowest TW reached by the pre-run)
        if (bestIdx < 0 && !snapshots.isEmpty()) {
            bestIdx = snapshots.size() - 1;
        }

        return bestIdx;
    }

    // -------------------------------------------------------------------------
    // Metric helpers
    // -------------------------------------------------------------------------

    /** Mean SHDundir between cpdag1 and each element of cpdagList. */
    static double meanSHDundir(Graph cpdag1, List<Graph> cpdagList) {
        double sum = 0;
        for (Graph c : cpdagList) sum += Utils.SHDundir(cpdag1, c);
        return sum / cpdagList.size();
    }

    // -------------------------------------------------------------------------
    // CSV output
    // -------------------------------------------------------------------------

    static String generateHeader(List<AlgorithmResult> results, boolean realNetwork) {
        StringBuilder h = new StringBuilder(
            "bbdd,nDags,popSize,nIterations,maxTWGeneratedDAGs,seed,useGES,useGreedyWarmstart," +
            "optimizeSMHD,optimizeAgainstOriginals," +
            "originalTW,unionTW,minTW,meanTW,maxTW,limitTW," +
            "node,benchmarkFactor," +
            "unionEdges,unionSMHDoriginals,unionFusSimOriginals,unionSHDoriginals");

        if (realNetwork) {
            h.append(",unionSMHDgoldStandard,unionFusSimGoldStandard,unionSHDgoldStandard");
            h.append(",unionBDeu,goldBDeu,meanInputBDeu");
        }

        // Per-algo metrics (all networks)
        String[] perAlgoMetrics = {"TW", "Edges", "SMHDfusion", "SMHDoriginals", "FusSimOriginals", "Time", "TimeNorm", "SHDfusion", "SHDoriginals"};
        for (String metric : perAlgoMetrics) {
            for (AlgorithmResult r : results) {
                h.append(",").append(r.name).append(metric);
            }
        }

        if (realNetwork) {
            for (AlgorithmResult r : results) h.append(",").append(r.name).append("SMHDgoldStandard");
            for (AlgorithmResult r : results) h.append(",").append(r.name).append("FusSimGoldStandard");
            for (AlgorithmResult r : results) h.append(",").append(r.name).append("BDeu");
            for (AlgorithmResult r : results) h.append(",").append(r.name).append("SHDgoldStandard");
        }

        return h.toString();
    }

    static void saveRound(String net, int nDags, int popSize, int nIterations,
                          double twLimit, int seed, boolean useGES, boolean useGreedyWarmstart,
                          boolean optimizeSMHD, boolean optimizeAgainstOriginals,
                          int originalTw, int unionTw,
                          int minTW, double meanTW, int maxTW, int tw,
                          double meanParents, int maxParents,
                          Dag fusionUnion, Graph moralFusionUnion, Graph cpdagFusionUnion,
                          List<Dag> dags, List<Graph> moralizedDags, List<Graph> cpdagDags,
                          Dag goldDag, Graph cpdagGold,
                          double unionBDeu, double goldBDeu, double meanInputBDeu,
                          DataSet seedData,
                          List<AlgorithmResult> results,
                          String nodeName, double benchmarkFactor,
                          String savePath) {

        boolean realNetwork = goldDag != null;

        // --- Fixed columns ---
        StringBuilder line = new StringBuilder();
        line.append(net).append(",")
            .append(nDags).append(",")
            .append(popSize).append(",")
            .append(nIterations).append(",")
            .append(twLimit).append(",")
            .append(seed).append(",")
            .append(useGES).append(",")
            .append(useGreedyWarmstart).append(",")
            .append(optimizeSMHD).append(",")
            .append(optimizeAgainstOriginals).append(",")
            .append(originalTw).append(",")
            .append(unionTw).append(",")
            .append(minTW).append(",")
            .append(meanTW).append(",")
            .append(maxTW).append(",")
            .append(tw).append(",")
            .append(nodeName).append(",")
            .append(benchmarkFactor).append(",")
            .append(fusionUnion.getNumEdges()).append(",")
            .append(Utils.SMHDwithoutMoralize(moralFusionUnion, moralizedDags)).append(",")
            .append(Utils.fusionSimilarity(fusionUnion, dags)).append(",")
            .append(meanSHDundir(cpdagFusionUnion, cpdagDags));

        if (realNetwork) {
            line.append(",").append(Utils.SMHD(fusionUnion, goldDag));
            line.append(",").append(Utils.fusionSimilarity(fusionUnion, List.of(goldDag)));
            line.append(",").append(Utils.SHDundir(cpdagFusionUnion, cpdagGold));
            line.append(",").append(unionBDeu);
            line.append(",").append(goldBDeu);
            line.append(",").append(meanInputBDeu);
        }

        // --- Per-algorithm metrics (all networks) ---
        for (AlgorithmResult r : results)
            line.append(",").append(getTreeWidth(r.dag));
        for (AlgorithmResult r : results)
            line.append(",").append(r.dag.getNumEdges());
        for (AlgorithmResult r : results) {
            Graph moral = Utils.moralize(r.dag);
            line.append(",").append(Utils.SMHDwithoutMoralize(moralFusionUnion, moral));
        }
        for (AlgorithmResult r : results) {
            Graph moral = Utils.moralize(r.dag);
            line.append(",").append(Utils.SMHDwithoutMoralize(moral, moralizedDags));
        }
        for (AlgorithmResult r : results)
            line.append(",").append(Utils.fusionSimilarity(r.dag, dags));
        for (AlgorithmResult r : results)
            line.append(",").append(r.time);
        for (AlgorithmResult r : results)
            line.append(",").append(r.time / benchmarkFactor);
        // CPDAG-SHD (all networks)
        for (AlgorithmResult r : results) {
            Graph cpdagAlgo = GraphTransforms.dagToCpdag(new EdgeListGraph(r.dag));
            line.append(",").append(Utils.SHDundir(cpdagAlgo, cpdagFusionUnion));
        }
        for (AlgorithmResult r : results) {
            Graph cpdagAlgo = GraphTransforms.dagToCpdag(new EdgeListGraph(r.dag));
            line.append(",").append(meanSHDundir(cpdagAlgo, cpdagDags));
        }

        // --- Gold standard metrics (real networks only) ---
        if (realNetwork) {
            for (AlgorithmResult r : results)
                line.append(",").append(Utils.SMHD(r.dag, goldDag));
            for (AlgorithmResult r : results)
                line.append(",").append(Utils.fusionSimilarity(r.dag, List.of(goldDag)));
            for (AlgorithmResult r : results)
                line.append(",").append(Experiments.getBDeuScore(r.dag, seedData));
            for (AlgorithmResult r : results) {
                Graph cpdagAlgo = GraphTransforms.dagToCpdag(new EdgeListGraph(r.dag));
                line.append(",").append(Utils.SHDundir(cpdagAlgo, cpdagGold));
            }
        }

        line.append("\n");

        // --- Write to CSV ---
        File file = new File(savePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file, true))) {
            if (file.length() == 0) {
                bw.write(generateHeader(results, realNetwork));
                bw.write("\n");
            }
            bw.write(line.toString());
            bw.flush();
        } catch (IOException e) { System.out.println(e); }
    }

    // -------------------------------------------------------------------------
    // Resume support
    // -------------------------------------------------------------------------

    /**
     * Returns true if the CSV already contains results up to the final treewidth
     * (limitTW + 1 >= unionTW), meaning nothing more needs to be computed.
     * Reads only the last line of the CSV — no algorithms are run.
     */
    static boolean isAlreadyComplete(String savePath) {
        File f = new File(savePath);
        if (!f.exists()) return false;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String last = null, line;
            while ((line = br.readLine()) != null) last = line;
            if (last == null || last.startsWith("bbdd")) return false;
            String[] cols = last.split(",");
            // Col 15 = limitTW, col 11 = unionTW (after adding optimizeSMHD/optimizeAgainstOriginals at cols 8-9)
            int limitTW = Integer.parseInt(cols[15]);
            int unionTW = Integer.parseInt(cols[11]);
            return limitTW + 1 >= unionTW;
        } catch (Exception e) { return false; }
    }

    /** Reads last saved treewidth from CSV to support resuming interrupted runs. */
    static int resumeTw(String savePath) {
        int tw = 2;
        File f = new File(savePath);
        if (!f.exists()) return tw;
        try (BufferedReader br = new BufferedReader(new FileReader(f))) {
            String last = null, line;
            while ((line = br.readLine()) != null) last = line;
            if (last != null && !last.startsWith("bbdd")) {
                // Col 15 (0-indexed) is limitTW in ExperimentsJournal CSV
                // (bbdd,nDags,popSize,nIterations,maxTWGeneratedDAGs,seed,useGES,useGreedyWarmstart,optimizeSMHD,optimizeAgainstOriginals,originalTW,unionTW,minTW,meanTW,maxTW,limitTW,...)
                tw = Integer.parseInt(last.split(",")[15]) + 1;
            }
        } catch (Exception ignored) {}
        return tw;
    }
}
