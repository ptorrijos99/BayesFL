package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import consensusBN.Method.InitialDAGs_Method;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.EmBayesEstimator;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.albacete.simd.utils.Utils.getTreeWidth;
import static org.albacete.simd.utils.Utils.readData;


public class ExperimentMinCutGenetic {

    public static String PATH = "./";
    public static boolean verbose = false;

    // TODO: UNCOMMENT THIS LINES
    /*public static void main(String[] args) {
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];

        // Read the parameters from args
        String[] parameters = null;
        try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
            String line;
            for (int i = 0; i < index; i++)
                br.readLine();
            line = br.readLine();
            parameters = line.split(" ");
        }
        catch(Exception e){ System.out.println(e); }
        //String[] parameters = new String[]{"andes", "0", "10", "1000", "1000", "1"};

        System.out.println("Number of hyperparams: " + parameters.length);
        int i=0;
        for (String string : parameters) {
            System.out.println("Param[" + i + "]: " + string);
            i++;
        }

        // Read the parameters from file
        String net = parameters[0];
        int nClients = Integer.parseInt(parameters[1]);
        int popSize = Integer.parseInt(parameters[2]);
        int nIterations = Integer.parseInt(parameters[3]);
        double twLimit = Double.parseDouble(parameters[4]);
        int seed = Integer.parseInt(parameters[5]);

        boolean againstOriginalDAGs = Boolean.parseBoolean(parameters[6]);
        boolean mectricIsSMHD = Boolean.parseBoolean(parameters[7]);

        ConsensusUnion.metricAgainstOriginalDAGs = againstOriginalDAGs;
        ConsensusUnion.metricSMHD = mectricIsSMHD;

        String savePath = "./results/Server/" + net + "_MinCutTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + "_" + againstOriginalDAGs + "_" + mectricIsSMHD + ".csv";

        // Launch the experiment
        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, savePath);
    }*/

    public static void main(String[] args) {
        // Real network (net = net.bbdd)
        String net = "hailfinder.0";

        // Generic network (net = number of nodes)
        //String net = ""+10;

        verbose = true;

        int nClients = 20;
        int popSize = 100;
        int nIterations = 100;
        double twLimit = 2;
        int seed = 1;

        boolean againstOriginalDAGs = true;
        boolean mectricIsSMHD = true;

        ConsensusUnion.metricAgainstOriginalDAGs = againstOriginalDAGs;
        ConsensusUnion.metricSMHD = mectricIsSMHD;

        String savePath = "./results/Server/" + net + "_MinCutTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + "_" + againstOriginalDAGs + "_" + mectricIsSMHD + ".csv";

        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, savePath);
    }

    public static void launchExperiment(String net, int nDags, int popSize, int nIterations, double twLimit, int seed, String savePath) {
        // Check if the folder (and subfolders) exists
        if (!new File("./results/Server/").exists()) {
            new File("./results/Server/").mkdirs();
        }

        // Check if the file exists, and if it does, get the maximum treewidth found
        int tw = 2;
        if (new File(savePath).exists()) {
            System.out.println("File exists: " + savePath);
            // Check the maximum executed treewidth in the file
            BufferedReader br = null;
            try {
                br = new BufferedReader(new FileReader(savePath));
            } catch (FileNotFoundException ignored) {}

            // Read last line
            String lastLine = null;
            try {
                String line;
                while ((line = br.readLine()) != null) {
                    lastLine = line;
                }
            } catch (IOException ignored) {}

            // Get the maximum treewidth
            tw = Integer.parseInt(lastLine.split(",")[13]);
            System.out.println("Maximum treewidth found in file: " + tw);
            tw += 1;

            // Close the file
            try {
                br.close();
            } catch (IOException ignored) {}
        }

        RandomBN randomBN;
        int originalTw = -1;
        boolean realNetwork = net.contains(".");
        // Real network
        if (realNetwork) {
            // Read the .csv
            DataSet data = readData(PATH + "res/networks/BBDD/" + net + ".csv");

            // Read the .xbif
            String netName = net.split("\\.")[0];
            BIFReader bayesianReader = new BIFReader();
            try {
                bayesianReader.processFile(PATH + "res/networks/" + netName + ".xbif");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // Generate the DAGs
            randomBN = new RandomBN(bayesianReader, data, seed, nDags, twLimit);

            // Calculate the treewidth of the original DAG
            originalTw = getTreeWidth(randomBN.originalBayesIm.getDag());
        }
        // Synthetic network
        else {
            // Generate the DAGs
            int numNodes = Integer.parseInt(net);
            randomBN = new RandomBN(seed, numNodes, nDags);
        }
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;


        GeneticTreeWidthUnion geneticUnionPuertaBES = new GeneticTreeWidthUnion(dags, seed);
        geneticUnionPuertaBES.populationSize = popSize;
        geneticUnionPuertaBES.candidatesFromInitialDAGs = true;
        geneticUnionPuertaBES.repeatCandidates = true;
        geneticUnionPuertaBES.numIterations = nIterations;
        geneticUnionPuertaBES.useMinCutBES = true;

        // Find the treewidth of the dags sampled
        int maxTW = 0;
        double meanTW = 0;
        int minTW = Integer.MAX_VALUE;
        for (Dag dag : dags) {
            int temp = getTreeWidth(dag);
            if (temp > maxTW) maxTW = temp;
            if (temp < minTW) minTW = temp;
            meanTW += temp;
        }
        meanTW /= dags.size();

        // Find the mean and max parents of the dags sampled
        double meanParents = 0;
        int maxParents = 0;
        for (Dag dag : dags) {
            meanParents += meanParents(dag);
            int temp = maxParents(dag);
            if (temp > maxParents) maxParents = temp;
        }
        meanParents /= dags.size();

        Dag unionDag = geneticUnionPuertaBES.fusionUnion;

        // Find the treewidth of the union of the dags
        int unionTw = getTreeWidth(unionDag);
        System.out.println("Fusion Union Treewidth: " + unionTw);


        // Save the moralized original DAGs into a new list
        ArrayList<Graph> moralizedDags = new ArrayList<>();
        for (Dag dag : dags) {
            moralizedDags.add(Utils.moralize(dag));
        }

        double[][] originalBNrecalcMarginals = null;
        double[][] unionBNMarginals = null;
        ArrayList<double[][]> sampledBNsMarginals = null;
        double timeRecalc = -1;
        double timeSampled = -1;
        double timeUnion = -1;
        if (realNetwork) {
            BayesIm originalBN = randomBN.originalBayesIm;
            sampledBNsMarginals = new ArrayList<>();
            // Recalculate probabilities of the original BN given the data
            try {
                double start = System.currentTimeMillis();
                BayesPm bayesPm = new BayesPm(originalBN.getBayesPm());
                for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                    bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                    bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
                }
                BayesIm originalBNrecalc = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
                timeRecalc = (System.currentTimeMillis() - start) / 1000.0;
                originalBNrecalcMarginals = Experiments.marginals(originalBNrecalc, randomBN.categories, randomBN.nodesDags);
            } catch (OutOfMemoryError | Exception ex) {
                System.gc();
                //Log the info
                System.err.println("REAL RECALCULATED GRAPH: Array size too large: " + ex.getClass());
            }

            // Get the BayesIm of the sampled graphs
            for (Dag sampledDag : randomBN.setOfRandomDags) {
                try {
                    double start = System.currentTimeMillis();
                    BayesPm bayesPm = new BayesPm(sampledDag);
                    for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                        bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                        bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
                    }
                    BayesIm sampledBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
                    sampledBNsMarginals.add(Experiments.marginals(sampledBN, randomBN.categories, randomBN.nodesDags));
                    if (timeSampled == -1) timeSampled = (System.currentTimeMillis() - start) / 1000.0;
                    else timeSampled += (System.currentTimeMillis() - start) / 1000.0;
                } catch (OutOfMemoryError | Exception ex) {
                    System.gc();
                    //Log the info
                    System.err.println("REAL RECALCULATED GRAPH: Array size too large: " + ex.getClass());
                }
            }
            timeSampled /= randomBN.setOfRandomDags.size();

            // Get the BayesIm of the union graph
            try {
                double start = System.currentTimeMillis();
                BayesPm bayesPm = new BayesPm(geneticUnionPuertaBES.fusionUnion);
                for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                    bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                    bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
                }
                BayesIm unionBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
                timeUnion = (System.currentTimeMillis() - start) / 1000.0;
                unionBNMarginals = Experiments.marginals(unionBN, randomBN.categories, randomBN.nodesDags);
            } catch (OutOfMemoryError | Exception ex) {
                System.gc();
                //Log the info
                System.err.println("UNION GRAPH: Array size too large: " + ex.getClass());
            }
        }

        // Execute the genetic union for each treewidth from the last executed to the limit
        for (; tw < unionTw; tw++) {
            System.out.println("\nTreewidth: " + tw);
            geneticUnionPuertaBES.maxTreewidth = tw;
            geneticUnionPuertaBES.fusionUnion();

            Dag geneticDag = geneticUnionPuertaBES.bestDag;
            double geneticTime = geneticUnionPuertaBES.executionTime;

            Dag greedyDag = geneticUnionPuertaBES.greedyDag;
            double greedyTime = geneticUnionPuertaBES.executionTimeGreedy;

            List<String> algorithms = List.of("greedyMinCut", "geneticMinCut");
            ExperimentData experimentData;
            if (realNetwork) {
                experimentData = new ExperimentData(true, algorithms, meanParents, maxParents, twLimit, geneticUnionPuertaBES.fusionUnion, Utils.moralize(geneticUnionPuertaBES.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw, timeRecalc, timeSampled, timeUnion, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals);
            } else {
                experimentData = new ExperimentData(false, algorithms, meanParents, maxParents, twLimit, geneticUnionPuertaBES.fusionUnion, Utils.moralize(geneticUnionPuertaBES.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw);
            }

            if (verbose) System.out.println("Greedy minCut SMHD: \t\t" + Utils.SMHD(geneticUnionPuertaBES.fusionUnion, greedyDag) + " | SMHD ORIG: " + Utils.SMHD(greedyDag, dags) + " | FusSim ORIG: " + Utils.fusionSimilarity(greedyDag, dags) + " | Edges: " + greedyDag.getNumEdges() + " | Time: " + greedyTime + " | TW: " + getTreeWidth(greedyDag));
            if (verbose) System.out.println("Genetic minCut SMHD: \t\t" + Utils.SMHD(geneticUnionPuertaBES.fusionUnion, geneticDag) + " | SMHD ORIG: " + Utils.SMHD(geneticDag, dags) + " | FusSim ORIG: " + Utils.fusionSimilarity(geneticDag, dags) + " | Edges: " + geneticDag.getNumEdges() + " | Time: " + geneticTime + " | TW: " + getTreeWidth(geneticDag));

            List<AlgorithmResults> algorithmResultsList = new ArrayList<>(Arrays.asList(
                    new AlgorithmResults(greedyDag, greedyTime),
                    new AlgorithmResults(geneticDag, geneticTime)
            ));

            saveRound(experimentData, algorithmResultsList, randomBN, savePath);
        }
    }

    public static void saveRound(ExperimentData experimentData, List<AlgorithmResults> algorithmResultsList, RandomBN randomBN, String savePath) {
        String header = generateDynamicHeader(experimentData.algorithms);
        String headerProbs = generateDynamicHeaderProbs(experimentData.algorithms);

        // Crear la primera parte de la línea con los datos fijos
        StringBuilder lineBuilder = new StringBuilder();
        lineBuilder.append(experimentData.bbdd).append(",")
                .append(experimentData.nDags).append(",")
                .append(experimentData.popSize).append(",")
                .append(experimentData.nIterations).append(",")
                .append(experimentData.maxTWGeneratedDAGs).append(",")
                .append(experimentData.seed).append(",")
                .append(ConsensusUnion.metricAgainstOriginalDAGs).append(",")
                .append(ConsensusUnion.metricSMHD).append(",")
                .append(experimentData.originalTw).append(",")
                .append(experimentData.unionTw).append(",")
                .append(experimentData.minTW).append(",")
                .append(experimentData.meanTW).append(",")
                .append(experimentData.maxTW).append(",")
                .append(experimentData.tw).append(",")
                .append(experimentData.meanParents).append(",")
                .append(experimentData.maxParents).append(",")
                .append(experimentData.fusionUnion.getNumEdges()).append(",")
                .append(Utils.SMHDwithoutMoralize(experimentData.fusionUnionMoralized, experimentData.originalDAGsMoralized)).append(",")
                .append(Utils.fusionSimilarity((Dag) experimentData.fusionUnion, experimentData.originalDAGs));

        // Añadir los resultados dinámicamente para cada algoritmo
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(getTreeWidth(results.dag));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(meanParents(results.dag));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(maxParents(results.dag));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(results.dag.getNumEdges());
        }
        for (AlgorithmResults results : algorithmResultsList) {
            Graph moralizedResult = Utils.moralize(results.dag);
            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(experimentData.fusionUnionMoralized, moralizedResult));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            Graph moralizedResult = Utils.moralize(results.dag);
            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(moralizedResult, experimentData.originalDAGsMoralized));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(Utils.fusionSimilarity(results.dag, experimentData.originalDAGs));
        }
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(results.executionTime);
        }

        // Convertir a cadena final
        String line = lineBuilder.toString();

        BufferedWriter csvWriter;
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            if (new File(savePath).length() == 0) {
                csvWriter.write(header);
                if (experimentData.realNetwork) {
                    csvWriter.write(headerProbs);
                }
                csvWriter.write("\n");
            }
            csvWriter.write(line);
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }

        if (experimentData.realNetwork) {
            Dag[] dags = algorithmResultsList.stream().map(a -> a.dag).toArray(Dag[]::new);
            line = calculateProbs(dags, randomBN, experimentData.timeRecalc, experimentData.timeSampled, experimentData.timeUnion, experimentData.originalBNrecalcMarginals, experimentData.sampledBNsMarginals, experimentData.unionBNMarginals) + "\n";
        }
        else {
            line = "\n";
        }
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            csvWriter.write(line);
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }
    }

    public static String calculateProbs(Dag[] dags, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        // Almacenar los diferentes DAGs y resultados
        double[][][] marginals = new double[dags.length][][];
        double[] times = new double[dags.length];

        // Calcular marginals y tiempos para todos los DAGs
        for (int i = 0; i < dags.length; i++) {
            Result result = calculateMarginals(dags[i], randomBN);
            marginals[i] = result.marginals;
            times[i] = result.time;

            System.out.println("  DAG " + i + " marginals calculated. Time: " + result.time);
        }

        // Generar las métricas
        StringBuilder returnString = new StringBuilder(",");

        // Listado de todas las matrices de márgenes para cálculo de métricas
        List<double[][]> allMarginalsList = new ArrayList<>(Arrays.asList(marginals));
        allMarginalsList.add(unionBNMarginals);

        double[][][] allMarginals = allMarginalsList.toArray(new double[0][][]);  // Convertir la lista a array de matrices

        // Añadir diferencias
        returnString.append(Experiments.getMeanAbsoluteDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(Experiments.getMeanAbsoluteDiff(m, originalBNrecalcMarginals)).append(",");

        returnString.append(Experiments.getMeanQuadraticDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(Experiments.getMeanQuadraticDiff(m, originalBNrecalcMarginals)).append(",");

        returnString.append(Experiments.getMeanKLDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(Experiments.getMeanKLDiff(m, originalBNrecalcMarginals)).append(",");

        // Añadir tiempos
        returnString.append(timeRecalc).append(",")
                .append(timeSampled).append(",");
        for (double time : times) {
            returnString.append(time).append(",");
        }
        returnString.append(timeUnion);

        return returnString.toString();
    }

    static class Result {
        double[][] marginals;
        double time;
    }

    private static Result calculateMarginals(Dag dag, RandomBN randomBN) {
        Result result = new Result();
        try {
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(dag);
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            BayesIm bayesIm = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            result.time = (System.currentTimeMillis() - start) / 1000.0;
            result.marginals = Experiments.marginals(bayesIm, randomBN.categories, randomBN.nodesDags);
        } catch (OutOfMemoryError | Exception ex) {
            System.gc();
            System.err.println("Array size too large: " + ex.getClass());
        }
        return result;
    }


    /** Returns the mean difference between two marginals.
     *  Example: marg1 = [[0.1, 0.9], [0.1, 0.6, 0.3]], marg2 = [[0.2, 0.8], [0.3, 0.7, 0.0]]
     *  Returns: ((0.1 + 0.1) / 2 + (0.2 + 0.1 + 0.3) / 3)) / 2 = 0.15
     */
    public static double getMeanAbsoluteDiff(double[][] marg1, double[][] marg2) {
        if (marg1 == null) return -1;
        if (marg2 == null) return -2;
        double diff = 0;
        for (int i = 0; i < marg1.length; i++) {
            for (int j = 0; j < marg1[i].length; j++) {
                diff += (Math.abs(marg1[i][j] - marg2[i][j]) / marg1[i].length);
            }
        }
        return diff / (marg1.length);
    }

    public static double getMeanAbsoluteDiff(ArrayList<double[][]> marg1, double[][] marg2) {
        double diff = 0;
        for (double[][] doubles : marg1) {
            diff += getMeanAbsoluteDiff(doubles, marg2);
        }
        return diff / marg1.size();
    }

    /** Returns the mean quadratic difference between two marginals. */
    public static double getMeanQuadraticDiff(double[][] marg1, double[][] marg2) {
        if (marg1 == null) return -1;
        if (marg2 == null) return -2;
        double diff = 0;
        for (int i = 0; i < marg1.length; i++) {
            for (int j = 0; j < marg1[i].length; j++) {
                diff += (Math.pow(marg1[i][j] - marg2[i][j], 2) / marg1[i].length);
            }
        }
        return diff / (marg1.length);
    }

    public static double getMeanQuadraticDiff(ArrayList<double[][]> marg1, double[][] marg2) {
        double diff = 0;
        for (double[][] doubles : marg1) {
            diff += getMeanQuadraticDiff(doubles, marg2);
        }
        return diff / marg1.size();
    }

    /** Returns the mean Kullback-Leiber difference between two marginals. */
    public static double getMeanKLDiff(double[][] marg1, double[][] marg2) {
        if (marg1 == null) return -1;
        if (marg2 == null) return -2;
        double diff = 0;
        for (int i = 0; i < marg1.length; i++) {
            for (int j = 0; j < marg1[i].length; j++) {
                diff += ((marg1[i][j] * Math.log(marg1[i][j] / marg2[i][j])) / marg1[i].length);
            }
        }
        return diff / (marg1.length);
    }

    public static double getMeanKLDiff(ArrayList<double[][]> marg1, double[][] marg2) {
        double diff = 0;
        for (double[][] doubles : marg1) {
            diff += getMeanKLDiff(doubles, marg2);
        }
        return diff / marg1.size();
    }

    public static double meanParents(Graph dag) {
        double nParents = 0;
        for (Node node : dag.getNodes()) {
            nParents += dag.getParents(node).size();
        }
        return nParents / dag.getNumNodes();
    }

    public static int maxParents(Graph dag) {
        int nParents = 0;
        for (Node node : dag.getNodes()) {
            if (dag.getParents(node).size() > nParents) nParents = dag.getParents(node).size();
        }
        return nParents;
    }


    private static String generateDynamicHeader(List<String> algorithms) {
        StringBuilder header = new StringBuilder("numNodes,nDags,popSize,nIterations,maxTWGeneratedDAGs,seed,metricAgainstOriginalDAGS,metricSMHD,originalTW,unionTW,minTW,meanTW,maxTW,limitTW,originalMeanParents,originalMaxParents,unionEdges,unionSMHDoriginals,unionFusSimOriginals");

        String[] metrics = {"TW", "MeanParents", "MaxParents", "Edges", "SMHD", "SMHDOriginal", "FusSim", "Time"};
        for (String metric : metrics) {
            for (String algo : algorithms) {
                header.append(",").append(algo).append(metric);
            }
        }

        return header.toString();
    }

    private static String generateDynamicHeaderProbs(List<String> algorithms) {
        StringBuilder headerProbs = new StringBuilder();

        algorithms = new ArrayList<>(algorithms);
        algorithms.add(0, "sampled");
        algorithms.add("union");

        String[] metrics = {"DiffAbs", "DiffCuad", "DiffKL"};
        for (String metric : metrics) {
            for (String algo : algorithms) {
                headerProbs.append(",").append(algo).append(metric);
            }
        }

        headerProbs.append(",timeRecalc");
        for (String algo : algorithms) {
            headerProbs.append(",").append(algo).append("TimeProbs");
        }

        return headerProbs.toString();
    }

    public static class AlgorithmResults {
        public Dag dag;
        public double executionTime;

        public AlgorithmResults(Dag dag, double executionTime) {
            this.dag = dag;
            this.executionTime = executionTime;
        }
    }

    public static class ExperimentData {
        public boolean realNetwork;
        public List<String> algorithms;
        public double meanParents;
        public int maxParents;
        public double maxTWGeneratedDAGs;
        public Graph fusionUnion;
        public Graph fusionUnionMoralized;
        public List<Dag> originalDAGs;
        public List<Graph> originalDAGsMoralized;
        public String bbdd;
        public int nDags;
        public int popSize;
        public int nIterations;
        public int originalTw;
        public int unionTw;
        public int seed;
        public int minTW;
        public double meanTW;
        public int maxTW;
        public int tw;
        public double timeRecalc;
        public double timeSampled;
        public double timeUnion;
        public double[][] originalBNrecalcMarginals;
        public ArrayList<double[][]> sampledBNsMarginals;
        public double[][] unionBNMarginals;

        // Constructor
        public ExperimentData(boolean realNetwork, List<String> algorithms, double meanParents, int maxParents, double maxTWGeneratedDAGs, Graph fusionUnion, Graph fusionUnionMoralized, List<Dag> originalDAGs, List<Graph> originalDAGsMoralized, String bbdd, int nDags, int popSize, int nIterations, int seed, int originalTw, int unionTw, int minTW, double meanTW, int maxTW, int tw) {
            this.realNetwork = realNetwork;
            this.fusionUnion = fusionUnion;
            this.meanParents = meanParents;
            this.maxParents = maxParents;
            this.maxTWGeneratedDAGs = maxTWGeneratedDAGs;
            this.fusionUnionMoralized = fusionUnionMoralized;
            this.originalDAGs = originalDAGs;
            this.originalDAGsMoralized = originalDAGsMoralized;
            this.algorithms = algorithms;
            this.bbdd = bbdd;
            this.nDags = nDags;
            this.popSize = popSize;
            this.nIterations = nIterations;
            this.unionTw = unionTw;
            this.originalTw = originalTw;
            this.seed = seed;
            this.minTW = minTW;
            this.meanTW = meanTW;
            this.maxTW = maxTW;
            this.tw = tw;
        }

        public ExperimentData(boolean realNetwork, List<String> algorithms, double meanParents, int maxParents, double maxTWGeneratedDAGs, Graph fusionUnion, Graph fusionUnionMoralized, List<Dag> originalDAGs, List<Graph> originalDAGsMoralized, String bbdd, int nDags, int popSize, int nIterations, int seed, int originalTw, int unionTw, int minTW, double meanTW, int maxTW, int tw, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
            this(realNetwork, algorithms, meanParents, maxParents, maxTWGeneratedDAGs, fusionUnion, fusionUnionMoralized, originalDAGs, originalDAGsMoralized, bbdd, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw);
            this.timeRecalc = timeRecalc;
            this.timeSampled = timeSampled;
            this.timeUnion = timeUnion;
            this.originalBNrecalcMarginals = originalBNrecalcMarginals;
            this.sampledBNsMarginals = sampledBNsMarginals;
            this.unionBNMarginals = unionBNMarginals;
        }
    }

}
