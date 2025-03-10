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
import java.lang.reflect.Array;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static consensusBN.ConsensusUnion.fusionUnion;
import static consensusBN.Experiments.Experiments.getBDeuScore;
import static org.albacete.simd.utils.Utils.*;


public class ExperimentMinCut {

    public static String PATH = "./";
    public static boolean verbose = false;

    // TODO: UNCOMMENT THIS LINES
    public static void main(String[] args) {
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

        boolean equivalenceSearch = Boolean.parseBoolean(parameters[6]);

        ConsensusUnion.metricAgainstOriginalDAGs = true;
        ConsensusUnion.metricSMHD = true;

        boolean probabilities = false;
        boolean inference = true;

        String savePath = "./results/Server/" + net + "_MinCutTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + "_" + equivalenceSearch + ".csv";

        // Launch the experiment
        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, equivalenceSearch, probabilities, inference, savePath);
    }

    /*public static void main(String[] args) {
        // Real network (net = net.bbdd)
        String net = "water.0";

        // Generic network (net = number of nodes)
        //String net = ""+10;

        verbose = true;

        int nClients = 10;
        int popSize = 100;
        int nIterations = 100;
        double twLimit = 2;
        int seed = 0;

        ConsensusUnion.metricAgainstOriginalDAGs = true;
        ConsensusUnion.metricSMHD = true;

        boolean equivalenceSearch = true;

        boolean probabilities = false;
        boolean inference = true;

        String savePath = "./results/Server/" + net + "_MinCutTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + "_" + equivalenceSearch + ".csv";

        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, equivalenceSearch, probabilities, inference, savePath);
    }*/

    public static void launchExperiment(String net, int nDags, int popSize, int nIterations, double twLimit, int seed, boolean equivalenceSearch, boolean probabilities, boolean inference, String savePath) {
        List<String> algorithms = List.of("minCutBES");

        // Check if the folder (and subfolders) exists
        if (!new File("./results/Server/").exists()) {
            new File("./results/Server/").mkdirs();
        }

        // Check if the file exists
        if (new File(savePath).exists()) {
            System.out.println("File exists: " + savePath);
            return;
        }

        RandomBN randomBN;
        Dag realDag = null;
        int originalTw = -1;
        boolean realNetwork = net.contains(".");
        // Real network
        if (realNetwork) {
            String netName = net.split("\\.")[0];

            // Read the .csv
            DataSet data = readData(PATH + "res/networks/BBDD/" + netName+ "/" + netName + "." + seed + ".csv");

            // Read the .xbif
            BIFReader bayesianReader = new BIFReader();
            try {
                bayesianReader.processFile(PATH + "res/networks/" + netName + ".xbif");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            // Generate the DAGs
            randomBN = new RandomBN(bayesianReader, data, seed, nDags, twLimit);
            randomBN.generate();

            // Calculate the treewidth of the original DAG
            realDag = new Dag(randomBN.originalBayesIm.getDag());
            originalTw = getTreeWidth(realDag);
        }
        // Synthetic network
        else {
            // Generate the DAGs
            int numNodes = Integer.parseInt(net);
            randomBN = new RandomBN(seed, numNodes, nDags);
            randomBN.generate();
            realDag = new Dag(randomBN.setOfRandomDags.get(0));
        }
        ArrayList<Dag> dags = randomBN.setOfRandomDags;

        // Copy of dags
        ArrayList<Dag> dagsCopy = new ArrayList<>();
        for (Dag dag : dags) {
            dagsCopy.add(new Dag(dag));
        }

        // Set the algorithm. tw=2 to not limit the treewidth
        MinCutTreeWidthUnion minCutUnion = new MinCutTreeWidthUnion(dagsCopy, seed, 2, Double.POSITIVE_INFINITY);
        minCutUnion.experiments_tw = false;  // Disable the experiments_tw mode (write the result of each tw)
        minCutUnion.experiments_perc = true;  // Enable the experiments_perc mode (write the result of each percentage)
        minCutUnion.equivalenceSearch = equivalenceSearch;

        // Find the treewidth of the dags sampled
        int maxTW = 0, maxEdges = 0, maxParents = 0;
        double meanTW = 0, meanEdges = 0, meanParents = 0;
        int minTW = Integer.MAX_VALUE, minEdges = Integer.MAX_VALUE;
        for (Dag dag : dags) {
            int temp = getTreeWidth(dag);
            if (temp > maxTW) maxTW = temp;
            if (temp < minTW) minTW = temp;
            meanTW += temp;

            int tempEdges = dag.getNumEdges();
            if (tempEdges > maxEdges) maxEdges = tempEdges;
            if (tempEdges < minEdges) minEdges = tempEdges;
            meanEdges += tempEdges;

            meanParents += meanParents(dag);
            temp = maxParents(dag);
            if (temp > maxParents) maxParents = temp;
        }
        meanTW /= dags.size();
        meanParents /= dags.size();
        meanEdges /= dags.size();

        Dag unionDag = minCutUnion.fusionUnion;

        // Find the treewidth of the union of the dags
        int unionTw = getTreeWidth(unionDag);
        System.out.println("Fusion Union Treewidth: " + unionTw);

        // Find data of empty network
        Dag emptyDag = new Dag(unionDag.getNodes());
        double emptySMHD = Utils.SMHD(unionDag, emptyDag);
        double emptyFusSim = Utils.fusionSimilarity(unionDag, emptyDag);
        double emptySampledSMHD = Utils.SMHD(emptyDag, dags);
        double emptySampledFusSim = Utils.fusionSimilarity(emptyDag, dags);

        // Save the moralized original DAGs into a new list
        ArrayList<Graph> moralizedDags = new ArrayList<>();
        for (Dag dag : dags) {
            moralizedDags.add(Utils.moralize(dag));
        }

        // Heuristic Consensus BES (all of the treewidths)
        minCutUnion.fusion();
        List<Dag> outputExperimentDAGs = minCutUnion.outputExperimentDAGs;
        List<Double> outputExperimentTimes = minCutUnion.outputExperimentTimes;
        List<Double> outputExperimentPercentages = minCutUnion.outputExperimentPercentages;
        List<List<Dag>> outputExperimentDAGsList = minCutUnion.outputExperimentDAGsList;

        // Write the results for each percentage
        for (int i = outputExperimentPercentages.size()-1; i >= 0; i--) {
            if (verbose) System.out.println("\nPercentage: " + outputExperimentPercentages.get(i));

            Dag resultDag = outputExperimentDAGs.get(i);
            double resultTime = outputExperimentTimes.get(i);
            List<Dag> resultOriginalDags = outputExperimentDAGsList.get(i);

            int tw = getTreeWidth(resultDag);

            ExperimentData experimentData = new ExperimentData(emptySMHD, emptyFusSim, emptySampledSMHD, emptySampledFusSim, realDag, resultOriginalDags, equivalenceSearch, outputExperimentPercentages.get(i), realNetwork, algorithms, minEdges, meanEdges, maxEdges, meanParents, maxParents, twLimit, minCutUnion.fusionUnion, Utils.moralize(minCutUnion.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw);
            Dag newFusion = fusionUnion(resultOriginalDags);
            if (verbose) System.out.println("minCut SMHD new: \t" + Utils.SMHD(minCutUnion.fusionUnion, newFusion) + " | SMHD ORIG: " + Utils.SMHD(newFusion, dags) + " | FusSim ORIG: " + Utils.fusionSimilarity(newFusion, dags) + " | Edges: " + newFusion.getNumEdges() + " | Time: " + resultTime);

            List<AlgorithmResults> algorithmResultsList = new ArrayList<>(List.of(
                    new AlgorithmResults(newFusion, resultTime)
            ));

            saveRound(experimentData, algorithmResultsList, probabilities, inference, savePath);
        }

        // Write the inference results for each percentage
        if (realNetwork && (probabilities || inference)) {
            double[][] originalBNrecalcMarginals = null;
            double[][] unionBNMarginals = null;
            ArrayList<double[][]> sampledBNsMarginals = new ArrayList<>();

            double timeRecalc = -1;
            double timeSampled = -1;
            double timeUnion = -1;

            BayesIm originalBN = randomBN.originalBayesIm;
            ArrayList<BayesIm> sampledBNs = new ArrayList<>();
            BayesIm unionBN = null;
            BayesIm originalBNrecalc = null;

            // Recalculate probabilities of the original BN given the data
            try {
                double start = System.currentTimeMillis();
                BayesPm bayesPm = new BayesPm(originalBN.getBayesPm());
                for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                    bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                    bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
                }
                originalBNrecalc = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
                timeRecalc = (System.currentTimeMillis() - start) / 1000.0;
            } catch (OutOfMemoryError | Exception ex) {
                System.gc();
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
                    sampledBNs.add(new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm());
                    if (timeSampled == -1) timeSampled = (System.currentTimeMillis() - start) / 1000.0;
                    else timeSampled += (System.currentTimeMillis() - start) / 1000.0;
                } catch (OutOfMemoryError | Exception ex) {
                    System.gc();
                    System.err.println("REAL RECALCULATED GRAPH: Array size too large: " + ex.getClass());
                }
            }
            timeSampled /= randomBN.setOfRandomDags.size();

            // Get the BayesIm of the union graph
            try {
                double start = System.currentTimeMillis();
                BayesPm bayesPm = new BayesPm(minCutUnion.fusionUnion);
                for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                    bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                    bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
                }
                unionBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
                timeUnion = (System.currentTimeMillis() - start) / 1000.0;

            } catch (OutOfMemoryError | Exception ex) {
                System.gc();
                System.err.println("UNION GRAPH: Array size too large: " + ex.getClass());
            }

            if (probabilities) {
                if (originalBNrecalc != null)
                    originalBNrecalcMarginals = Experiments.marginals(originalBNrecalc, randomBN.categories, randomBN.nodesDags);
                if (unionBN != null)
                    unionBNMarginals = Experiments.marginals(unionBN, randomBN.categories, randomBN.nodesDags);
                for (BayesIm sampledBN : sampledBNs) {
                    if (sampledBN != null)
                        sampledBNsMarginals.add(Experiments.marginals(sampledBN, randomBN.categories, randomBN.nodesDags));
                }
            }

            double emptyOriginalSMHD = Utils.SMHD(originalBNrecalc.getDag(), emptyDag);
            double emptyOriginalFusSim = Utils.fusionSimilarity(new Dag (originalBNrecalc.getDag()), emptyDag);

            for (int i = outputExperimentPercentages.size()-1; i >= 0; i--) {
                if (verbose) System.out.println("\nPercentage minCut Inference: " + outputExperimentPercentages.get(i));

                double resultTime = outputExperimentTimes.get(i);
                List<Dag> resultOriginalDags = outputExperimentDAGsList.get(i);

                Dag newFusion = fusionUnion(resultOriginalDags);
                int tw = getTreeWidth(newFusion);

                ExperimentData experimentData = new ExperimentData(emptySMHD, emptyFusSim, emptySampledSMHD, emptySampledFusSim, emptyOriginalSMHD, emptyOriginalFusSim, realDag, resultOriginalDags, equivalenceSearch, outputExperimentPercentages.get(i), realNetwork, algorithms, minEdges, meanEdges, maxEdges, meanParents, maxParents, twLimit, minCutUnion.fusionUnion, Utils.moralize(minCutUnion.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw, timeRecalc, timeSampled, timeUnion, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals);

                List<AlgorithmResults> algorithmResultsList = new ArrayList<>(List.of(
                        new AlgorithmResults(newFusion, resultTime)
                ));

                if (probabilities) saveRoundProbabilities(experimentData, algorithmResultsList, originalBNrecalc, sampledBNs, unionBN, randomBN, false, savePath);
                if (inference) saveRoundProbabilities(experimentData, algorithmResultsList, originalBNrecalc, sampledBNs, unionBN, randomBN, true, savePath);
            }
        }
    }

    public static void saveRound(ExperimentData experimentData, List<AlgorithmResults> algorithmResultsList, boolean probabilities, boolean inference, String savePath) {
        String header = generateDynamicHeader(experimentData.algorithms);
        String headerProbs = generateDynamicHeaderProbs(experimentData.algorithms);
        String headerInf = generateDynamicHeaderInf(experimentData.algorithms);

        List<Graph> originalDagsMoralized = new ArrayList<>();
        for (Dag dag : experimentData.resultOriginalDags) {
            originalDagsMoralized.add(Utils.moralize(dag));
        }

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
                .append(experimentData.percentage).append(",")
                .append(experimentData.equivalenceSearch).append(",")
                .append(experimentData.originalTw).append(",")
                .append(experimentData.unionTw).append(",")
                .append(experimentData.minTW).append(",")
                .append(experimentData.meanTW).append(",")
                .append(experimentData.maxTW).append(",")
                .append(experimentData.tw).append(",")
                .append(experimentData.minEdges).append(",")
                .append(experimentData.meanEdges).append(",")
                .append(experimentData.maxEdges).append(",")
                .append(experimentData.meanParents).append(",")
                .append(experimentData.maxParents).append(",")
                .append(experimentData.fusionUnion.getNumEdges()).append(",")
                .append(Utils.SMHDwithoutMoralize(experimentData.fusionUnionMoralized, experimentData.originalDAGsMoralized)).append(",")
                .append(Utils.fusionSimilarity((Dag) experimentData.fusionUnion, experimentData.originalDAGs)).append(",")
                .append(experimentData.emptySMHD).append(",")
                .append(experimentData.emptyFusSim).append(",")
                .append(experimentData.emptySampledSMHD).append(",")
                .append(experimentData.emptySampledFusSim);


        // Añadir los resultados dinámicamente para cada algoritmo
        for (AlgorithmResults results : algorithmResultsList) {
            lineBuilder.append(",").append(getTreeWidth(results.dag));
            lineBuilder.append(",").append(meanParents(results.dag));
            lineBuilder.append(",").append(maxParents(results.dag));
            lineBuilder.append(",").append(results.dag.getNumEdges());

            Graph moralizedResult = Utils.moralize(results.dag);
            Graph moralizedReal = Utils.moralize(experimentData.realDAG);
            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(moralizedResult, experimentData.fusionUnionMoralized));
            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(moralizedResult, experimentData.originalDAGsMoralized));
            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(moralizedResult, moralizedReal));

            List<Double> fusSim = Utils.fusionSimilarityList(results.dag, experimentData.originalDAGs);
            lineBuilder.append(",").append(fusSim.get(0));
            lineBuilder.append(",").append(fusSim.get(1));
            lineBuilder.append(",").append(fusSim.get(2));
            lineBuilder.append(",").append(fusSim.get(3));

            List<Integer> fusSim3 = Utils.fusionSimilarityList(results.dag, experimentData.realDAG);
            lineBuilder.append(",").append(fusSim3.get(0));
            lineBuilder.append(",").append(fusSim3.get(1));
            lineBuilder.append(",").append(fusSim3.get(2));
            lineBuilder.append(",").append(fusSim3.get(3));

            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(originalDagsMoralized, experimentData.originalDAGsMoralized));

            List<Double> fusSim2 = Utils.fusionSimilarityList(experimentData.resultOriginalDags, experimentData.originalDAGs);
            lineBuilder.append(",").append(fusSim2.get(0));
            lineBuilder.append(",").append(fusSim2.get(1));
            lineBuilder.append(",").append(fusSim2.get(2));
            lineBuilder.append(",").append(fusSim2.get(3));

            lineBuilder.append(",").append(Utils.SMHDwithoutMoralize(moralizedReal, originalDagsMoralized));

            List<Double> fusSim4 = Utils.fusionSimilarityList(experimentData.realDAG, experimentData.resultOriginalDags);
            lineBuilder.append(",").append(fusSim4.get(0));
            lineBuilder.append(",").append(fusSim4.get(1));
            lineBuilder.append(",").append(fusSim4.get(2));
            lineBuilder.append(",").append(fusSim4.get(3));

            lineBuilder.append(",").append(results.executionTime);

            if (verbose) System.out.println("FusSim: " + fusSim + " | FusSim2: " + fusSim2);
            if (verbose) System.out.println("FusSim3: " + fusSim3 + " | FusSim4: " + fusSim4 + " | SMHD: " + Utils.SMHDwithoutMoralize(moralizedReal, originalDagsMoralized));
        }

        if (experimentData.realNetwork) {
            if (probabilities) {
                // If is a real network, add the spaces for the probabilities
                String[] headerProbsColumns = headerProbs.split(",");
                int numColumns = headerProbsColumns.length;

                // Add commas to the line
                lineBuilder.append(",".repeat(Math.max(0, numColumns - 1)));
            }
            if (inference) {
                // If is a real network, add the spaces for the probabilities
                String[] headerInfColumns = headerInf.split(",");
                int numColumns = headerInfColumns.length;

                // Add commas to the line
                lineBuilder.append(",".repeat(Math.max(0, numColumns - 1)));
            }
        }

        // Convertir a cadena final
        String line = lineBuilder.toString();

        BufferedWriter csvWriter;
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            if (new File(savePath).length() == 0) {
                csvWriter.write(header);
                if (experimentData.realNetwork) {
                    if (probabilities) csvWriter.write(headerProbs);
                    if (inference) csvWriter.write(headerInf);
                }
                csvWriter.write("\n");
            }
            csvWriter.write(line + "\n");
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }
    }

    public static void saveRoundProbabilities(ExperimentData experimentData, List<AlgorithmResults> algorithmResultsList, BayesIm originalBNrecalc, ArrayList<BayesIm> sampledBNs, BayesIm unionBN, RandomBN randomBN, boolean inference, String savePath) {
        if (verbose) {
            String temp = inference ? "Inference" : "Probabilities";
            System.out.println("Calculating " + temp + " for the real network " + experimentData.bbdd + ", percentage " + experimentData.percentage);
        }

        Dag[] dags = algorithmResultsList.stream().map(a -> a.dag).toArray(Dag[]::new);
        String lineProbs = "";
        lineProbs += "," + experimentData.emptyOriginalSMHD;
        lineProbs += "," + experimentData.emptyOriginalFusSim;

        if (!inference) lineProbs += calculateProbs(dags, randomBN, experimentData.timeRecalc, experimentData.timeSampled, experimentData.timeUnion, experimentData.originalBNrecalcMarginals, experimentData.sampledBNsMarginals, experimentData.unionBNMarginals);
        else lineProbs += calculateInference(dags, originalBNrecalc, sampledBNs, unionBN, randomBN, experimentData.timeRecalc, experimentData.timeSampled, experimentData.timeUnion);


        List<String> lines = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(savePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
            return;
        }

        String[] headerColumns = lines.get(0).split(",");
        int percentageTag = Arrays.asList(headerColumns).indexOf("percentage");

        String percentageValue = String.valueOf(experimentData.percentage);
        for (int i = 0; i < lines.size(); i++) {
            String currentLine = lines.get(i);

            String[] fields = currentLine.split(",");
            if (fields[percentageTag].equals(percentageValue)) {
                while (currentLine.endsWith(",")) {
                    currentLine = currentLine.substring(0, currentLine.length() - 1);
                }
                lines.set(i, currentLine + lineProbs);
                break;
            }
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(savePath))) {
            for (String updatedLine : lines) {
                writer.write(updatedLine);
                writer.newLine();
            }
        } catch (IOException e) {
            System.out.println("Error writing file: " + e.getMessage());
        }
    }

    public static String calculateInference(Dag[] dags, BayesIm originalBNrecalc, ArrayList<BayesIm> sampledBNs, BayesIm unionBN, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion) {
        DataSet data = randomBN.data;

        // Save DAGs and results
        BayesIm[] bayesIms = new BayesIm[dags.length];
        double[] timesLL = new double[dags.length + 4];
        double[] times = new double[dags.length + 4];

        // Calculate bayesIm and times
        for (int i = 0; i < dags.length; i++) {
            Result2 result = calculateBayesIm(dags[i], randomBN);
            bayesIms[i] = result.bayesIm;
            times[i+4] = result.time;
        }

        double llTemp = 0, bdeuTemp = 0;

        // Generate the metrics
        StringBuilder returnString = new StringBuilder(",");

        double timeTemp = System.currentTimeMillis();
        try {
            llTemp = Experiments.calculateLogLikelihood(originalBNrecalc, data);
        } catch (Exception e) {
            System.err.println("Error calculating log likelihood of original: " + e.getMessage());
        }
        try {
            bdeuTemp = getBDeuScore(originalBNrecalc.getDag(), data);
        } catch (Exception e) {
            System.err.println("Error calculating BDeu Score of original: " + e.getMessage());
        }

        timesLL[0] = (System.currentTimeMillis() - timeTemp) / 1000.0;
        times[0] = timeRecalc;
        returnString.append(llTemp).append(",");
        returnString.append(bdeuTemp).append(",");

        double sampledLogLik = 0, sampledBDeu = 0;
        llTemp = 0; bdeuTemp = 0;
        boolean errorLL = false, errorBDeu = false;
        timeTemp = System.currentTimeMillis();
        for (BayesIm sampledBN : sampledBNs) {
            try {
                llTemp = Experiments.calculateLogLikelihood(sampledBN, data);
                if (llTemp == 0) throw new Exception("Log likelihood is 0");
                sampledLogLik += llTemp;
            } catch (Exception e) {
                errorLL = true;
                System.err.println("Error calculating log likelihood of sampled: " + e.getMessage());
            }
            try {
                bdeuTemp = getBDeuScore(sampledBN.getDag(), data);
                if (bdeuTemp == 0) throw new Exception("BDeu Score is 0");
                sampledBDeu += bdeuTemp;
            } catch (Exception e) {
                errorBDeu = true;
                System.err.println("Error calculating BDeu Score of sampled: " + e.getMessage());
            }
        }
        if (!errorLL & !errorBDeu) timesLL[1] = ((System.currentTimeMillis() - timeTemp) / 1000.0) / sampledBNs.size();
        else timesLL[1] = -1;
        if (!errorLL) sampledLogLik /= sampledBNs.size(); else sampledLogLik = 0;
        if (!errorBDeu) sampledBDeu /= sampledBNs.size(); else sampledBDeu = 0;
        times[1] = timeSampled;
        returnString.append(sampledLogLik).append(",");
        returnString.append(sampledBDeu).append(",");

        llTemp = 0; bdeuTemp = 0;
        boolean error = false;
        timeTemp = System.currentTimeMillis();
        try {
            llTemp = Experiments.calculateLogLikelihood(unionBN, data);
        } catch (Exception e) {
            error = true;
            System.err.println("Error calculating log likelihood of union: " + e.getMessage());
        }
        try {
            bdeuTemp = getBDeuScore(unionBN.getDag(), data);
        } catch (Exception e) {
            error = true;
            System.err.println("Error calculating BDeu Score of union: " + e.getMessage());
        }

        if (!error && llTemp!=0 && bdeuTemp!=0) timesLL[2] = (System.currentTimeMillis() - timeTemp) / 1000.0;
        else timesLL[2] = -1;
        times[2] = timeUnion;
        returnString.append(llTemp).append(",");
        returnString.append(bdeuTemp).append(",");

        // Calculate empty graph
        llTemp = 0; bdeuTemp = 0;
        error = false;
        timeTemp = System.currentTimeMillis();
        Dag emptyDag = new Dag(unionBN.getVariables());
        Result2 result = calculateBayesIm(emptyDag, randomBN);
        BayesIm emptyBN = result.bayesIm;
        try {
            llTemp = Experiments.calculateLogLikelihood(emptyBN, data);
        } catch (Exception e) {
            error = true;
            System.err.println("Error calculating log likelihood of empty: " + e.getMessage());
        }
        try {
            bdeuTemp = getBDeuScore(emptyBN.getDag(), data);
        } catch (Exception e) {
            error = true;
            System.err.println("Error calculating BDeu Score of empty: " + e.getMessage());
        }
        if (!error && llTemp!=0 && bdeuTemp!=0) timesLL[3] = (System.currentTimeMillis() - timeTemp) / 1000.0;
        else timesLL[3] = -1;
        times[3] = result.time;
        returnString.append(llTemp).append(",");
        returnString.append(bdeuTemp).append(",");

        int i = 4;
        for (BayesIm bayesIm : bayesIms) {
            llTemp = 0; bdeuTemp = 0;
            error = false;
            timeTemp = System.currentTimeMillis();

            try {
                llTemp = Experiments.calculateLogLikelihood(bayesIm, data);
            } catch (Exception e) {
                error = true;
                System.err.println("Error calculating log likelihood of bayesIm: " + e.getMessage());
            }
            try {
                bdeuTemp = getBDeuScore(bayesIm.getDag(), data);
            } catch (Exception e) {
                error = true;
                System.err.println("Error calculating BDeu Score of bayesIm: " + e.getMessage());
            }

            if (!error && llTemp!=0 && bdeuTemp!=0) timesLL[i] = (System.currentTimeMillis() - timeTemp) / 1000.0;
            else timesLL[i] = -1;
            returnString.append(llTemp).append(",");
            returnString.append(bdeuTemp).append(",");
            i++;
        }

        // Add times of parametric learning
        for (double time : times) {
            returnString.append(time).append(",");
        }

        // Add times for the log likelihood
        for (double time : timesLL) {
            returnString.append(time).append(",");
        }

        // Remove the last comma
        returnString.deleteCharAt(returnString.length() - 1);

        return returnString.toString();
    }


    public static String calculateProbs(Dag[] dags, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        // Save DAGs and results
        double[][][] marginals = new double[dags.length][][];
        double[] times = new double[dags.length];

        // Calculate bayesIm and times
        for (int i = 0; i < dags.length; i++) {
            Result result = calculateMarginals(dags[i], randomBN);
            marginals[i] = result.marginals;
            times[i] = result.time;
        }

        // Generate metrics
        StringBuilder returnString = new StringBuilder(",");

        // Matrix for the marginals of the union graph
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

        // Add times
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

    public static Result calculateMarginals(Dag dag, RandomBN randomBN) {
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

    static class Result2 {
        BayesIm bayesIm;
        double time;
    }

    public static Result2 calculateBayesIm(Dag dag, RandomBN randomBN) {
        Result2 result = new Result2();
        try {
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(dag);
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            result.bayesIm = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            result.time = (System.currentTimeMillis() - start) / 1000.0;
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
        StringBuilder header = new StringBuilder("numNodes,nDags,popSize,nIterations,maxTWGeneratedDAGs,seed,metricAgainstOriginalDAGS,metricSMHD,percentage,equivalenceSearch,originalTW,unionTW,minTW,meanTW,maxTW,limitTW,minEdges,meanEdges,maxEdges,originalMeanParents,originalMaxParents,unionEdges,unionSMHDoriginals,unionFusSimOriginals,emptySMHD,emptyFusSim,emptySampledSMHD,emptySampledFusSim");

        String[] metrics = {"TW", "MeanParents", "MaxParents", "Edges", "SMHD", "SMHDOriginal", "SMHDReal", "FusSim1", "FusSim2", "FusSim3", "FusSim4", "FusSimReal1", "FusSimReal2", "FusSimReal3", "FusSimReal4", "SMHDIniciales", "FusSimIniciales1", "FusSimIniciales2", "FusSimIniciales3", "FusSimIniciales4", "SMHDInicialesReal", "FusSimInicialesReal1", "FusSimInicialesReal2", "FusSimInicialesReal3", "FusSimInicialesReal4", "Time"};

        for (String algo : algorithms) {
            for (String metric : metrics) {
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

        headerProbs.append(",recalcTimeProbs");
        for (String algo : algorithms) {
            headerProbs.append(",").append(algo).append("TimeProbs");
        }

        return headerProbs.toString();
    }

    private static String generateDynamicHeaderInf(List<String> algorithms) {
        StringBuilder headerProbs = new StringBuilder();

        ArrayList<String> algorithmsTemp = new ArrayList<>();
        algorithmsTemp.add("recalc");
        algorithmsTemp.add("sampled");
        algorithmsTemp.add("union");
        algorithmsTemp.add("empty");
        algorithmsTemp.addAll(algorithms);

        headerProbs.append(",emptyOriginalSMHD,emptyOriginalFusSim");

        String[] metrics = {"LL", "BDeu"};
        for (String algo : algorithmsTemp) {
            for (String metric : metrics) {
                headerProbs.append(",").append(algo).append(metric);
            }
        }

        for (String algo : algorithmsTemp) {
            headerProbs.append(",").append(algo).append("TimeProbs");
        }

        for (String algo : algorithmsTemp) {
            headerProbs.append(",").append(algo).append("TimeLL");
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
        public Dag realDAG;
        public List<Dag> resultOriginalDags;
        public boolean equivalenceSearch;
        public double percentage;
        public boolean realNetwork;
        public List<String> algorithms;
        public double minEdges;
        public double meanEdges;
        public double maxEdges;
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

        double emptySMHD;
        double emptyFusSim;
        double emptySampledSMHD;
        double emptySampledFusSim;
        double emptyOriginalSMHD;
        double emptyOriginalFusSim;

        // Constructor
        public ExperimentData(double emptySMHD, double emptyFusSim, double emptySampledSMHD, double emptySampledFusSim, Dag realDAG, List<Dag> resultOriginalDags, boolean equivalenceSearch, double percentage, boolean realNetwork, List<String> algorithms, double minEdges, double meanEdges, double maxEdges, double meanParents, int maxParents, double maxTWGeneratedDAGs, Graph fusionUnion, Graph fusionUnionMoralized, List<Dag> originalDAGs, List<Graph> originalDAGsMoralized, String bbdd, int nDags, int popSize, int nIterations, int seed, int originalTw, int unionTw, int minTW, double meanTW, int maxTW, int tw) {
            this.realDAG = realDAG;
            this.resultOriginalDags = resultOriginalDags;
            this.equivalenceSearch = equivalenceSearch;
            this.percentage = percentage;
            this.realNetwork = realNetwork;
            this.fusionUnion = fusionUnion;
            this.minEdges = minEdges;
            this.meanEdges = meanEdges;
            this.maxEdges = maxEdges;
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
            this.emptySMHD = emptySMHD;
            this.emptyFusSim = emptyFusSim;
            this.emptySampledSMHD = emptySampledSMHD;
            this.emptySampledFusSim = emptySampledFusSim;
        }

        public ExperimentData(double emptySMHD, double emptyFusSim, double emptySampledSMHD, double emptySampledFusSim, double emptyOriginalSMHD, double emptyOriginalFusSim, Dag realDAG, List<Dag> resultOriginalDags, boolean equivalenceSearch, double percentage, boolean realNetwork, List<String> algorithms,double minEdges, double meanEdges, double maxEdges, double meanParents, int maxParents, double maxTWGeneratedDAGs, Graph fusionUnion, Graph fusionUnionMoralized, List<Dag> originalDAGs, List<Graph> originalDAGsMoralized, String bbdd, int nDags, int popSize, int nIterations, int seed, int originalTw, int unionTw, int minTW, double meanTW, int maxTW, int tw, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
            this(emptySMHD, emptyFusSim, emptySampledSMHD, emptySampledFusSim, realDAG, resultOriginalDags, equivalenceSearch, percentage, realNetwork, algorithms, minEdges, meanEdges, maxEdges, meanParents, maxParents, maxTWGeneratedDAGs, fusionUnion, fusionUnionMoralized, originalDAGs, originalDAGsMoralized, bbdd, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw);
            this.timeRecalc = timeRecalc;
            this.timeSampled = timeSampled;
            this.timeUnion = timeUnion;
            this.originalBNrecalcMarginals = originalBNrecalcMarginals;
            this.sampledBNsMarginals = sampledBNsMarginals;
            this.unionBNMarginals = unionBNMarginals;
            this.emptyOriginalSMHD = emptyOriginalSMHD;
            this.emptyOriginalFusSim = emptyOriginalFusSim;
        }
    }

}
