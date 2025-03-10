package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.MinCutTreeWidthUnion;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.EmBayesEstimator;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;

import static consensusBN.ConsensusUnion.fusionUnion;
import static consensusBN.Experiments.ExperimentMinCut.*;
import static consensusBN.Experiments.Experiments.getBDeuScore;
import static org.albacete.simd.utils.Utils.getTreeWidth;
import static org.albacete.simd.utils.Utils.readData;

import consensusBN.Experiments.ExperimentMinCut.*;


public class ExperimentMinCutGES {

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

        boolean equivalenceSearch = Boolean.parseBoolean(parameters[6]);

        ConsensusUnion.metricAgainstOriginalDAGs = true;
        ConsensusUnion.metricSMHD = true;

        boolean probabilities = false;
        boolean inference = true;

        String savePath = "./results/Server/" + net + "_MinCutTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + "_" + equivalenceSearch + ".csv";

        // Launch the experiment
        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed, equivalenceSearch, probabilities, inference, savePath);
    }*/

    public static void main(String[] args) {
        // Real network (net = net.bbdd)
        String net = "barley";

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
    }

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

        String netName = net.split("\\.")[0];

        // Read the .xbif
        BIFReader bayesianReader = new BIFReader();
        String pathXBIF = PATH + "res/networks/" + netName + ".xbif";
        try {
            bayesianReader.processFile(pathXBIF);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Calculate the treewidth of the original DAG
        BayesIm originalBN = null;

        String pathNet = PATH + "res/networks/BBDD/" + netName+ "/" + netName + "." + seed + ".csv";
        DataSet data = readData(pathNet);
        // Original DAG
        ArrayList<String>[] categories = new ArrayList[data.getNumColumns()];
        for (Node node : data.getVariables()) {
            categories[data.getColumn(node)] = new ArrayList<>(((DiscreteVariable)node).getCategories());
        }

        // Carga la caché persistida si existe
        String cachePath = PATH + "results/Cache/";
        File cacheDir = new File(cachePath);
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }

        File cacheFile = new File(cachePath + "cachedDAGs-"+net+".ser");
        Map<Integer, Dag> cachedDags = readCache(cacheFile);

        // Generate DAGs with GES algorithm
        ArrayList<Dag> dags = new ArrayList<>();
        ArrayList<RandomBN> randomBNs = new ArrayList<>();
        for (int i = 0; i < nDags; i++) {
            if (verbose) System.out.println("Generating DAG " + i);
            DataSet dataTemp = readData(PATH + "res/networks/BBDD/" + netName+ "/" + netName + "." + i + ".csv");

            BNBuilder algorithm = new GES_BNBuilder(dataTemp, true);
            Dag temp;
            if (cachedDags.containsKey(i)) {
                temp = cachedDags.get(i);
            } else {
                temp = new Dag(algorithm.search());

                cachedDags.put(i, temp);
            }
            dags.add(temp);

            randomBNs.add(new RandomBN(bayesianReader, dataTemp, seed, 1, twLimit));

            if (verbose) System.out.println("DAG " + i + " TW: " + getTreeWidth(temp) + " | Edges: " + temp.getNumEdges() + " | BDeu: " + getBDeuScore(temp, dataTemp) + " | Global BDeu: " + getBDeuScore(temp, data));
        }

        // Guardar la caché persistida
        writeCache(cacheFile, cachedDags);

        originalBN =  Utils.transformBayesNetToBayesIm(bayesianReader, categories);
        realDag = new Dag(originalBN.getDag());
        originalTw = getTreeWidth(realDag);

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

        try {
            System.out.println("\nFusion Union Treewidth: " + unionTw + " | Edges: " + unionDag.getNumEdges() + " | BDeu: " + getBDeuScore(unionDag, data));
        }
        catch (Exception e) {
            System.out.println("Error calculating BDeu: " + e);
        }

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

        randomBN = new RandomBN(bayesianReader, data, seed, nDags, twLimit);

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

            ExperimentData experimentData = new ExperimentData(emptySMHD, emptyFusSim, emptySampledSMHD, emptySampledFusSim, realDag, resultOriginalDags, equivalenceSearch, outputExperimentPercentages.get(i), true, algorithms, minEdges, meanEdges, maxEdges, meanParents, maxParents, twLimit, minCutUnion.fusionUnion, Utils.moralize(minCutUnion.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw);
            Dag newFusion = fusionUnion(resultOriginalDags);
            if (verbose) System.out.println("minCut SMHD new: \t" + Utils.SMHD(minCutUnion.fusionUnion, newFusion) + " | SMHD ORIG: " + Utils.SMHD(newFusion, dags) + " | FusSim ORIG: " + Utils.fusionSimilarity(newFusion, dags) + " | Edges: " + newFusion.getNumEdges() + " | Time: " + resultTime);

            List<AlgorithmResults> algorithmResultsList = new ArrayList<>(List.of(
                    new AlgorithmResults(newFusion, resultTime)
            ));

            saveRound(experimentData, algorithmResultsList, probabilities, inference, savePath);
        }

        // Write the inference results for each percentage
        if (probabilities || inference) {
            double[][] originalBNrecalcMarginals = null;
            double[][] unionBNMarginals = null;
            ArrayList<double[][]> sampledBNsMarginals = new ArrayList<>();

            double timeRecalc = -1;
            double timeSampled = -1;
            double timeUnion = -1;

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
            for (Dag sampledDag : dags) {
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
            timeSampled /= dags.size();

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
                System.out.println(ex);
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

                ExperimentData experimentData = new ExperimentData(emptySMHD, emptyFusSim, emptySampledSMHD, emptySampledFusSim, emptyOriginalSMHD, emptyOriginalFusSim, realDag, resultOriginalDags, equivalenceSearch, outputExperimentPercentages.get(i), true, algorithms, minEdges, meanEdges, maxEdges, meanParents, maxParents, twLimit, minCutUnion.fusionUnion, Utils.moralize(minCutUnion.fusionUnion), dags, moralizedDags, net, nDags, popSize, nIterations, seed, originalTw, unionTw, minTW, meanTW, maxTW, tw, timeRecalc, timeSampled, timeUnion, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals);

                List<AlgorithmResults> algorithmResultsList = new ArrayList<>(List.of(
                        new AlgorithmResults(newFusion, resultTime)
                ));

                if (probabilities) saveRoundProbabilities(randomBNs, experimentData, algorithmResultsList, originalBNrecalc, sampledBNs, unionBN, unionDag, randomBN, false, savePath);
                if (inference) saveRoundProbabilities(randomBNs, experimentData, algorithmResultsList, originalBNrecalc, sampledBNs, unionBN, unionDag, randomBN, true, savePath);
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

    public static void saveRoundProbabilities(ArrayList<RandomBN> randomBNs, ExperimentData experimentData, List<AlgorithmResults> algorithmResultsList, BayesIm originalBNrecalc, ArrayList<BayesIm> sampledBNs, BayesIm unionBN, Dag unionDag, RandomBN randomBN,boolean inference, String savePath) {
        if (verbose) {
            String temp = inference ? "Inference" : "Probabilities";
            System.out.println("Calculating " + temp + " for the real network " + experimentData.bbdd + ", percentage " + experimentData.percentage);
        }

        Dag[] dags = algorithmResultsList.stream().map(a -> a.dag).toArray(Dag[]::new);
        String lineProbs = "";
        lineProbs += "," + experimentData.emptyOriginalSMHD;
        lineProbs += "," + experimentData.emptyOriginalFusSim;

        if (!inference) lineProbs += calculateProbs(dags, randomBN, experimentData.timeRecalc, experimentData.timeSampled, experimentData.timeUnion, experimentData.originalBNrecalcMarginals, experimentData.sampledBNsMarginals, experimentData.unionBNMarginals);
        else lineProbs += calculateInference(randomBNs, dags, originalBNrecalc, sampledBNs, unionBN, unionDag, randomBN, experimentData.timeRecalc, experimentData.timeSampled, experimentData.timeUnion);

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

    public static String calculateInference(ArrayList<RandomBN> randomBNs, Dag[] dags, BayesIm originalBNrecalc, ArrayList<BayesIm> sampledBNs, BayesIm unionBN, Dag unionDag, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion) {
        DataSet data = randomBN.data;

        // Save DAGs and results
        BayesIm[] bayesIms = new BayesIm[dags.length];
        double[] timesLL = new double[dags.length + 4];
        double[] times = new double[dags.length + 4];

        // Calculate bayesIm and times
        for (int i = 0; i < dags.length; i++) {
            Result2 result = calculateBayesIm(dags[i], randomBNs.get(i));
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
            bdeuTemp = getBDeuScore(unionDag, data);
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
        Dag emptyDag = new Dag(dags[0].getNodes());
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


    public static Map<Integer, Dag> readCache(File cacheFile) {
        Map<Integer, Dag> cachedDags = new HashMap<>();
        if (!cacheFile.exists()) return cachedDags;
        try (RandomAccessFile raf = new RandomAccessFile(cacheFile, "r");
             FileChannel channel = raf.getChannel();
             FileLock lock = channel.lock(0L, Long.MAX_VALUE, true)) {
            ByteBuffer buffer = ByteBuffer.allocate((int) channel.size());
            channel.read(buffer);
            buffer.flip();
            byte[] data = new byte[buffer.remaining()];
            buffer.get(data);
            try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data))) {
                cachedDags = (Map<Integer, Dag>) ois.readObject();
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return cachedDags;
    }

    public static void writeCache(File cacheFile, Map<Integer, Dag> newEntries) {
        Map<Integer, Dag> mergedCache = readCache(cacheFile);
        newEntries.forEach(mergedCache::putIfAbsent);
        try (RandomAccessFile raf = new RandomAccessFile(cacheFile, "rw");
             FileChannel channel = raf.getChannel();
             FileLock lock = channel.lock()) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(mergedCache);
            oos.flush();
            byte[] data = bos.toByteArray();
            channel.truncate(0);
            channel.position(0);
            channel.write(ByteBuffer.wrap(data));
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
