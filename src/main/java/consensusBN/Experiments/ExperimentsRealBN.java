package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static bayesfl.data.BN_DataSet.readData;
import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;
import static org.albacete.simd.utils.Utils.getTreeWidth;


public class ExperimentsRealBN {

    public static String PATH = "./";

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
        String bbdd = parameters[1];
        int nClients = Integer.parseInt(parameters[2]);
        int popSize = Integer.parseInt(parameters[3]);
        int nIterations = Integer.parseInt(parameters[4]);
        double twLimit = Double.parseDouble(parameters[5]);
        int seed = Integer.parseInt(parameters[6]);

        // Launch the experiment
        launchExperiment(net, bbdd, nClients, popSize, nIterations, twLimit, seed);
    }*/

    public static void main(String[] args) {
        String net = "child";
        String bbdd = "0";
        int nClients = 10;
        int popSize = 100;
        int nIterations = 100;
        double twLimit = 2;
        int seed = 1;

        launchExperiment(net, bbdd, nClients, popSize, nIterations, twLimit, seed);
    }

    public static void launchExperiment(String net, String bbdd, int nDags, int popSize, int nIterations, double twLimit, int seed) {
        String savePath = "./results/Server/" + net+"."+bbdd + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + ".csv";
        int tw = 2;

        // TODO: DESCOMENTAR ESTAS LÍNEAS
        /*
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
            tw = Integer.parseInt(lastLine.split(",")[11]);
            System.out.println("Maximum treewidth found in file: " + tw);
            tw += 1;

            // Close the file
            try {
                br.close();
            } catch (IOException ignored) {}
        }
        */


        // Read the .csv
        DataSet data = readData(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");

        // Read the .xbif
        BIFReader bayesianReader = new BIFReader();
        try {
            bayesianReader.processFile(PATH + "res/networks/" + net + ".xbif");
        } catch (Exception e) {throw new RuntimeException(e);}

        // Generate the DAGs
        RandomBN randomBN = new RandomBN(bayesianReader, data, seed, nDags, twLimit);
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;

        marginals(randomBN.originalBayesIm, randomBN.categories, randomBN.nodesDags);

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnion = new GeneticTreeWidthUnion(dags, seed);
        geneticUnion.populationSize = popSize;
        geneticUnion.candidatesFromInitialDAGs = false;
        geneticUnion.repeatCandidates = false;
        geneticUnion.numIterations = nIterations;

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnionPuerta = new GeneticTreeWidthUnion(dags, seed);
        geneticUnionPuerta.populationSize = popSize;
        geneticUnionPuerta.candidatesFromInitialDAGs = true;
        geneticUnionPuerta.repeatCandidates = true;
        geneticUnionPuerta.numIterations = nIterations;

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnionPuerta2 = new GeneticTreeWidthUnion(dags, seed);
        geneticUnionPuerta2.populationSize = popSize;
        geneticUnionPuerta2.candidatesFromInitialDAGs = true;
        geneticUnionPuerta2.repeatCandidates = false;
        geneticUnionPuerta2.numIterations = nIterations;

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

        Dag unionDag = geneticUnion.fusionUnion;

        // Find the treewidth of the union of the dags
        int treewidth = getTreeWidth(unionDag);
        System.out.println("Fusion Union Treewidth: " + treewidth);

        if (treewidth <= tw) {
            System.out.println("The treewidth of the union is lower or equal than the treewidth limit");
            return;
        }

        BayesIm originalBN = randomBN.originalBayesIm;
        double[][] originalBNrecalcMarginals = null;
        double[][] unionBNMarginals = null;
        ArrayList<double[][]> sampledBNsMarginals = new ArrayList<>();
        double timeRecalc = -1;
        double timeSampled = -1;
        double timeUnion = -1;
        // Recalculate probabilities of the original BN given the data
        try{
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(originalBN.getBayesPm());
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            BayesIm originalBNrecalc = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            timeRecalc = (System.currentTimeMillis() - start) / 1000.0;
            originalBNrecalcMarginals = marginals(originalBNrecalc, randomBN.categories, randomBN.nodesDags);
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
                sampledBNsMarginals.add(marginals(sampledBN, randomBN.categories, randomBN.nodesDags));
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
        try{
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(geneticUnion.fusionUnion);
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            BayesIm unionBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            timeUnion = (System.currentTimeMillis() - start) / 1000.0;
            unionBNMarginals = marginals(unionBN, randomBN.categories, randomBN.nodesDags);
        } catch (OutOfMemoryError | Exception ex) {
            System.gc();
            //Log the info
            System.err.println("UNION GRAPH: Array size too large: " + ex.getClass());
        }

        // Execute the genetic union for each treewidth from the last executed to the limit
        for (; tw < treewidth; tw++) {
            System.out.println("\nTreewidth: " + tw);
            geneticUnion.maxTreewidth = tw;
            geneticUnion.fusionUnion();

            System.out.println("Greedy SMHD:\t\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + " | Edges: " + geneticUnion.greedyDag.getNumEdges() + " | Time: " + geneticUnion.executionTimeGreedy);
            System.out.println("Genetic SMHD: \t\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + " | Edges: " + geneticUnion.bestDag.getNumEdges() + " | Time: " + geneticUnion.executionTime);

            geneticUnionPuerta.maxTreewidth = tw;
            geneticUnionPuerta.fusionUnion();

            System.out.println("Greedy Puerta SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta.greedyDag) + " | Edges: " + geneticUnionPuerta.greedyDag.getNumEdges() + " | Time: " + geneticUnionPuerta.executionTimeGreedy);
            System.out.println("Genetic Puerta SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta.bestDag) + " | Edges: " + geneticUnionPuerta.bestDag.getNumEdges() + " | Time: " + geneticUnionPuerta.executionTime);

            geneticUnionPuerta2.maxTreewidth = tw;
            geneticUnionPuerta2.fusionUnion();

            System.out.println("Greedy Puerta2 SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta2.greedyDag) + " | Edges: " + geneticUnionPuerta2.greedyDag.getNumEdges() + " | Time: " + geneticUnionPuerta2.executionTimeGreedy);
            System.out.println("Genetic Puerta2 SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta2.bestDag) + " | Edges: " + geneticUnionPuerta2.bestDag.getNumEdges() + " | Time: " + geneticUnionPuerta2.executionTime);

            double start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = false;
            Dag superGreedyVacia = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyVacia = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy v SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyVacia) + " | Edges: " + superGreedyVacia.getNumEdges() + " | Time: " + timeSuperGreedyVacia);

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = true;
            Dag superGreedyVaciaAll = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyVaciaAll = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy v a SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyVaciaAll) + " | Edges: " + superGreedyVaciaAll.getNumEdges() + " | Time: " + timeSuperGreedyVaciaAll);

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = false;
            ConsensusUnion.initialDag = geneticUnion.greedyDag;
            Dag superGreedy = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedy = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedy) + " | Edges: " + superGreedy.getNumEdges() + " | Time: " + timeSuperGreedy);

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = true;
            ConsensusUnion.initialDag = geneticUnion.greedyDag;
            Dag superGreedyAll = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyAll = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy SMHD a:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyAll) + " | Edges: " + superGreedyAll.getNumEdges() + " | Time: " + timeSuperGreedyAll);

            // Save results
            saveRound(net+"."+bbdd, geneticUnion, randomBN, superGreedy, timeSuperGreedy, superGreedyAll, timeSuperGreedyAll, superGreedyVacia, timeSuperGreedyVacia, superGreedyVaciaAll, timeSuperGreedyVaciaAll, minTW, meanTW, maxTW, tw, nDags, popSize, nIterations, twLimit, seed, timeRecalc, timeSampled, timeUnion, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals);
        }
    }

    public static void saveRound(String bbdd, GeneticTreeWidthUnion geneticUnion, RandomBN randomBN, Dag superGreedy, double timeSuperGreedy, Dag superGreedyAll, double timeSuperGreedyAll, Dag superGreedyVacia, double timeSuperGreedyVacia, Dag superGreedyVaciaAll, double timeSuperGreedyVaciaAll, int minTW, double meanTW, int maxTW, int tw, int nDags, int popSize, int nIterations, double twLimit, int seed, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        String savePath = "./results/Server/" + bbdd + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + ".csv";
        String header = "numNodes,nDags,popSize,nIterations,seed," +
                "sampledTWLimit,originalTW,minTW,meanTW,maxTW,unionTW,limitTW,greedyTW,sgTW,sgaTW,sgvTW,sgvaTW,geneticTW," +
                "originalMeanParents,greedyMeanParents,sgMeanParents,sgaMeanParents,sgvMeanParents,sgvaMeanParents,geneticMeanParents,unionMeanParents," +
                "originalMaxParents,greedyMaxParents,sgMaxParents,sgaMaxParents,sgvMaxParents,sgvaMaxParents,geneticMaxParents,unionMaxParents," +
                "unionEdges,greedyEdges,sgEdges,sgaEdges,sgvEdges,sgvaEdges,geneticEdges," +
                "greedySMHD,sgSMHD,sgaSMHD,sgvSMHD,sgvaSMHD,geneticSMHD," +
                "timeUnion,timeGreedy,timeSG,timeSGa,timeSGv,timeSGva,time," +
                "diffAbsSampled,diffAbsGreedy,diffAbsSG,diffAbsSGa,diffAbsSGv,diffAbsSGva,diffAbsGenetic,diffAbsUnion," +
                "diffCuadSampled,diffCuadGreedy,diffCuadSG,diffCuadSGa,diffCuadSGv,diffCuadSGva,diffCuadGenetic,diffCuadUnion," +
                "diffKLSampled,diffKLGreedy,diffKLSG,diffKLSGa,diffKLSGv,diffKLSGva,diffKLGenetic,diffKLUnion," +
                "timeProbsRecalc,timeProbsSampled,timeProbsGreedy,timeProbsSG,timeProbsSGa,timeProbsSGv,timeProbsSGva,timeProbsGenetic,timeProbsUnion" +
                "\n";

        String line = bbdd + "," +
                nDags + "," +
                popSize + "," +
                nIterations + "," +
                seed + "," +
                twLimit + "," +
                getTreeWidth(new Dag(randomBN.originalBayesIm.getDag())) + "," +
                minTW + "," +
                meanTW + "," +
                maxTW + "," +
                getTreeWidth(geneticUnion.fusionUnion) + "," +
                tw + "," +
                getTreeWidth(geneticUnion.greedyDag) + "," +
                getTreeWidth(superGreedy) + "," +
                getTreeWidth(superGreedyAll) + "," +
                getTreeWidth(superGreedyVacia) + "," +
                getTreeWidth(superGreedyVaciaAll) + "," +
                getTreeWidth(geneticUnion.bestDag) + "," +

                meanParents(randomBN.originalBayesIm.getDag()) + "," +
                meanParents(geneticUnion.greedyDag) + "," +
                meanParents(superGreedy) + "," +
                meanParents(superGreedyAll) + "," +
                meanParents(superGreedyVacia) + "," +
                meanParents(superGreedyVaciaAll) + "," +
                meanParents(geneticUnion.bestDag) + "," +
                meanParents(geneticUnion.fusionUnion) + "," +
                maxParents(randomBN.originalBayesIm.getDag()) + "," +
                maxParents(geneticUnion.greedyDag) + "," +
                maxParents(superGreedy) + "," +
                maxParents(superGreedyAll) + "," +
                maxParents(superGreedyVacia) + "," +
                maxParents(superGreedyVaciaAll) + "," +
                maxParents(geneticUnion.bestDag) + "," +
                maxParents(geneticUnion.fusionUnion) + "," +

                geneticUnion.fusionUnion.getNumEdges() + "," +
                geneticUnion.greedyDag.getNumEdges() + "," +
                superGreedy.getNumEdges() + "," +
                superGreedyAll.getNumEdges() + "," +
                superGreedyVacia.getNumEdges() + "," +
                superGreedyVaciaAll.getNumEdges() + "," +
                geneticUnion.bestDag.getNumEdges() + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedy) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedyAll) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedyVacia) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedyVaciaAll) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + "," +
                geneticUnion.executionTimeUnion + "," +
                geneticUnion.executionTimeGreedy + "," +
                timeSuperGreedy + "," +
                timeSuperGreedyAll + "," +
                timeSuperGreedyVacia + "," +
                timeSuperGreedyVaciaAll + "," +
                geneticUnion.executionTime +
                calculateProbs(geneticUnion, randomBN, timeRecalc, timeSampled, timeUnion, superGreedy, superGreedyAll, superGreedyVacia, superGreedyVaciaAll, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals) + "\n";

        BufferedWriter csvWriter;
        try {
            if (!new File("./results/Server/").exists()) {
                new File("./results/Server/").mkdir();
            }

            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            if (new File(savePath).length() == 0) {
                csvWriter.write(header);
            }
            csvWriter.write(line);
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }
    }

    public static String calculateProbs(GeneticTreeWidthUnion geneticUnion, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion, Dag superGreedy, Dag superGreedyAll, Dag superGreedyVacia, Dag superGreedyVaciaAll, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        // Almacenar los diferentes DAGs y resultados
        Dag[] dags = {geneticUnion.greedyDag, superGreedy, superGreedyAll, superGreedyVacia, superGreedyVaciaAll, geneticUnion.bestDag};
        double[][][] marginals = new double[dags.length][][];
        double[] times = new double[dags.length];

        // Calcular marginals y tiempos para todos los DAGs
        for (int i = 0; i < dags.length; i++) {
            Result result = calculateMarginals(dags[i], randomBN);
            marginals[i] = result.marginals;
            times[i] = result.time;
        }

        // Generar las métricas
        StringBuilder returnString = new StringBuilder(",");

        // Listado de todas las matrices de márgenes para cálculo de métricas
        List<double[][]> allMarginalsList = new ArrayList<>(Arrays.asList(marginals));
        allMarginalsList.add(unionBNMarginals);

        double[][][] allMarginals = allMarginalsList.toArray(new double[0][][]);  // Convertir la lista a array de matrices

        // Añadir diferencias
        returnString.append(getMeanAbsoluteDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(getMeanAbsoluteDiff(m, originalBNrecalcMarginals)).append(",");

        returnString.append(getMeanQuadraticDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(getMeanQuadraticDiff(m, originalBNrecalcMarginals)).append(",");

        returnString.append(getMeanKLDiff(sampledBNsMarginals, originalBNrecalcMarginals)).append(",");
        for (double[][] m : allMarginals)
            returnString.append(getMeanKLDiff(m, originalBNrecalcMarginals)).append(",");

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
            result.marginals = marginals(bayesIm, randomBN.categories, randomBN.nodesDags);
        } catch (OutOfMemoryError | Exception ex) {
            System.gc();
            System.err.println("Array size too large: " + ex.getClass());
        }
        return result;
    }


    public static double[][] marginals(BayesIm bn, ArrayList<String>[] categories, ArrayList<Node> orderNodes) {
        double[][] marginals = new double[bn.getNumNodes()][];

        int indexOrder = 0;
        for (Node node : orderNodes) {
            int indexBN = bn.getNodeIndex(node);
            marginals[indexOrder] = new double[categories[indexOrder].size()];

            // Get the multiplication of the categories size of the parents
            List<Node> parents = bn.getDag().getParents(node);
            int size = 1;
            for (Node parent : parents) {
                int indexParent = orderNodes.indexOf(parent);
                size *= categories[indexParent].size();
            }

            // Calculate the marginals
            for (int j = 0; j < marginals[indexOrder].length; j++) {
                for (int k = 0; k < size; k++) {
                    marginals[indexOrder][j] += bn.getProbability(indexBN, k, j);
                }
                marginals[indexOrder][j] /= size;
            }
            indexOrder++;
        }

        return marginals;
    }

    public static ArrayList<double[][]> marginals(ArrayList<BayesIm> bns, ArrayList<String>[] categories, ArrayList<Node> orderNodes) {
        ArrayList<double[][]> margs = new ArrayList<>();
        for (BayesIm bn : bns) {
            margs.add(marginals(bn,categories,orderNodes));
        }
        return margs;
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



}
