package consensusBN.Experiments;

import consensusBN.ConsensusUnion;
import consensusBN.GeneticTreeWidthUnion;
import consensusBN.Method.Fusion_Method;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.albacete.simd.utils.Utils.readData;
import static org.albacete.simd.utils.Utils.getTreeWidth;


public class Experiments {

    public static String PATH = "./";

    // TODO: DESCOMENTAR ESTAS LÍNEAS
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

        // Launch the experiment
        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed);
    }*/

    public static void main(String[] args) {
        // Real network (net = net.bbdd)
        String net = "asia.0";

        // Generic network (net = number of nodes)
        //String net = ""+30;

        int nClients = 5;
        int popSize = 100;
        int nIterations = 100;
        double twLimit = 2;
        int seed = 1;

        launchExperiment(net, nClients, popSize, nIterations, twLimit, seed);
    }

    public static void launchExperiment(String net, int nDags, int popSize, int nIterations, double twLimit, int seed) {
        String savePath = "./results/Server/" + net + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + ".csv";

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
            tw = Integer.parseInt(lastLine.split(",")[11]);
            System.out.println("Maximum treewidth found in file: " + tw);
            tw += 1;

            // Close the file
            try {
                br.close();
            } catch (IOException ignored) {}
        }

        RandomBN randomBN;
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
        }
        // Synthetic network
        else {
            // Generate the DAGs
            int numNodes = Integer.parseInt(net);
            randomBN = new RandomBN(seed, numNodes, nDags);
        }
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;





        /*// Print the dags
        // Original DAG
        Dag originalDag = new Dag(randomBN.originalBayesIm.getDag());
        System.out.println("\nOriginal DAG with " + originalDag.getNumEdges() + " edges:");
        System.out.println("STATS: ");
        System.out.println("Treewidth: " + getTreeWidth(originalDag));
        System.out.println("MaxParents: " + maxParents(originalDag));
        System.out.println("MeanParents: " + meanParents(originalDag));
        System.out.println(Utils.graphToDot(randomBN.originalBayesIm.getDag()));
        // Generated DAGs
        for (int i = 0; i < dags.size(); i++) {
            System.out.println("\nDAG " + i + " with " + dags.get(i).getNumEdges() + " edges:");
            System.out.println("STATS: ");
            System.out.println("Treewidth: " + getTreeWidth(dags.get(i)));
            System.out.println("SHD: " + Utils.SHD(originalDag, dags.get(i)));
            System.out.println("SMHD: " + Utils.SMHD(originalDag, dags.get(i)));
            System.out.println("FusSim: " + Utils.fusionSimilarity(originalDag, dags.get(i)));
            System.out.println("MaxParents: " + maxParents(dags.get(i)));
            System.out.println("MeanParents: " + meanParents(dags.get(i)));
            System.out.println(Utils.graphToDot(dags.get(i)));
        }

        // Fusion DAG
        Dag unionDagPrueba = ConsensusUnion.fusionUnion(dags);
        System.out.println("\nFusion DAG with " + unionDagPrueba.getNumEdges() + " edges:");
        System.out.println("STATS: ");
        System.out.println("Treewidth: " + getTreeWidth(unionDagPrueba));
        System.out.println("SHD: " + Utils.SHD(originalDag, unionDagPrueba));
        System.out.println("SMHD: " + Utils.SMHD(originalDag, unionDagPrueba));
        System.out.println("FusSim: " + Utils.fusionSimilarity(originalDag, unionDagPrueba));
        System.out.println("MaxParents: " + maxParents(unionDagPrueba));
        System.out.println("MeanParents: " + meanParents(unionDagPrueba));
        System.out.println(Utils.graphToDot(unionDagPrueba));

*/


        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnion = new GeneticTreeWidthUnion(dags, seed);
        geneticUnion.populationSize = popSize;
        geneticUnion.candidatesFromInitialDAGs = false;
        geneticUnion.repeatCandidates = false;
        geneticUnion.useSuperGreedy = false;
        geneticUnion.addEmptySuperGreedy = false;
        geneticUnion.numIterations = nIterations;

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnionsg = new GeneticTreeWidthUnion(dags, seed);
        geneticUnionsg.populationSize = popSize;
        geneticUnionsg.candidatesFromInitialDAGs = false;
        geneticUnionsg.repeatCandidates = false;
        geneticUnionsg.useSuperGreedy = true;
        geneticUnionsg.addEmptySuperGreedy = false;
        geneticUnionsg.numIterations = nIterations;

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnionsgv = new GeneticTreeWidthUnion(dags, seed);
        geneticUnionsgv.populationSize = popSize;
        geneticUnionsgv.candidatesFromInitialDAGs = false;
        geneticUnionsgv.repeatCandidates = false;
        geneticUnionsgv.useSuperGreedy = true;
        geneticUnionsgv.addEmptySuperGreedy = true;
        geneticUnionsgv.numIterations = nIterations;

        /*// Find the treewidth of the union of the dags
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
        geneticUnionPuerta2.numIterations = nIterations;*/

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
            try {
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
        }

        // Execute the genetic union for each treewidth from the last executed to the limit
        for (; tw < treewidth; tw++) {
            System.out.println("\nTreewidth: " + tw);
            geneticUnion.maxTreewidth = tw;
            geneticUnion.fusionUnion();

            System.out.println("Genetic SMHD: \t\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + " | Edges: " + geneticUnion.bestDag.getNumEdges() + " | Time: " + geneticUnion.executionTime);

            geneticUnionsg.maxTreewidth = tw;
            geneticUnionsg.fusionUnion();

            System.out.println("GeneticSG SMHD: \t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionsg.bestDag) + " | Edges: " + geneticUnionsg.bestDag.getNumEdges() + " | Time: " + geneticUnionsg.executionTime);

            geneticUnionsgv.maxTreewidth = tw;
            geneticUnionsgv.fusionUnion();

            System.out.println("GeneticSG v SMHD: \t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionsgv.bestDag) + " | Edges: " + geneticUnionsgv.bestDag.getNumEdges() + " | Time: " + geneticUnionsgv.executionTime);

            System.out.println("__________________________________________________________");


            Dag superGreedy = ((Fusion_Method)geneticUnionsgv.method).superGreedyDag;
            Dag superGreedyVacia = ((Fusion_Method)geneticUnionsgv.method).superGreedyEmptyDag;
            double timeSuperGreedy = ((Fusion_Method)geneticUnionsgv.method).timeSuperGreedy;
            double timeSuperGreedyVacia = ((Fusion_Method)geneticUnionsgv.method).timeSuperGreedyEmpty;

            System.out.println("Greedy SMHD:\t\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + " | Edges: " + geneticUnion.greedyDag.getNumEdges() + " | Time: " + geneticUnion.executionTimeGreedy);
            System.out.println("SuperGreedy SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedy) + " | Edges: " + superGreedy.getNumEdges() + " | Time: " + timeSuperGreedy);
            System.out.println("SuperGreedy v SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyVacia) + " | Edges: " + superGreedyVacia.getNumEdges() + " | Time: " + timeSuperGreedyVacia);


            /*geneticUnionPuerta.maxTreewidth = tw;
            geneticUnionPuerta.fusionUnion();

            System.out.println("Greedy Puerta SMHD:\t\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta.greedyDag) + " | Edges: " + geneticUnionPuerta.greedyDag.getNumEdges() + " | Time: " + geneticUnionPuerta.executionTimeGreedy);
            System.out.println("Genetic Puerta SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta.bestDag) + " | Edges: " + geneticUnionPuerta.bestDag.getNumEdges() + " | Time: " + geneticUnionPuerta.executionTime);

            geneticUnionPuerta2.maxTreewidth = tw;
            geneticUnionPuerta2.fusionUnion();

            System.out.println("Greedy Puerta2 SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta2.greedyDag) + " | Edges: " + geneticUnionPuerta2.greedyDag.getNumEdges() + " | Time: " + geneticUnionPuerta2.executionTimeGreedy);
            System.out.println("Genetic Puerta2 SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,geneticUnionPuerta2.bestDag) + " | Edges: " + geneticUnionPuerta2.bestDag.getNumEdges() + " | Time: " + geneticUnionPuerta2.executionTime);


            /* double start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = false;
            ConsensusUnion.initialDag = null;
            Dag superGreedyVacia = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyVacia = (System.currentTimeMillis() - start) / 1000.0;

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = false;
            ConsensusUnion.initialDag = geneticUnion.greedyDag;
            Dag superGreedy = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedy = (System.currentTimeMillis() - start) / 1000.0;

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = true;
            ConsensusUnion.initialDag = null;
            Dag superGreedyVaciaAll = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyVaciaAll = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy_all v SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyVaciaAll) + " | Edges: " + superGreedyVaciaAll.getNumEdges() + " | Time: " + timeSuperGreedyVaciaAll);

            start = System.currentTimeMillis();
            ConsensusUnion.allPossibleArcs = true;
            ConsensusUnion.initialDag = geneticUnion.greedyDag;
            Dag superGreedyAll = ConsensusUnion.fusionUnion(dags, "SuperGreedyMaxTreewidth", ""+tw);
            double timeSuperGreedyAll = (System.currentTimeMillis() - start) / 1000.0;
            System.out.println("SuperGreedy_all SMHD:\t" + Utils.SMHD(geneticUnion.fusionUnion,superGreedyAll) + " | Edges: " + superGreedyAll.getNumEdges() + " | Time: " + timeSuperGreedyAll);
            */

            // Save results
            saveRound(realNetwork, net, geneticUnion, geneticUnionsg, randomBN, superGreedy, timeSuperGreedy, superGreedyVacia, timeSuperGreedyVacia, minTW, meanTW, maxTW, tw, nDags, popSize, nIterations, twLimit, seed, timeRecalc, timeSampled, timeUnion, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals);
        }
    }

    public static void saveRound(boolean realNetwork, String bbdd, GeneticTreeWidthUnion geneticUnion, GeneticTreeWidthUnion geneticUnionsg, RandomBN randomBN, Dag superGreedy, double timeSuperGreedy, Dag superGreedyVacia, double timeSuperGreedyVacia, int minTW, double meanTW, int maxTW, int tw, int nDags, int popSize, int nIterations, double twLimit, int seed, double timeRecalc, double timeSampled, double timeUnion, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        String savePath = "./results/Server/" + bbdd + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + "_" + twLimit + ".csv";
        String header = "numNodes,nDags,popSize,nIterations,seed," +
                "sampledTWLimit,originalTW,minTW,meanTW,maxTW,unionTW,limitTW,greedyTW,sgTW,sgvTW,geneticTW,geneticsgTW," +
                "originalMeanParents,greedyMeanParents,sgMeanParents,sgvMeanParents,geneticMeanParents,geneticsgMeanParents,unionMeanParents," +
                "originalMaxParents,greedyMaxParents,sgMaxParents,sgvMaxParents,geneticMaxParents,geneticsgMaxParents,unionMaxParents," +
                "unionEdges,greedyEdges,sgEdges,sgvEdges,geneticEdges,geneticsgEdges," +
                "greedySMHD,sgSMHD,sgvSMHD,geneticSMHD,geneticsgSMHD," +
                "timeUnion,timeGreedy,timeSG,timeSGv,timeGenetic,timeGeneticsg";
        String headerProbs = ",diffAbsSampled,diffAbsGreedy,diffAbsSG,diffAbsSGv,diffAbsGenetic,diffAbsGeneticsg,diffAbsUnion," +
                "diffCuadSampled,diffCuadGreedy,diffCuadSG,diffCuadSGv,diffCuadGenetic,diffCuadGeneticsg,diffCuadUnion," +
                "diffKLSampled,diffKLGreedy,diffKLSG,diffKLSGv,diffKLGenetic,diffKLGeneticsg,diffKLUnion," +
                "timeProbsRecalc,timeProbsSampled,timeProbsGreedy,timeProbsSG,timeProbsSGv,timeProbsGenetic,timeProbsGeneticsg,timeProbsUnion" +
                "\n";

        double originalTW=-1, meanParents=0, maxParents=0;
        if (realNetwork) {
            originalTW = getTreeWidth(new Dag(randomBN.originalBayesIm.getDag()));
            meanParents = meanParents(randomBN.originalBayesIm.getDag());
            maxParents = maxParents(randomBN.originalBayesIm.getDag());
        }
        else {
            originalTW = meanTW;
            for (Dag dag : randomBN.setOfRandomDags) {
                meanParents += meanParents(dag);
                maxParents += maxParents(dag);
            }
            meanParents /= randomBN.setOfRandomDags.size();
            maxParents /= randomBN.setOfRandomDags.size();
        }


        String line = bbdd + "," +
                nDags + "," +
                popSize + "," +
                nIterations + "," +
                seed + "," +
                twLimit + "," +
                originalTW + "," +
                minTW + "," +
                meanTW + "," +
                maxTW + "," +
                getTreeWidth(geneticUnion.fusionUnion) + "," +
                tw + "," +
                getTreeWidth(geneticUnion.greedyDag) + "," +
                getTreeWidth(superGreedy) + "," +
                getTreeWidth(superGreedyVacia) + "," +
                getTreeWidth(geneticUnion.bestDag) + "," +
                getTreeWidth(geneticUnionsg.bestDag) + "," +

                meanParents + "," +
                meanParents(geneticUnion.greedyDag) + "," +
                meanParents(superGreedy) + "," +
                meanParents(superGreedyVacia) + "," +
                meanParents(geneticUnion.bestDag) + "," +
                meanParents(geneticUnionsg.bestDag) + "," +
                meanParents(geneticUnion.fusionUnion) + "," +
                maxParents + "," +
                maxParents(geneticUnion.greedyDag) + "," +
                maxParents(superGreedy) + "," +
                maxParents(superGreedyVacia) + "," +
                maxParents(geneticUnion.bestDag) + "," +
                maxParents(geneticUnionsg.bestDag) + "," +
                maxParents(geneticUnion.fusionUnion) + "," +

                geneticUnion.fusionUnion.getNumEdges() + "," +
                geneticUnion.greedyDag.getNumEdges() + "," +
                superGreedy.getNumEdges() + "," +
                superGreedyVacia.getNumEdges() + "," +
                geneticUnion.bestDag.getNumEdges() + "," +
                geneticUnionsg.bestDag.getNumEdges() + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedy) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,superGreedyVacia) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + "," +
                Utils.SMHD(geneticUnionsg.fusionUnion,geneticUnionsg.bestDag) + "," +
                geneticUnion.executionTimeUnion + "," +
                geneticUnion.executionTimeGreedy + "," +
                timeSuperGreedy + "," +
                timeSuperGreedyVacia + "," +
                geneticUnionsg.executionTime + "," +
                geneticUnion.executionTime;

        BufferedWriter csvWriter;
        try {
            if (!new File("./results/Server/").exists()) {
                new File("./results/Server/").mkdir();
            }

            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            if (new File(savePath).length() == 0) {
                csvWriter.write(header);
                if (realNetwork) {
                    csvWriter.write(headerProbs);
                }
                else {
                    csvWriter.write("\n");
                }
            }
            csvWriter.write(line);
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException e) { System.out.println(e); }

        if (realNetwork) {
            line = calculateProbs(geneticUnion, geneticUnionsg, randomBN, timeRecalc, timeSampled, timeUnion, superGreedy, superGreedyVacia, originalBNrecalcMarginals, sampledBNsMarginals, unionBNMarginals) + "\n";
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

    public static String calculateProbs(GeneticTreeWidthUnion geneticUnion, GeneticTreeWidthUnion geneticUnionsg, RandomBN randomBN, double timeRecalc, double timeSampled, double timeUnion, Dag superGreedy, Dag superGreedyVacia, double[][] originalBNrecalcMarginals, ArrayList<double[][]> sampledBNsMarginals, double[][] unionBNMarginals) {
        // Almacenar los diferentes DAGs y resultados
        Dag[] dags = {geneticUnion.greedyDag, superGreedy, superGreedyVacia, geneticUnion.bestDag, geneticUnionsg.bestDag};
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
