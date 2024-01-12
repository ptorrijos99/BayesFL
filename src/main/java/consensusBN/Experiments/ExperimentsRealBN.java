package consensusBN.Experiments;

import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.albacete.simd.bayesfl.data.BN_DataSet.readData;
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
        int seed = Integer.parseInt(parameters[5]);

        // Launch the experiment
        launchExperiment(net, bbdd, nClients, popSize, nIterations, seed);
    }*/

    public static void main(String[] args) {
        String net = "child";
        String bbdd = "0";
        int nClients = 4;
        int popSize = 2;
        int nIterations = 1;
        int seed = 42;

        launchExperiment(net, bbdd, nClients, popSize, nIterations, seed);
    }

    public static void launchExperiment(String net, String bbdd, int nDags, int popSize, int nIterations, int seed) {
        String savePath = "./results/Server/" + bbdd + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + ".csv";
        if (new File(savePath).exists()) {
            System.out.println("The experiment has already been done.");
            System.out.println("Path: " + savePath);
            return;
        }

        // Read the .csv
        DataSet data = readData(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");

        // Read the .xbif
        BIFReader bayesianReader = new BIFReader();
        try {
            bayesianReader.processFile(PATH + "res/networks/" + net + ".xbif");
        } catch (Exception e) {throw new RuntimeException(e);}

        // Generate the DAGs
        RandomBN randomBN = new RandomBN(bayesianReader, data, seed, nDags);
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;

        marginals(randomBN.originalBayesIm, randomBN.categories, randomBN.nodesDags);

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnion = new GeneticTreeWidthUnion(seed);
        geneticUnion.populationSize = popSize;
        geneticUnion.numIterations = nIterations;

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

        geneticUnion.initializeVars(dags);
        Dag unionDag = geneticUnion.fusionUnion;

        // Find the treewidth of the union of the dags
        int treewidth = getTreeWidth(unionDag);
        System.out.println("Fusion Union Treewidth: " + treewidth);

        // Execute the genetic union for each treewidth from 2 to the limit
        for (int tw = 2; tw < treewidth; tw++) {
            System.out.println("Treewidth: " + tw);
            geneticUnion.maxTreewidth = tw;
            geneticUnion.fusionUnion(dags);

            // Save results
            saveRound(net+"."+bbdd, geneticUnion, randomBN, minTW, meanTW, maxTW, tw, nDags, popSize, nIterations, seed);
        }
    }

    public static void saveRound(String bbdd, GeneticTreeWidthUnion geneticUnion, RandomBN randomBN, int minTW, double meanTW, int maxTW, int tw, int nDags, int popSize, int nIterations, int seed) {
        String savePath = "./results/Server/" + bbdd + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + ".csv";
        String header = "numNodes,nDags,popSize,nIterations,seed," +
                "minTW,meanTW,maxTW,unionTW,limitTW,greedyTW,geneticTW," +
                "unionEdges,greedyEdges,geneticEdges," +
                "greedySMHD,geneticSMHD," +
                "timeUnion,timeGreedy,time," +
                "diffRealRecalc,diffRealSampled,diffRealGreedy,diffRealGenetic,diffRealUnion,diffRecalcSampled,diffRecalcGreedy,diffRecalcGenetic,diffRecalcUnion," +
                "timeRecalc,timeSampled,timeGreedy,timeGenetic,timeUnion" +
                "\n";

        String line = bbdd + "," +
                nDags + "," +
                popSize + "," +
                nIterations + "," +
                seed + "," +
                minTW + "," +
                meanTW + "," +
                maxTW + "," +
                getTreeWidth(geneticUnion.fusionUnion) + "," +
                tw + "," +
                getTreeWidth(geneticUnion.greedyDag) + "," +
                getTreeWidth(geneticUnion.bestDag) + "," +
                geneticUnion.fusionUnion.getNumEdges() + "," +
                geneticUnion.greedyDag.getNumEdges() + "," +
                geneticUnion.bestDag.getNumEdges() + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + "," +
                geneticUnion.executionTimeUnion + "," +
                geneticUnion.executionTimeGreedy + "," +
                geneticUnion.executionTime +
                calculateProbs(geneticUnion, randomBN) + "\n";

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

    public static String calculateProbs(GeneticTreeWidthUnion geneticUnion, RandomBN randomBN) {
        // Calculate the marginals of the original BN
        BayesIm originalBN = randomBN.originalBayesIm;
        double[][] originalBNMarginals = marginals(originalBN, randomBN.categories, randomBN.nodesDags);
        double[][] originalBNrecalcMarginals = null;
        double[][] unionBNMarginals = null;
        double[][] greedyBNMarginals = null;
        double[][] geneticBNMarginals = null;

        double timeRecalc = -1;
        double timeSampled = -1;
        double timeGreedy = -1;
        double timeGenetic = -1;
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
        } catch (OutOfMemoryError | IllegalArgumentException ex) {
            //Log the info
            System.err.println("REAL RECALCULATED GRAPH: Array size too large: " + ex.getClass());
        }

        // Get the BayesIm of the sampled graphs
        ArrayList<BayesIm> sampledBNs = randomBN.setOfRandomBNs;
        ArrayList<double[][]> sampledBNsMarginals = marginals(sampledBNs, randomBN.categories, randomBN.nodesDags);
        timeSampled = randomBN.timeSample;

        // Get the BayesIm of the greedy graph
        try{
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(geneticUnion.greedyDag);
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            BayesIm greedyBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            timeGreedy = (System.currentTimeMillis() - start) / 1000.0;
            greedyBNMarginals = marginals(greedyBN, randomBN.categories, randomBN.nodesDags);
        } catch (OutOfMemoryError | IllegalArgumentException ex) {
            //Log the info
            System.err.println("GREEDY GRAPH: Array size too large: " + ex.getClass());
        }

        // Get the BayesIm of the genetic graph
        try{
            double start = System.currentTimeMillis();
            BayesPm bayesPm = new BayesPm(geneticUnion.bestDag);
            for (int j = 0; j < bayesPm.getNumNodes(); j++) {
                bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
                bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
            }
            BayesIm geneticBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();
            timeGenetic = (System.currentTimeMillis() - start) / 1000.0;
            geneticBNMarginals = marginals(geneticBN, randomBN.categories, randomBN.nodesDags);
        } catch (OutOfMemoryError | IllegalArgumentException ex) {
            //Log the info
            System.err.println("GENETIC GRAPH: Array size too large: " + ex.getClass());
        }

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
        } catch (OutOfMemoryError | IllegalArgumentException ex) {
            //Log the info
            System.err.println("UNION GRAPH: Array size too large: " + ex.getClass());
        }

        String returnString = ",";

        // Calculate the mean difference between all the marginals and the original BN
        returnString += getMeanDiffBetweenMarginals(originalBNrecalcMarginals, originalBNMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(sampledBNsMarginals, originalBNMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(greedyBNMarginals, originalBNMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(geneticBNMarginals, originalBNMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(unionBNMarginals, originalBNMarginals) + ",";

        // Calculate the mean difference between all the marginals and the original BN with the recalculated probabilities
        returnString += getMeanDiffBetweenMarginals(sampledBNsMarginals, originalBNrecalcMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(greedyBNMarginals, originalBNrecalcMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(geneticBNMarginals, originalBNrecalcMarginals) + ",";
        returnString += getMeanDiffBetweenMarginals(unionBNMarginals, originalBNrecalcMarginals) + ",";

        // Add the times
        returnString += timeRecalc + ",";
        returnString += timeSampled + ",";
        returnString += timeGreedy + ",";
        returnString += timeGenetic + ",";
        returnString += timeUnion;

        return returnString;
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
     *  Returns: ((0.1 + 0.1) + (0.2 + 0.1 + 0.3)) / 2 = 0.4
     */
    public static double getMeanDiffBetweenMarginals(double[][] marg1, double[][] marg2) {
        if (marg1 == null) return -1;
        if (marg2 == null) return -2;
        double diff = 0;
        for (int i = 0; i < marg1.length; i++) {
            for (int j = 0; j < marg1[i].length; j++) {
                diff += Math.abs(marg1[i][j] - marg2[i][j]);
            }
        }
        return diff / (marg1.length);
    }

    public static double getMeanDiffBetweenMarginals(ArrayList<double[][]> marg1, double[][] marg2) {
        double diff = 0;
        for (double[][] doubles : marg1) {
            diff += getMeanDiffBetweenMarginals(doubles, marg2);
        }
        return diff / marg1.size();
    }


}
