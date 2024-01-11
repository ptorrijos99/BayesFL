package consensusBN.Experiments;

import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import static org.albacete.simd.bayesfl.data.BN_DataSet.divideDataSet;
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
        int nClients = 5;
        int popSize = 20;
        int nIterations = 100;
        int seed = 42;

        launchExperiment(net, bbdd, nClients, popSize, nIterations, seed);
    }

    public static void launchExperiment(String net, String bbdd, int nDags, int popSize, int nIterations, int seed) {
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
        String header = "numNodes,nDags,popSize,nIterations,seed,unionTW,maxTW,greedyTW,geneticTW,unionEdges,greedyEdges,geneticEdges,greedySMHD,geneticSMHD,timeUnion,timeGreedy,time\n";

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
        // Original BN
        BayesIm originalBN = randomBN.originalBayesIm;

        // Recalculate probabilities of the original BN given the data
        BayesPm bayesPm = new BayesPm(originalBN.getBayesPm());
        BayesIm originalBNrecalc = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();

        // Get the BayesIm of the sampled graphs
        ArrayList<BayesIm> sampledBNs = randomBN.setOfRandomBNs;

        // Get the BayesIm of the greedy graph
        bayesPm = new BayesPm(geneticUnion.greedyDag);
        for (int j = 0; j < bayesPm.getNumNodes(); j++) {
            bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
            bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
        }
        BayesIm greedyBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();

        // Get the BayesIm of the genetic graph
        bayesPm = new BayesPm(geneticUnion.bestDag);
        for (int j = 0; j < bayesPm.getNumNodes(); j++) {
            bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
            bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
        }
        BayesIm geneticBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();

        // Get the BayesIm of the union graph
        bayesPm = new BayesPm(geneticUnion.fusionUnion);
        for (int j = 0; j < bayesPm.getNumNodes(); j++) {
            bayesPm.setNumCategories(randomBN.nodesDags.get(j), randomBN.categories[j].size());
            bayesPm.setCategories(randomBN.nodesDags.get(j), randomBN.categories[j]);
        }
        BayesIm unionBN = new EmBayesEstimator(bayesPm, randomBN.data).getEstimatedIm();


        return null;
    }

    public static double[][] meanMarginals(BayesIm bn) {
        double[][] marginals = new double[bn.getNumNodes()][];
        for (int i = 0; i < bn.getNumNodes(); i++) {
            //marginals[i] = bn.getNode(i).getMarginal();
        }
        return marginals;
    }

    public static ArrayList<double[][]> meanMarginals(ArrayList<BayesIm> bns) {
        ArrayList<double[][]> marginals = new ArrayList<>();
        for (BayesIm bn : bns) {
            marginals.add(meanMarginals(bn));
        }
        return marginals;
    }

    /** Returns the mean difference between two marginals.
     *  Example: marg1 = [[0.1, 0.9], [0.1, 0.6, 0.3]], marg2 = [[0.2, 0.8], [0.3, 0.7, 0.0]]
     *  Returns: ((0.1 + 0.1) + (0.2 + 0.1 + 0.3)) / 2 = 0.4
     */
    public static double getMeanDifferenceBetweenMarginals(double[][] marg1, double[][] marg2) {
        double diff = 0;

        for (int i = 0; i < marg1.length; i++) {
            for (int j = 0; j < marg1[i].length; j++) {
                diff += Math.abs(marg1[i][j] - marg2[i][j]);
            }
        }

        return diff / (marg1.length);
    }


}
