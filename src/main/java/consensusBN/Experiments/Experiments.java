package consensusBN.Experiments;

import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.utils.Utils;

import java.io.*;
import java.util.ArrayList;

import static org.albacete.simd.bayesfl.data.BN_DataSet.divideDataSet;
import static org.albacete.simd.bayesfl.data.BN_DataSet.readData;
import static org.albacete.simd.utils.Utils.getTreeWidth;


public class Experiments {

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
        //String[] parameters = new String[]{"10", "10", "1000", "1000", "1"};

        System.out.println("Number of hyperparams: " + parameters.length);
        int i=0;
        for (String string : parameters) {
            System.out.println("Param[" + i + "]: " + string);
            i++;
        }

        // Read the parameters from file
        //String net = parameters[0];
        //String bbdd = parameters[1];
        int numNodes = Integer.parseInt(parameters[0]);
        int nClients = Integer.parseInt(parameters[1]);
        int popSize = Integer.parseInt(parameters[2]);
        int nIterations = Integer.parseInt(parameters[3]);
        int seed = Integer.parseInt(parameters[4]);

        // Launch the experiment
        launchExperiment(numNodes, nClients, popSize, nIterations, seed);
    }*/

    public static void main(String[] args) {
        //String net = "alarm";
        //String bbdd = "0";
        int numNodes = 10;
        int nClients = 20;
        int popSize = 20;
        int nIterations = 100;
        int seed = 42;

        launchExperiment(numNodes, nClients, popSize, nIterations, seed);
    }

    public static void launchExperimentData(String net, String bbdd, int nClients, int popSize, int nIterations, int seed) {
        String savePath = "./results/Server/" + net + "." + bbdd + "_GeneticTWFusion_" + nClients + "_" + popSize + "_" + nIterations + "_" + seed + ".csv";

        // Read the .csv and divide the data into nClients
        DataSet allData = readData(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");
        ArrayList<DataSet> divisionData = divideDataSet(allData, nClients);
        ArrayList<Dag> dags = new ArrayList<>();

        // Execute GES in each data division
        for (int i = 0; i < nClients; i++) {
            GES_BNBuilder algorithm = new GES_BNBuilder(divisionData.get(i), true);
            algorithm.setnItInterleaving(Integer.MAX_VALUE);
            Graph graph = algorithm.search();
            dags.add(new Dag(Utils.removeInconsistencies(graph)));
        }

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnion = new GeneticTreeWidthUnion(seed);
        geneticUnion.populationSize = popSize;
        geneticUnion.numIterations = nIterations;

        geneticUnion.initializeVars(dags);
        Dag unionDag = geneticUnion.fusionUnion;

        // Find the treewidth of the union of the dags
        int treewidth = getTreeWidth(unionDag);
        System.out.println("Fusion Union Treewidth: " + treewidth);

        // Execute the genetic union for each treewidth from 2 to the limit
        for (int tw = 2; tw < treewidth; tw++) {
            System.out.println("Treewidth: " + tw);
            Dag twUnion = geneticUnion.fusionUnion(dags);
        }
    }

    public static void launchExperiment(int numNodes, int nClients, int popSize, int nIterations, int seed) {
        // Generate the DAGs
        RandomBN randomBN = new RandomBN(seed, numNodes, nClients);
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;

        // Find the treewidth of the union of the dags
        GeneticTreeWidthUnion geneticUnion = new GeneticTreeWidthUnion(seed);
        geneticUnion.populationSize = popSize;
        geneticUnion.numIterations = nIterations;

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
            saveRound(geneticUnion, tw, numNodes, nClients, popSize, nIterations, seed);
        }
    }

    public static void saveRound(GeneticTreeWidthUnion geneticUnion, int maxTW, int numNodes, int nDags, int popSize, int nIterations, int seed) {
        String savePath = "./results/Server/" + numNodes + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + ".csv";

        String header = "numNodes,nDags,popSize,nIterations,seed,unionTW,maxTW,greedyTW,geneticTW,unionEdges,greedyEdges,geneticEdges,greedySMHD,geneticSMHD,timeUnion,timeGreedy,time\n";

        String line = numNodes + "," +
                nDags + "," +
                popSize + "," +
                nIterations + "," +
                seed + "," +
                getTreeWidth(geneticUnion.fusionUnion) + "," +
                maxTW + "," +
                getTreeWidth(geneticUnion.greedyDag) + "," +
                getTreeWidth(geneticUnion.bestDag) + "," +
                geneticUnion.fusionUnion.getNumEdges() + "," +
                geneticUnion.greedyDag.getNumEdges() + "," +
                geneticUnion.bestDag.getNumEdges() + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.greedyDag) + "," +
                Utils.SMHD(geneticUnion.fusionUnion,geneticUnion.bestDag) + "," +
                geneticUnion.executionTimeUnion + "," +
                geneticUnion.executionTimeGreedy + "," +
                geneticUnion.executionTime + "\n";

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
}
