package consensusBN.Experiments;

import consensusBN.GeneticTreeWidthUnion;
import edu.cmu.tetrad.graph.Dag;
import org.albacete.simd.utils.Utils;

import java.io.*;
import java.util.ArrayList;

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
        int nClients = 10;
        int popSize = 20;
        int nIterations = 100;
        int seed = 1;

        launchExperiment(numNodes, nClients, popSize, nIterations, seed);
    }

    public static void launchExperiment(int numNodes, int nDags, int popSize, int nIterations, int seed) {
        // Generate the DAGs
        RandomBN randomBN = new RandomBN(seed, numNodes, nDags);
        randomBN.generate();
        ArrayList<Dag> dags = randomBN.setOfRandomDags;

        int i = 0;
        for (Dag dag : dags) {
            System.out.println("DAG " + i + ", Treewidth: " + getTreeWidth(dag));
            System.out.println(dag);
            System.out.println();
            i++;
        }

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
            saveRound(geneticUnion, minTW, meanTW, maxTW, tw, numNodes, nDags, popSize, nIterations, seed);
        }
    }


    public static void saveRound(GeneticTreeWidthUnion geneticUnion, int minTW, double meanTW, int maxTW, int tw, int numNodes, int nDags, int popSize, int nIterations, int seed) {
        String savePath = "./results/Server/" + numNodes + "_GeneticTWFusion_" + nDags + "_" + popSize + "_" + nIterations + "_" + seed + ".csv";
        String header = "numNodes,nDags,popSize,nIterations,seed,minTW,meanTW,maxTW,unionTW,limitTW,greedyTW,geneticTW,unionEdges,greedyEdges,geneticEdges,greedySMHD,geneticSMHD,timeUnion,timeGreedy,time\n";

        String line = numNodes + "," +
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
