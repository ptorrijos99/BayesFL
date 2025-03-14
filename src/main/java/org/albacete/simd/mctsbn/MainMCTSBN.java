package org.albacete.simd.mctsbn;

import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;

import java.util.ArrayList;


public class MainMCTSBN {

    public static void main(String[] args) {
        String networkFolder = "./res/networks/";
        String net_name = "alarm";
        String bbdd_path = networkFolder + "BBDD/" + net_name + ".0.csv";
        String netPath = networkFolder + net_name + ".xbif";


        Problem problem = new Problem(bbdd_path);

        MCTSBN mctsbn = new MCTSBN(problem, 3000);

        mctsbn.EXPLOITATION_CONSTANT = 100;
        mctsbn.NUMBER_SWAPS = 0;
        mctsbn.PROBABILITY_SWAP = 0;
        mctsbn.initializeAlgorithm = "HC";

        long startTime = System.currentTimeMillis();
        addEndHook(mctsbn,startTime, netPath, problem);

        Dag result = mctsbn.search();
        //long endTime = System.currentTimeMillis();
        //double score = GESThread.scoreGraph(result, problem);

        System.out.println("MCTSBN FINISHED!");
        //System.out.println("Total time: " + (endTime - startTime)*1.0 / 1000);
        //System.out.println("Score: " + score);
        //System.out.println("Best Order");
        //System.out.println(toStringOrder(mctsbn.getBestOrder()));
        //System.out.println("Best Dag: ");
        //System.out.println(result);
    }


    public static void addEndHook(MCTSBN mctsbn, long startTime, String netPath, Problem problem){
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run(){
                try {
                    sleep(1000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                long endTime = System.currentTimeMillis();
                //save data here
                System.out.println("\n\n-------------------------------------");
                System.out.println("User shutdown...");
                System.out.println("-------------------------------------");
                System.out.println("Total time: " + (endTime - startTime)*1.0 / 1000);
                System.out.println("Best Order");
                System.out.println(mctsbn.getBestOrder());

                MlBayesIm controlBayesianNetwork;
                try {
                    controlBayesianNetwork = readOriginalBayesianNetwork(netPath);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                HillClimbingEvaluator hc = mctsbn.hc;

                Dag dagOriginal = new Dag(controlBayesianNetwork.getDag());
                ArrayList<Node> ordenOriginal = Utils.getTopologicalOrder(dagOriginal);
                ArrayList<Node> ordenOriginal2 = new ArrayList<>();
                System.out.println(ordenOriginal);
                ArrayList<Integer> ordenNuevosNodos = new ArrayList<>(ordenOriginal.size());
                for (Node node : ordenOriginal) {
                    for (Node node2 : problem.getVariables()) {
                        if (node.getName().equals(node2.getName())) {
                            ordenNuevosNodos.add(problem.getHashIndices().get(node2));
                            ordenOriginal2.add(problem.getNode(node2.getName()));
                        }
                    }
                }
                System.out.println(ordenOriginal2);
                System.out.println(ordenNuevosNodos);
                hc.setOrder(ordenNuevosNodos);
                hc.search();
                Dag hcDag = new Dag(hc.getGraph());
                double bdeu = GESThread.scoreGraph(hcDag, problem);
                double shd = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), hcDag);
                System.out.println("\n Best HC: \n    BDeu: " + bdeu + "\n    SHD: " + shd);

                bdeu = GESThread.scoreGraph(dagOriginal, problem);
                shd = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), dagOriginal);
                System.out.println("\n Original: \n    BDeu: " + bdeu + "\n    SHD: " + shd);

                Dag PGESdag = new Dag(mctsbn.getInitializeDag());
                bdeu = GESThread.scoreGraph(PGESdag, problem);
                shd = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), PGESdag);
                System.out.println("\n " + mctsbn.initializeAlgorithm + ": \n    BDeu: " + bdeu + "\n    SHD: " + shd);

                Dag mctsDag = new Dag(mctsbn.getBestDag());
                bdeu = GESThread.scoreGraph(mctsDag, problem);
                shd = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), mctsDag);
                System.out.println("\n MCTS: \n    BDeu: " + bdeu + "\n    SHD: " + shd);

            }
        });
    }

    public static MlBayesIm readOriginalBayesianNetwork(String netPath) throws Exception {
        BIFReader bayesianReader = new BIFReader();
        bayesianReader.processFile(netPath);
        BayesNet bayesianNet = bayesianReader;
        System.out.println("Numero de variables: " + bayesianNet.getNrOfNodes());

        //Transforming the BayesNet into a BayesPm
        BayesPm bayesPm = Utils.transformBayesNetToBayesPm(bayesianNet);
        MlBayesIm bn2 = new MlBayesIm(bayesPm);

        return bn2;
    }

}