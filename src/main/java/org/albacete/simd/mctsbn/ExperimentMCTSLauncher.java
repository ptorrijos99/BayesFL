package org.albacete.simd.mctsbn;

import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.DataReader;
import edu.cmu.tetrad.data.DelimiterType;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ExperimentMCTSLauncher {

    static String PATH = "./";

    public static void main(String[] args) throws Exception {
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];
        int threads = Integer.parseInt(args[2]);

        String[] parameters = readParameters(paramsFileName, index);

        String net = parameters[0];
        String bbdd = parameters[1];
        String algName = parameters[2];
        int iterationLimit = Integer.parseInt(parameters[3]);
        double exploitConstant = Double.parseDouble(parameters[4]);
        double numberSwaps = Double.parseDouble(parameters[5]);
        double probabilitySwap = Double.parseDouble(parameters[6]);

        String savePath = PATH + "results/experiment_" + net + "_mcts-" + algName + "_" +
                bbdd + "_t" + threads + "_it" + iterationLimit + "_ex" + exploitConstant
                + "_ps" + numberSwaps + "_ns" + probabilitySwap + ".csv";
        File file = new File(savePath);

        // Si no existe el fichero
        if(file.length() == 0) {
            Problem problem = new Problem(PATH + "res/networks/BBDD/" + net + "." + bbdd + ".csv");
            MCTSBN mctsbn = new MCTSBN(problem, iterationLimit, exploitConstant, numberSwaps, probabilitySwap, algName);

            double init = System.currentTimeMillis();
            Dag result = mctsbn.search();
            double time = (System.currentTimeMillis() - init)/1000.0;


            // Calculate scores
            MlBayesIm controlBayesianNetwork;
            try {
                controlBayesianNetwork = readOriginalBayesianNetwork(PATH + "res/networks/" + net + ".xbif");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            HillClimbingEvaluator hc = mctsbn.hc;

            Dag dagOriginal = new Dag(controlBayesianNetwork.getDag());
            ArrayList<Node> ordenOriginal = dagOriginal.getTopologicalOrder();
            ArrayList<Integer> ordenNuevosNodos = new ArrayList<>(ordenOriginal.size());
            for (Node node : ordenOriginal) {
                for (Node node2 : problem.getVariables()) {
                    if (node.getName().equals(node2.getName())) {
                        ordenNuevosNodos.add(problem.getHashIndices().get(node2));
                    }
                }
            }
            hc.setOrder(ordenNuevosNodos);
            hc.search();
            Dag hcDag = new Dag(hc.getGraph());
            double bdeuHCPerfect = GESThread.scoreGraph(hcDag, problem);
            double shdHCPerfect = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), hcDag);
            System.out.println("\n Best HC: \n    BDeu: " + bdeuHCPerfect + "\n    SMHD: " + shdHCPerfect);

            double bdeuOriginal = GESThread.scoreGraph(dagOriginal, problem);
            double shdOriginal = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), dagOriginal);
            System.out.println("\n Original: \n    BDeu: " + bdeuOriginal + "\n    SMHD: " + shdOriginal);

            Dag initialDag = new Dag(mctsbn.getInitializeDag());
            double bdeuPGES = GESThread.scoreGraph(initialDag, problem);
            double shdPGES = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), initialDag);
            System.out.println("\n PGES: \n    BDeu: " + bdeuPGES + "\n    SMHD: " + shdPGES);

            double bdeuMCTS = GESThread.scoreGraph(result, problem);
            double shdMCTS = Utils.SMHD(Utils.removeInconsistencies(controlBayesianNetwork.getDag()), result);
            System.out.println("\n MCTS: \n    BDeu: " + bdeuMCTS + "\n    SMHD: " + shdMCTS);

            file = new File(savePath);
            if(file.length() == 0) {
                BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
                String header = "algorithm,network,bbdd,threads,itLimit,exploitConst,numSwaps,probSwap,bdeuMCTS,shdMCTS,bdeuPGES,shdPGES,bdeuOrig,shdOrig,bdeuPerfect,shdPerfect,timePGES,time\n";
                csvWriter.append(header);

                String results = (algName + ","
                        + net + ","
                        + bbdd + ","
                        + threads + ","
                        + iterationLimit + ","
                        + exploitConstant + ","
                        + numberSwaps + ","
                        + probabilitySwap + ","
                        + bdeuMCTS + ","
                        + shdMCTS + ","
                        + bdeuPGES + ","
                        + shdPGES + ","
                        + bdeuOriginal + ","
                        + shdOriginal + ","
                        + bdeuHCPerfect + ","
                        + shdHCPerfect + ","
                        + (double) mctsbn.PGESTime + ","
                        + (double) time + "\n");
                csvWriter.append(results);
                csvWriter.flush();
                csvWriter.close();
            }
        }
        else {
            System.out.println("Experimento:  " + savePath + "    ya existente.");
        }
    }

    private static String getDatabaseNameFromPattern(String databasePath){
        // Matching the end of the csv file to get the name of the database
        Pattern pattern = Pattern.compile(".*/(.*).csv");
        Matcher matcher = pattern.matcher(databasePath);
        if (matcher.find()) {
            return matcher.group(1);
        }
        return null;
    }

    public static String[] readParameters(String paramsFileName, int index) throws Exception {
        String[] parameterStrings = null;
        try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
            String line;
            for (int i = 0; i < index; i++)
                br.readLine();
            line = br.readLine();
            parameterStrings = line.split(" ");
        }
        catch(FileNotFoundException e){
            System.out.println(e);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return parameterStrings;
    }

    private static MlBayesIm readOriginalBayesianNetwork(String netPath) throws Exception {
        BIFReader bayesianReader = new BIFReader();
        bayesianReader.processFile(netPath);
        BayesNet bayesianNet = bayesianReader;
        System.out.println("Numero de variables: " + bayesianNet.getNrOfNodes());

        //Transforming the BayesNet into a BayesPm
        BayesPm bayesPm = Utils.transformBayesNetToBayesPm(bayesianNet);
        MlBayesIm bn2 = new MlBayesIm(bayesPm);

        DataReader reader = new DataReader();
        reader.setDelimiter(DelimiterType.COMMA);
        reader.setMaxIntegralDiscrete(100);
        return bn2;
    }

}