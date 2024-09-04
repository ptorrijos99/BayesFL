package org.albacete.simd.mctsbn;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.utils.Problem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class MainHCEvaluator {

    public static void main(String[] args) {
        String networkFolder = "./res/networks/";
        String net_name = "pigs";
        String bbdd_path = networkFolder + "BBDD/" + net_name + ".3.csv";
        String net_path = networkFolder + net_name + ".xbif";

        int seed = 11231231;
        Problem problem = new Problem(bbdd_path);
        //List<Node> bestOrder = new ArrayList<>();
        //HypDistrib,DuctFlow,CardiacMixing,Disease,LungFlow,CO2Report,HypoxiaInO2,XrayReport,Age,LVH,LowerBodyO2,CO2,ChestXray,LVHreport,BirthAsphyxia,GruntingReport,Grunting,Sick,LungParench,RUQO2
        /*bestOrder.add(problem.getNode("BirthAsphyxia"));
        bestOrder.add(problem.getNode("Disease"));
        bestOrder.add(problem.getNode("Sick"));
        bestOrder.add(p roblem.getNode("DuctFlow"));
        bestOrder.add(problem.getNode("CardiacMixing"));
        bestOrder.add(problem.getNode("LungParench"));
        bestOrder.add(problem.getNode("LungFlow"));
        bestOrder.add(problem.getNode("LVH"));
        bestOrder.add(problem.getNode("Age"));
        bestOrder.add(problem.getNode("Grunting"));
        bestOrder.add(problem.getNode("HypDistrib"));
        bestOrder.add(problem.getNode("HypoxiaInO2"));
        bestOrder.add(problem.getNode("CO2"));
        bestOrder.add(problem.getNode("ChestXray"));
        bestOrder.add(problem.getNode("LVHreport"));
        bestOrder.add(problem.getNode("GruntingReport"));
        bestOrder.add(problem.getNode("LowerBodyO2"));
        bestOrder.add(problem.getNode("RUQO2"));
        bestOrder.add(problem.getNode("CO2Report"));
        bestOrder.add(problem.getNode("XrayReport"));
*/
        ConcurrentHashMap<String,Double> cache = new ConcurrentHashMap();
        long startTime1 = System.currentTimeMillis();
        for (int i = 0; i < 60; i++) {

            HillClimbingEvaluator hc = new HillClimbingEvaluator(problem, cache);
            
            List<Integer> order = new ArrayList<>(hc.nodeToIntegerList(problem.getVariables()));
            Random random = new Random(i);
            Collections.shuffle(order, random);
        
            System.out.println("\n\nThe order is: ");
            System.out.println(order);
            System.out.println();
            
            long startTime = System.currentTimeMillis();
            double score = hc.search();
            long endTime = System.currentTimeMillis();
            Dag resultingDag = new Dag(hc.getGraph());

            System.out.println("----------------------------------");
            System.out.println("FINISHED!");
            System.out.println("For the order: " + order);
            System.out.println("The score is: " + score);
            System.out.println("Number of edges: " + resultingDag.getEdges().size());
            //System.out.println("The resulting Dag is: \n" + resultingDag);
            System.out.println("Time: " + (endTime - startTime) / (double)1000 + " seconds");
            
        }
        
        System.out.println("Total time: " + (System.currentTimeMillis() - startTime1) / (double)1000 + " seconds");
    }

    public static String toStringOrder(List<Node> order){
        StringBuilder result = new StringBuilder();
        for (Node o: order) {
            result.append(o).append(" < ");
        }
        return result.toString();
    }   
}
