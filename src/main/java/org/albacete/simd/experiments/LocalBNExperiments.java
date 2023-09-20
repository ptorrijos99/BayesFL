package org.albacete.simd.experiments;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.Clustering;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.framework.BNBuilder;

import org.albacete.simd.algorithms.bnbuilders.Circular_GES;
import org.albacete.simd.algorithms.bnbuilders.Empty;
import org.albacete.simd.algorithms.bnbuilders.Fges_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;

public class LocalBNExperiments {
    
    protected static BNBuilder algorithm;
    protected static String EXPERIMENTS_FOLDER = "resultados/";
    
    public static void main(String[] args) throws Exception {
        // 1. Configuration
        String networkFolder = "./res/networks/";

        // 2. Algorithm
        //String[] algorithmsPGES = new String[]{"pges", "pges update", "pges update speed", "pges-jc", "pges-jc update", "pges-jc update speed"};  // "pges update noParallel", 
        String[] algorithmsPGES = new String[]{"circular_ges_c1", "circular_ges_c2", "circular_ges_c3", "circular_ges_c4"};
        String[] bbdds = new String[]{"alarm", "barley", "child", "hepar2", "insurance", "mildew", "water", "win95pts", "andes", "pigs", "link", "diabetes", "pathfinder", "hailfinder", "munin"};

        Integer[] nThreads = new Integer[]{8,4};
        String[] bbdd_patchs = new String[]{"50003_", "50002_", "50001_", "50004_","50005_", "50006_", "50007_",
                                            "50008_", "50009_", "50001246_", "_"};

        for (int threads : nThreads) {
            for (String net_name : bbdds) {
                for (String bb : bbdd_patchs) {
                    String net_path = networkFolder + net_name + ".xbif";
                    String bbdd_path = networkFolder + "BBDD/" + net_name + ".xbif" + bb + ".csv";
                    String test_path = networkFolder + "BBDD/tests/" + net_name + "_test.csv";

                    //launchExperiment("empty", net_name, net_path, bbdd_path, test_path, 1, -1, bb);

                    /*launchExperiment("ges", net_name, net_path, bbdd_path, test_path, 1, -1, bb);
                    System.gc();
                    launchExperiment("ges parallel", net_name, net_path, bbdd_path, test_path, 1, -1, bb);
                    System.gc();*/
                    for (String alg : algorithmsPGES) {
                        try {
                            launchExperiment(alg, net_name, net_path, bbdd_path, test_path, threads, Integer.MAX_VALUE, bb);
                        } catch(Exception ex) {System.out.println(ex);}
                        System.gc();
                    }
                    /*launchExperiment("fges", net_name, net_path, bbdd_path, test_path, 1, -1, bb);
                    System.gc();
                    launchExperiment("fges-faithfulness", net_name, net_path, bbdd_path, test_path, 1, -1, bb);
                    System.gc();*/
                }
            }
        }
    }
    
    private static void launchExperiment(String algName, String net_name, String net_path, String bbdd_path, String test_path, int numberOfPGESThreads, int interleaving, String database) throws Exception {
        Clustering clustering;
        
        System.out.println("\n\n\n----------------------------------------------------------------------------- \n"
                    + "Alg Name: " + algName + ""
                            + "\n-----------------------------------------------------------------------------");
        
        String savePath = EXPERIMENTS_FOLDER  + "experiment_results_" + net_name + "_" + algName + "_" + 
                net_name + ".xbif" + database + "_t8_PGESt" + numberOfPGESThreads +
                "_i" + interleaving + "_s-1.csv";
        
        if ((!checkExistentFile(savePath))) { 
                
            switch(algName) {
                case "pges":
                    clustering = new HierarchicalClustering(true, false);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, false, true);
                    break;
                case "pges update":
                    clustering = new HierarchicalClustering(true, false);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, true, true);
                    break;
                case "pges update noParallel":
                    clustering = new HierarchicalClustering(true, false);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, true, false);
                    break;
                case "pges update speed":
                    clustering = new HierarchicalClustering(true, false);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, true, true, true);
                    break;

                case "pges-jc":
                    clustering = new HierarchicalClustering(true, true);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, false, true);
                    break;
                case "pges-jc update":
                    clustering = new HierarchicalClustering(true, true);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, true, true);
                    break;
                case "pges-jc update noParallel":
                    clustering = new HierarchicalClustering(true, true);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, false, true, false);
                    break;
                case "pges-jc update speed":
                    clustering = new HierarchicalClustering(true, true);
                    algorithm = new PGESwithStages(bbdd_path, clustering, numberOfPGESThreads, ExperimentBNLauncher.MAXITERATIONS, interleaving, true, true, true);
                    break;


                case "ges":
                    algorithm = new GES_BNBuilder(bbdd_path, false);
                    break;
                case "ges parallel":
                    algorithm = new GES_BNBuilder(bbdd_path, true);
                    break;

                    
                case "circular_ges_c1":
                    clustering = new HierarchicalClustering();
                    algorithm = new Circular_GES(bbdd_path, clustering, numberOfPGESThreads, interleaving, "c1");
                    break;
                case "circular_ges_c2":
                    clustering = new HierarchicalClustering();
                    algorithm = new Circular_GES(bbdd_path, clustering, numberOfPGESThreads, interleaving, "c2");
                    break;
                case "circular_ges_c3":
                    clustering = new HierarchicalClustering();
                    algorithm = new Circular_GES(bbdd_path, clustering, numberOfPGESThreads, interleaving, "c3");
                    break;
                case "circular_ges_c4":
                    clustering = new HierarchicalClustering();
                    algorithm = new Circular_GES(bbdd_path, clustering, numberOfPGESThreads, interleaving, "c4");
                    break;


                case "fges":
                    algorithm = new Fges_BNBuilder(bbdd_path, true, false);
                    break;
                case "fges-faithfulness":
                    algorithm = new Fges_BNBuilder(bbdd_path, false, false);
                    break;
                    
                case "empty":
                    algorithm = new Empty(bbdd_path);
                    break;


                default:
                    throw new Exception("Error... Algoritmo incorrecto: " + algName);
            }

            // Experiment
            ExperimentBNBuilder experiment = new ExperimentBNBuilder(algorithm, net_name, net_path, bbdd_path, test_path);
            experiment.algName = algName;

            experiment.runExperiment();
            experiment.printResults();

            try {
                experiment.saveExperiment(savePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("\n EXPERIMENTO YA EXISTENTE: " + savePath + "\n");
        }
        
    }
    
    public static boolean checkExistentFile(String savePath) throws IOException {
        File file = new File(savePath);
        
        boolean check = file.length() != 0;
        
        BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
        csvWriter.append(" ");
        csvWriter.flush();
        csvWriter.close();

        return check;
    }

}
