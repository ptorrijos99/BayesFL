package org.albacete.simd.experiments;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.Clustering;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;

import java.io.IOException;
import org.albacete.simd.algorithms.bnbuilders.Circular_GES;
import org.albacete.simd.algorithms.bnbuilders.Empty;
import org.albacete.simd.algorithms.bnbuilders.Fges_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.clustering.RandomClustering;

public class SimpleBNExperiment {


    public static void main(String[] args){
        // 1. Configuration
        String networkFolder = "./res/networks/";
        String net_name = "pathfinder";
        String net_path = networkFolder + net_name + ".xbif";
        String bbdd_path = networkFolder + "BBDD/" + net_name + ".xbif50003_.csv";
        DataSet ds = Utils.readData(bbdd_path);
        String test_path = networkFolder + "BBDD/tests/" + net_name + "_test.csv";

        // 2. Algorithm
        //BNBuilder algorithm = new GES_BNBuilder(bbdd_path);
        Clustering clustering = new HierarchicalClustering();
        //Clustering clustering = new RandomClustering();

        //BNBuilder algorithm = new PGESwithStages(ds, clustering, 4, 30, 10000, false, true, true);
        BNBuilder algorithm = new GES_BNBuilder(ds, true);
        //BNBuilder algorithm = new Circular_GES(ds, clustering, 4, 100000, "c4");
        //BNBuilder algorithm = new Fges_BNBuilder(ds);
        //BNBuilder algorithm = new Empty(ds);
        
        // Experiment
        ExperimentBNBuilder experiment = new ExperimentBNBuilder(algorithm, net_name, net_path, bbdd_path, test_path);//new ExperimentBNBuilder(algorithm, net_path, bbdd_path, test_path, 42);
        
        System.out.println("Alg Name: " + experiment.getAlgName());
        experiment.runExperiment();
        experiment.printResults();
        String savePath = "results/prueba.txt";
        
        /*BDeuScore bdeu = new BDeuScore(ds);
        Fges fges = new Fges(bdeu);
        System.out.println("Score FGES: " + fges.scoreDag(experiment.resultingBayesianNetwork));*/
        
        try {
            experiment.saveExperiment(savePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
