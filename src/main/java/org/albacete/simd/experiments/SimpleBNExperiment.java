package org.albacete.simd.experiments;

import edu.cmu.tetrad.data.DataSet;

import org.albacete.simd.algorithms.bnbuilders.*;
import org.albacete.simd.clustering.*;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;

import java.io.IOException;

public class SimpleBNExperiment {


    public static void main(String[] args){
        // 1. Configuration
        String networkFolder = "./res/networks/";
        String net_name = "hailfinder";
        String net_path = networkFolder + net_name + ".xbif";

        String bbdd_path = networkFolder + "BBDD/" + net_name + "/" + net_name + ".0.csv";
        //String bbdd_path = networkFolder + "BBDD_old/" + net_name + ".0.csv";

        DataSet ds = Utils.readData(bbdd_path);

        // 2. Algorithm
        //BNBuilder algorithm = new GES_BNBuilder(bbdd_path);
        Clustering clustering = new HierarchicalClustering();
        //Clustering clustering = new RandomClustering();

        //BNBuilder algorithm = new PGESwithStages(ds, clustering, 4, 30, 10000, false, true, true);
        BNBuilder algorithm = new GES_BNBuilder(ds, true);
        //BNBuilder algorithm = new Circular_GES(ds, clustering, 4, 100000, "c4");
        //BNBuilder algorithm = new Fges_BNBuilder(ds, false);
        //BNBuilder algorithm = new Empty(ds);
        
        // Experiment
        ExperimentBNBuilder experiment = new ExperimentBNBuilder(algorithm, net_name, net_path, bbdd_path, bbdd_path);//new ExperimentBNBuilder(algorithm, net_path, bbdd_path, test_path, 42);
        
        System.out.println("Alg Name: " + experiment.getAlgName());
        experiment.runExperiment();
        experiment.printResults();
        String savePath = "results/prueba.txt";
        
        /*BdeuScore bdeu = new BdeuScore(ds);
        Fges fges = new Fges(bdeu);
        System.out.println("Score FGES: " + fges.scoreDag(experiment.resultingBayesianNetwork));*/
        
        try {
            experiment.saveExperiment(savePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
