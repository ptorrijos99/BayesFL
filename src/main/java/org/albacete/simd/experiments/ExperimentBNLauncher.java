package org.albacete.simd.experiments;

import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.Clustering;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.clustering.RandomClustering;
import org.albacete.simd.framework.BNBuilder;
import sun.misc.Signal;
import sun.misc.SignalHandler;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ExperimentBNLauncher {

    public static final int MAXITERATIONS = 100;

    private final String EXPERIMENTS_FOLDER;
    private int index;
    private String paramsFileName;
    private int threads;
    private ExperimentBNBuilder experiment;
    private String savepath;

    public ExperimentBNLauncher(int index, String paramsFileName, int threads, String saveFolder){
        this.index = index;
        this.paramsFileName = paramsFileName;
        this.threads = threads;
        this.EXPERIMENTS_FOLDER = saveFolder;
    }

    public static void main(String[] args) throws Exception {

        ExperimentBNLauncher experimentBNLauncher = getExperimentBNLauncherFromCommandLineArguments(args);
        String[] parameters = experimentBNLauncher.readParameters();

        System.out.println("Launching experiment");
        experimentBNLauncher.createExperiment(parameters);
        
        if (!experimentBNLauncher.checkExistentFile()){
            System.out.println("Starting experiment");
            experimentBNLauncher.runExperiment();
            experimentBNLauncher.saveExperiment();
            System.out.println("Experiment finished");
        }
        else{
            System.out.println("Experiment has already been done. Therefore, it has not been run again.");
        }
    }

    private static ExperimentBNLauncher getExperimentBNLauncherFromCommandLineArguments(String[] args) {
        int i = 1;
        System.out.println("Number of args: "  + args.length);
        for (String string : args) {
            System.out.println("Args " + i + ": " + string);
            i++;
        }
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];
        int threads = Integer.parseInt(args[2]);
        String saveFolder = args.length == 4 ? args[3] : "results/";
        
        return new ExperimentBNLauncher(index, paramsFileName, threads, saveFolder);
    }

    public String[] readParameters() throws Exception {
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

    private void createExperiment(String[] parameters){
        try {
            experiment = new ExperimentBNBuilder(parameters, threads);
        } catch (Exception e) {
            System.out.println("Exception when creating the experiment");
            int i=0;
            for (String string : parameters) {
                System.out.println("Param[" + i + "]: " + string);
                i++;
            }
            e.printStackTrace();
        }
    }
    
    private void runExperiment(){
        experiment.runExperiment();
    }

    private boolean checkExistentFile() throws IOException{
        String savePath = EXPERIMENTS_FOLDER  + "experiment_results_" + experiment.netName + "_" + experiment.algName + "_" + 
                experiment.databaseName + "_t" + experiment.numberOfThreads + "_PGESt" + experiment.numberOfPGESThreads +
                "_i" + experiment.interleaving + "_s" + experiment.seed + ".csv";
        
        return experiment.checkExistentFile(savePath);
    }

    private void saveExperiment() {
        String results = experiment.getResults();

        String savePath = EXPERIMENTS_FOLDER  + "experiment_results_" + experiment.netName + "_" + experiment.algName + "_" + 
                experiment.databaseName + "_t" + experiment.numberOfThreads + "_PGESt" + experiment.numberOfPGESThreads +
                "_i" + experiment.interleaving + "_s" + experiment.seed + ".csv";
        try {
            saveExperiment(savePath, results);
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error saving results at: " + savePath);
        }
    }
    
    public static void saveExperiment(String savePath, String results) throws IOException{
        File file = new File(savePath);
            BufferedWriter csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            //FileWriter csvWriter = new FileWriter(savePath, true);
            if(file.length() == 0) {
                String header = "algorithm,network,bbdd,threads,pges_threads,interleaving,seed,SHD,loglike,bdeu,deltaMM,deltaMM+,deltaMM-,iterations,time(s)\n";
                csvWriter.append(header);
            }
            csvWriter.append(results);

            csvWriter.flush();
            csvWriter.close();
            System.out.println("Results saved at: " + savePath);
    }
}
