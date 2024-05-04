package bayesfl.experiments;

import bayesfl.data.CPT_Instances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import static bayesfl.experiments.ExperimentUtils.getClassificationMetrics;


public class classificationBaselinesExperiment {
    public static String PATH = "./";

    public static void main(String[] args) {
        int i=0;
        for (String string : args) {
            System.out.println("arg[" + i + "]: " + string);
            i++;
        }
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];
        int threads = Integer.parseInt(args[2]);

        // Read the parameters from args
        String[] parameters = null;
        try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
            String line;
            for (i = 0; i < index; i++)
                br.readLine();
            line = br.readLine();
            parameters = line.split(" ");
        }
        catch(Exception e){ System.out.println(e); }

        System.out.println("Number of hyperparams: " + parameters.length);
        i=0;
        for (String string : parameters) {
            System.out.println("Param[" + i + "]: " + string);
            i++;
        }

        // Read the parameters from file
        String folder = parameters[0];
        String bbdd = parameters[1];
        int nClients = Integer.parseInt(parameters[2]);
        int seed = Integer.parseInt(parameters[3]);
        int folds = Integer.parseInt(parameters[4]);
        String algorithm = parameters[5];
        int nTrees = Integer.parseInt(parameters[6]);

        try {
            experimentBaseline(folder, bbdd, nClients, seed, folds, algorithm, nTrees);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /*public static void main(String[] args) {
        String folder = "AnDE";
        String bbdd = "Iris_Classification";
        int nClients = 5;
        String algorithm = "RF";
        int seed = 42;
        int folds = 5;
        int nTrees = 100;

        try {
            experimentBaseline(folder, bbdd, nClients, seed, folds, algorithm, nTrees);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }*/

    public static void experimentBaseline(String folder, String bbdd, int nClients, int seed, int folds, String algorithm, int nTrees) throws Exception {
        String bbddPath = PATH + "res/classification/" + folder + "/" + bbdd + ".arff";
        String completePath = PATH + "results/baseline/" + bbdd + "_" + algorithm + "_" + nTrees + "_" + folds + "_" + nClients + "_" + seed + ".csv";
        String header = "bbdd,id,cv,algorithm,seed,nTrees,nClients,iteration,instances,threads,trAcc,trPr,trRc,trF1,teAcc,tePr,teRc,teF1,time,timeTrain,timeTest\n";

        int threads = Runtime.getRuntime().availableProcessors();

        Random random = new Random(seed);
        // Parallel execution as posible, and seed
        String[] options = new String[4];
        options[0] = "-num-slots";
        options[1] = ""+threads;
        options[2] = "-S";
        options[3] = ""+seed;

        // Read the data and stratify it to the number of clients
        Instances allData = CPT_Instances.readData(bbddPath);

        // Stratify each of the divisions
        ArrayList<Instances> divisionData = CPT_Instances.divideDataSet(allData, nClients, folds, random);
        System.out.println(" N instances divided: " + allData.numInstances() + " N clients: " + nClients + " N folds: " + folds);

        // Initialize the classifier
        AbstractClassifier classifier = null;
        switch (algorithm) {
            case "J48":
                classifier = new J48();
                break;
            case "REPTree":
                classifier = new REPTree();
                break;
            case "Bagging":
                classifier = new Bagging();
                classifier.setOptions(options);
                ((Bagging)classifier).setNumIterations(nTrees);
                break;
            case "RF":
                classifier = new RandomForest();
                classifier.setOptions(options);
                ((RandomForest)classifier).setNumIterations(nTrees);
                break;
            case "NB":
                classifier = new NaiveBayes();
                break;
            case "A1DE":
                classifier = new A1DE();
                break;
            case "A2DE":
                classifier = new A2DE();
                break;
            case "TAN":
                classifier = new BayesNet();
                TAN alg = new TAN();
                ((BayesNet)classifier).setSearchAlgorithm(alg);
                break;
            case "KNN":
                classifier = new IBk();
                ((IBk)classifier).setKNN(nTrees);
                break;
        }

        // Repetitions of cross-validation
        for (int cv = 0; cv < folds; cv++) {
            for (int i = 0; i < nClients; i++) {
                // Divide data in train and test
                System.out.println("Client: " + i + " Fold: " + cv + " Train: " + divisionData.get(i).numInstances());

                Instances train = divisionData.get(i).trainCV(folds, cv, random);
                Instances test = divisionData.get(i).testCV(folds, cv);

                double start = System.currentTimeMillis();
                classifier.buildClassifier(train);
                double time = (System.currentTimeMillis() - start) / 1000;

                start = System.currentTimeMillis();
                double[] trainMetrics = getClassificationMetrics(classifier, train);
                double timeTrain = (System.currentTimeMillis() - start) / 1000;

                start = System.currentTimeMillis();
                double[] testMetrics = getClassificationMetrics(classifier, test);
                double timeTest = (System.currentTimeMillis() - start) / 1000;

                String results = bbdd + "," +
                        i + "," +
                        cv + "," +
                        algorithm + "," +
                        seed + "," +
                        nTrees + "," +
                        nClients + "," +
                        "1," +
                        train.numInstances() + "," +
                        threads + "," +

                        trainMetrics[0] + "," +
                        trainMetrics[1] + "," +
                        trainMetrics[2] + "," +
                        trainMetrics[3] + "," +
                        testMetrics[0] + "," +
                        testMetrics[1] + "," +
                        testMetrics[2] + "," +
                        testMetrics[3] + "," +

                        time + "," +
                        timeTrain + "," +
                        timeTest + "\n";

                System.out.println(results);

                ExperimentUtils.saveExperiment(completePath, header, results);
            }
        }


    }
}
