package bayesfl.experiments;

import EBNC.wdBayes;
import bayesfl.experiments.utils.ExperimentUtils;
import org.apache.commons.math3.ode.nonstiff.AdaptiveStepsizeFieldIntegrator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.supervised.attribute.Discretize;

import java.util.Random;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.getClassificationMetrics;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;


public class classificationExperiment {
    public static String PATH = "./";

    public static void main(String[] args) {
        String[] parameters = readParametersFromArgs(args);

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
        String folder = "Nuevas";
        String bbdd = "Amazon_Employee_Access";
        int nClients = 5;
        String algorithm = "NB";
        int seed = 1;
        int folds = 5;
        int nTrees = 100;

        try {
            experimentBaseline(folder, bbdd, nClients, seed, folds, algorithm, nTrees);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }*/

    public static void experimentBaseline(String folder, String bbdd, int nClients, int seed, int nFolds, String algorithm, int nTrees) throws Exception {
        String bbddPath = PATH + "res/classification/" + folder + "/" + bbdd + ".arff";
        String completePath = PATH + "results/baseline/" + bbdd + "_" + algorithm + "_" + nTrees + "_" + nFolds + "_" + nClients + "_" + seed + ".csv";
        String header = "bbdd,id,cv,algorithm,seed,nTrees,nClients,iteration,instances,threads,trAcc,trPr,trRc,trF1,timeTrain,teAcc,tePr,teRc,teF1,timeTest,time\n";

        int threads = Runtime.getRuntime().availableProcessors();

        // Parallel execution as posible, and seed
        String[] options = new String[4];
        options[0] = "-num-slots";
        options[1] = ""+threads;
        options[2] = "-S";
        options[3] = ""+seed;

        // Read the data and stratify it to the number of clients
        Instances[][][] splits = divide(bbdd, bbddPath, nFolds, nClients, seed);

        // Initialize the classifier
        AbstractClassifier classifier = null;
        switch (algorithm) {
            // WEKA classifiers
            case "J48" -> classifier = new J48();
            case "REPTree" -> classifier = new REPTree();
            case "Bagging" -> {
                classifier = new Bagging();
                classifier.setOptions(options);
                ((Bagging) classifier).setNumIterations(nTrees);
            }
            case "RF" -> {
                classifier = new RandomForest();
                classifier.setOptions(options);
                ((RandomForest) classifier).setNumIterations(nTrees);
            }
            case "NB" -> {
                classifier = new NaiveBayes();
                ((NaiveBayes) classifier).setUseSupervisedDiscretization(true);
            }
            case "GaussianNB" -> classifier = new NaiveBayes();
            case "A1DE" -> classifier = new A1DE();
            case "A2DE" -> classifier = new A2DE();
            case "TAN" -> {
                classifier = new BayesNet();
                TAN alg = new TAN();
                ((BayesNet) classifier).setSearchAlgorithm(alg);
            }
            case "KNN" -> {
                classifier = new IBk();
                ((IBk) classifier).setKNN(nTrees);
            }

            // WEKA classifiers with discretization
            case "J48-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                ((FilteredClassifier) classifier).setClassifier(new J48());
            }
            case "REPTree-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                ((FilteredClassifier) classifier).setClassifier(new REPTree());
            }
            case "Bagging-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                Bagging bagging = new Bagging();
                bagging.setOptions(options);
                bagging.setNumIterations(nTrees);
                ((FilteredClassifier) classifier).setClassifier(bagging);
            }
            case "RF-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                Bagging rf = new RandomForest();
                rf.setOptions(options);
                rf.setNumIterations(nTrees);
                ((FilteredClassifier) classifier).setClassifier(rf);
            }

            // wdBayes classifier
            case "NBw" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "wCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
            case "NBd" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "dCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
            case "NBe" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(new Discretize());
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "eCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
        }

        // Repetitions of cross-validation
        for (int cv = 0; cv < nFolds; cv++) {
            for (int i = 0; i < nClients; i++) {
                // Divide data in train and test
                Instances train = splits[cv][i][0];
                Instances test = splits[cv][i][1];

                double start = System.currentTimeMillis();
                classifier.buildClassifier(train);
                double time = (System.currentTimeMillis() - start) / 1000;

                Evaluation evaluation = null;
                try {
                    evaluation = new Evaluation(train);
                }
                catch (Exception e) {
                    e.printStackTrace();
                }

                String trainMetrics = getClassificationMetrics(classifier, evaluation, train);

                String testMetrics = getClassificationMetrics(classifier, evaluation, test);

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

                        trainMetrics +
                        testMetrics +

                        time + "\n";

                System.out.println(results);

                ExperimentUtils.saveExperiment(completePath, header, results);
            }
        }


    }
}
