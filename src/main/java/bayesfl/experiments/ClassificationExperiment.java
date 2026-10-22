package bayesfl.experiments;

import EBNC.wdBayes;
import bayesfl.experiments.utils.ExperimentUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.getClassificationMetrics;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;


public class ClassificationExperiment {
    public static String PATH = "./";

    public static void main(String[] args) {
        String[] parameters = readParametersFromArgs(args);

        String folder    = parameters[0];
        String bbdd      = parameters[1];
        int nClients     = Integer.parseInt(parameters[2]);
        int seed         = Integer.parseInt(parameters[3]);
        int folds        = Integer.parseInt(parameters[4]);
        int nBins        = Integer.parseInt(parameters[5]);
        String algorithm = parameters[6];
        int nTrees       = Integer.parseInt(parameters[7]);

        try {
            experimentBaseline(folder, bbdd, nClients, seed, folds, nBins, algorithm, nTrees);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static void experimentBaseline(String folder, String bbdd, int nClients, int seed, int nFolds, int nBins, String algorithm, int nTrees) throws Exception {
        String bbddPath = PATH + "res/classification/" + folder + "/" + bbdd + ".arff";
        String completePath = PATH + "results/baseline/" + bbdd + "_" + algorithm + "_" + nBins + "_" + nTrees + "_" + nFolds + "_" + nClients + "_" + seed + ".csv";
        String header = "bbdd,id,cv,algorithm,bins,seed,nTrees,nClients,iteration,instances,threads,trAcc,trPr,trRc,trF1,trTime,teAcc,tePr,teRc,teF1,teTime,time\n";

        int threads = Runtime.getRuntime().availableProcessors();

        // Parallel execution as posible, and seed
        String[] options = new String[4];
        options[0] = "-num-slots";
        options[1] = ""+threads;
        options[2] = "-S";
        options[3] = ""+seed;

        // Read the data and stratify it to the number of clients.
        // divide() expands nFolds=-1 into numInstances (LOO).
        Instances[][][] splits = divide(bbdd, bbddPath, nFolds, nClients, seed);
        int actualFolds = splits.length;

        // Initialize the filter
        Filter filter = null;
        if (nBins > 0) {
            filter = new weka.filters.unsupervised.attribute.Discretize();
            String[] filterOptions = new String[] {"-F", "-B", ""+nBins};
            filter.setOptions(filterOptions);
        } else {
            filter = new weka.filters.supervised.attribute.Discretize();
        }

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
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(new NaiveBayes());
            }
            case "GaussianNB" -> classifier = new NaiveBayes();
            case "A1DE" -> classifier = new A1DE();
            case "A2DE" -> classifier = new A2DE();
            // A1DE/A2DE wrapped with the configured Discretize filter (for continuous data).
            case "A1DE-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(new A1DE());
            }
            case "A2DE-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(new A2DE());
            }
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
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(new J48());
            }
            case "REPTree-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(new REPTree());
            }
            case "Bagging-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                Bagging bagging = new Bagging();
                bagging.setOptions(options);
                bagging.setNumIterations(nTrees);
                ((FilteredClassifier) classifier).setClassifier(bagging);
            }
            case "RF-Dis" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                Bagging rf = new RandomForest();
                rf.setOptions(options);
                rf.setNumIterations(nTrees);
                ((FilteredClassifier) classifier).setClassifier(rf);
            }

            // wdBayes classifier
            case "NBw" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "wCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
            case "NBd" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "dCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
            case "NBe" -> {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                wdBayes wd = new wdBayes();
                String[] algorithmOptions = new String[] {"-S", "NB", "-P", "eCCBN"};
                wd.setOptions(algorithmOptions);
                ((FilteredClassifier) classifier).setClassifier(wd);
            }
        }

        // Linear SVM (SMO, standardized inputs) and L1 logistic regression, on the original
        // attributes or, with "-Dis", on the discretized ones.
        if (algorithm.equals("SMO") || algorithm.equals("SMO-Dis") || algorithm.equals("L1LR") || algorithm.equals("L1LR-Dis")) {
            AbstractClassifier linear;
            if (algorithm.startsWith("SMO")) {
                weka.classifiers.functions.SMO smo = new weka.classifiers.functions.SMO();
                smo.setFilterType(new weka.core.SelectedTag(weka.classifiers.functions.SMO.FILTER_STANDARDIZE, weka.classifiers.functions.SMO.TAGS_FILTER));
                linear = smo;
            } else {
                linear = new bayesfl.experiments.utils.L1Logistic(1.0);
            }
            if (algorithm.endsWith("-Dis")) {
                classifier = new FilteredClassifier();
                ((FilteredClassifier) classifier).setFilter(filter);
                ((FilteredClassifier) classifier).setClassifier(linear);
            } else {
                classifier = linear;
            }
        }

        // "Mini<n>[-G][-S1|-S2|-A1|-A2][-C<c>]": MiniAnDE with the default configuration of the paper
        // (gamma = 0.2), Gaussian children with "-G", and the components combined by an
        // L1 ("-S1") or L2 ("-S2") logistic regression instead of averaged ("-A1", "-A2": only when that
        // regression beats the average in an internal cross-validation).
        java.util.regex.Matcher mini = java.util.regex.Pattern.compile("Mini([12])(-G)?(-[SA][12])?(-C[0-9.]+)?").matcher(algorithm);
        if (mini.matches()) {
            org.albacete.simd.mAnDE.mAnDE m = new org.albacete.simd.mAnDE.mAnDE();
            m.setN(Integer.parseInt(mini.group(1)));
            m.setnTrees(nTrees);
            m.setEnsemble("Bagging");
            m.setBagSize(100);
            m.setAddNB(0.2);
            if (mini.group(2) != null) {
                m.setContinuous(true);
                m.setCgPriorVarDof(3.0);
                m.setCgNBDiscretized(true);
            }
            if (mini.group(3) != null) m.setStack((mini.group(3).endsWith("1") ? "l1" : "l2") + (mini.group(3).startsWith("-A") ? "auto" : ""));
            if (mini.group(4) != null) m.setStackC(Double.parseDouble(mini.group(4).substring(2)));
            classifier = m;
        }

        // "FS<k>-NB", "FS<k>-A1DE", "FS<k>-A2DE", "FS<k>-TAN": discretize, keep the k attributes with the
        // highest information gain, and learn the classifier on them. Both filters are
        // fitted on the training fold only.
        java.util.regex.Matcher fs = java.util.regex.Pattern.compile("FS(\\d+)-(NB|A1DE|A2DE|TAN)").matcher(algorithm);
        if (fs.matches()) {
            weka.filters.supervised.attribute.AttributeSelection selection = new weka.filters.supervised.attribute.AttributeSelection();
            selection.setEvaluator(new weka.attributeSelection.InfoGainAttributeEval());
            weka.attributeSelection.Ranker ranker = new weka.attributeSelection.Ranker();
            ranker.setNumToSelect(Integer.parseInt(fs.group(1)));
            selection.setSearch(ranker);
            weka.filters.MultiFilter pipeline = new weka.filters.MultiFilter();
            pipeline.setFilters(new Filter[] {filter, selection});
            classifier = new FilteredClassifier();
            ((FilteredClassifier) classifier).setFilter(pipeline);
            ((FilteredClassifier) classifier).setClassifier(switch (fs.group(2)) {
                case "NB" -> new NaiveBayes();
                case "A1DE" -> new A1DE();
                case "A2DE" -> new A2DE();
                default -> {
                    BayesNet tan = new BayesNet();
                    tan.setSearchAlgorithm(new TAN());
                    yield tan;
                }
            });
        }

        // Resume: skip (fold, client) pairs already written by a previous run
        java.util.Set<String> done = completedPairs(completePath);

        // Repetitions of cross-validation
        for (int cv = 0; cv < actualFolds; cv++) {
            for (int i = 0; i < nClients; i++) {
                if (done.contains(cv + ":" + i)) continue;
                // Divide data in train and test
                Instances train = splits[cv][i][0];
                Instances test = splits[cv][i][1];

                double start = System.currentTimeMillis();
                classifier.buildClassifier(train);
                double time = (System.currentTimeMillis() - start) / 1000;

                String trainMetrics = getClassificationMetrics(classifier, train);
                String testMetrics = getClassificationMetrics(classifier, test);

                String results = bbdd + "," +
                        i + "," +
                        cv + "," +
                        algorithm + "," +
                        nBins + "," +
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

    /**
     * Returns the "cv:client" pairs already present in a result file and rewrites it
     * without the truncated line a killed job may leave behind.
     */
    private static java.util.Set<String> completedPairs(String path) {
        java.util.Set<String> done = new java.util.HashSet<>();
        java.io.File f = new java.io.File(path);
        if (!f.isFile()) return done;
        java.util.List<String> kept = new java.util.ArrayList<>();
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(f))) {
            String header = br.readLine();
            if (header == null) return done;
            kept.add(header);
            int nCols = header.split(",", -1).length;
            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = line.split(",", -1);
                if (cols.length != nCols) continue;
                String key = cols[2].trim() + ":" + cols[1].trim();
                if (done.add(key)) kept.add(line);
            }
        } catch (java.io.IOException e) {
            return new java.util.HashSet<>();
        }
        try (java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.FileWriter(f, false))) {
            for (String l : kept) pw.println(l);
        } catch (java.io.IOException e) {
            System.err.println("WARNING: could not rewrite " + f + " for resume: " + e.getMessage());
        }
        return done;
    }
}
