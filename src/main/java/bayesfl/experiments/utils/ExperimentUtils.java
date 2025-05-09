/*
 *  The MIT License (MIT)
 *  
 *  Copyright (c) 2022 Universidad de Castilla-La Mancha, España
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */
/**
 *    ExperimentUtils.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package bayesfl.experiments.utils;

import bayesfl.data.BN_DataSet;
import bayesfl.data.Data;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.score.BdeuScore;
import edu.cmu.tetrad.search.Fges;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.*;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ExperimentUtils {

    public static int experimentID = -1;

    private static final Object fileLock = new Object();

    public static void saveExperiment(String path, String header, String data) {
        // Create the directory if it does not exist
        File directory = new File(path.substring(0, path.lastIndexOf("/")));
        if (!directory.exists()){
            directory.mkdirs();
        }

        File file = new File(path);
        synchronized (fileLock) {
            try (BufferedWriter csvWriter = new BufferedWriter(new FileWriter(file, true))) {
                if (file.length() == 0) {
                    csvWriter.append(header);
                }
                csvWriter.append(data);
            } catch (IOException ex) {
                Logger.getLogger(ExperimentUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    public static String[] readParametersFromArgs(String[] args) {
        int i=0;
        for (String string : args) {
            System.out.println("arg[" + i + "]: " + string);
            i++;
        }
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];

        ExperimentUtils.experimentID = index;

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

        return parameters;
    }
    
    public static double calculateBDeuGESThread(Data data, Dag dag) {
        if ((data instanceof BN_DataSet dat)) {
            Problem problem = dat.getProblem();
            if (dat.getProblem() != null) {
                return GESThread.scoreGraph(dag, problem);
            } 
        }
        return -1;
    }
    
    public static double calculateBDeu(Data data, Dag dag) {
        if ((data instanceof BN_DataSet dat)) {
            if (dat.getData() != null) {
                BdeuScore bdeu = new BdeuScore(dat.getData());
                Fges fges = new Fges(bdeu);
                return fges.scoreDag(dag);
            } 
        }
        return -1;
    }
    
    public static int calculateSMHD(Data data, Dag dag) {
        if ((data instanceof BN_DataSet dat)) {
            if (dat.getOriginalBNPath() != null) {
                try {
                    BayesPm originalBN = readOriginalBayesianNetwork(dat.getOriginalBNPath());
                    return Utils.SMHD(Utils.removeInconsistencies(originalBN.getDag()), dag);
                } catch (Exception e) { e.printStackTrace(); }
            }
        }
        return -1;
    }

    public static int calculateSHD(Data data, Dag dag) {
        if ((data instanceof BN_DataSet dat)) {
            if (dat.getOriginalBNPath() != null) {
                try {
                    BayesPm originalBN = readOriginalBayesianNetwork(dat.getOriginalBNPath());
                    return Utils.SHD(Utils.removeInconsistencies(originalBN.getDag()), dag);
                } catch (Exception e) { e.printStackTrace(); }
            }
        }
        return -1;
    }

    public static int calculateFusSim(Data data, Dag dag) {
        if ((data instanceof BN_DataSet dat)) {
            if (dat.getOriginalBNPath() != null) {
                try {
                    BayesPm originalBN = readOriginalBayesianNetwork(dat.getOriginalBNPath());
                    return Utils.fusionSimilarity(Utils.removeInconsistencies(originalBN.getDag()), dag);
                } catch (Exception e) { e.printStackTrace(); }
            }
        }
        return -1;
    }

    /**
     * Get the metrics of the model. The metrics are accuracy, precision, recall, F1-score, and prediction time.
     *
     * @param instances The instances.
     * @return The metrics of the model in the form of a string.
     */
    public static String getClassificationMetrics(AbstractClassifier classifier, Instances instances) {
        Evaluation evaluation;
        try {
            evaluation = new Evaluation(instances);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Initialize the start time of the evaluation
        double startTime = System.currentTimeMillis();

        try {
            evaluation.evaluateModel(classifier, instances);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Get the time of the evaluation and convert it to seconds
        double time = (System.currentTimeMillis() - startTime) / 1000.0;

        int numClasses = instances.numClasses();
        double accuracy = evaluation.pctCorrect() / 100.0;
        double f1 = 0.0;
        double precision = 0.0;
        double recall = 0.0;
        double metric;

        for (int i = 0; i < numClasses; i++) {
            // If the classifier predicts no instances of a class
            // or they not exist, the metrics must be set to zero
            metric = evaluation.precision(i);
            precision += Double.isNaN(metric) ? 0 : metric;
            metric = evaluation.recall(i);
            recall += Double.isNaN(metric) ? 0 : metric;
            metric = evaluation.fMeasure(i);
            f1 += Double.isNaN(metric) ? 0 : metric;
        }

        // Compute the macro average of the metrics
        precision /= numClasses;
        recall /= numClasses;
        f1 /= numClasses;

        return accuracy + "," + precision + "," + recall + "," + f1 + "," + time + ",";
    }

    /**
     * Get the metrics of a ensemble model. The metrics are accuracy, precision, recall, F1-score, and prediction time.
     *
     * @param ensemble The ensemble of classifiers.
     * @param syntheticClassMaps The synthetic class maps.
     * @param instances The instances.
     * @return The metrics of the model in the form of a string.
     */
    public static String getClassificationMetricsEnsemble(List<AbstractClassifier> ensemble, List<Map<String, Integer>> syntheticClassMaps, Instances instances) {
        Evaluation evaluation;
        try {
            evaluation = new Evaluation(instances);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Initialize the start time of the evaluation
        double startTime = System.currentTimeMillis();

        try {
            EnsembleClassifier ensembleClassifier = new EnsembleClassifier(ensemble, syntheticClassMaps);
            ensembleClassifier.buildClassifier(instances);
            evaluation.evaluateModel(ensembleClassifier, instances);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Get the time of the evaluation and convert it to seconds
        double time = (System.currentTimeMillis() - startTime) / 1000.0;

        int numClasses = instances.numClasses();
        double accuracy = evaluation.pctCorrect() / 100.0;
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        double metric;

        for (int i = 0; i < numClasses; i++) {
            metric = evaluation.precision(i);
            precision += Double.isNaN(metric) ? 0 : metric;
            metric = evaluation.recall(i);
            recall += Double.isNaN(metric) ? 0 : metric;
            metric = evaluation.fMeasure(i);
            f1 += Double.isNaN(metric) ? 0 : metric;
        }

        precision /= numClasses;
        recall /= numClasses;
        f1 /= numClasses;

        return accuracy + "," + precision + "," + recall + "," + f1 + "," + time + ",";
    }


    /**
     * Read the original Bayesian Network from the BIF file in the netPath.
     * @return The original Bayesian Network.
     * @throws Exception If the file is not found.
     */
    private static BayesPm readOriginalBayesianNetwork(String netPath) throws Exception {
        final PrintStream err = new PrintStream(System.err);
        System.setErr(new PrintStream(OutputStream.nullOutputStream()));

        BIFReader bayesianReader = new BIFReader();
        bayesianReader.processFile(netPath);

        System.setErr(err);

        // Transforming the BayesNet into a BayesPm
        BayesPm bayesPm = Utils.transformBayesNetToBayesPm(bayesianReader);

        //return new MlBayesIm(bayesPm);
        return bayesPm;
    }

    public static Dag readDagFromMatrix(double[][] matrix, List<Node> nodes) {
        Dag dag = new Dag(nodes);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > 0) {
                    Edge edge = new Edge(nodes.get(i), nodes.get(j), Endpoint.TAIL, Endpoint.ARROW);
                    dag.addEdge(edge);
                }
            }
        }
        return dag;
    }

    public static Graph readGraphFromMatrix(double[][] matrix, List<Node> nodes) {
        Graph dag = new EdgeListGraph(nodes);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > 0) {
                    if (matrix[j][i] > 0 && i < j) {
                        Edge edge = new Edge(nodes.get(i), nodes.get(j), Endpoint.TAIL, Endpoint.TAIL);
                        dag.addEdge(edge);
                    } else {
                        Edge edge = new Edge(nodes.get(i), nodes.get(j), Endpoint.TAIL, Endpoint.ARROW);
                        dag.addEdge(edge);
                    }
                }
            }
        }
        return dag;
    }
}


class EnsembleClassifier extends AbstractClassifier {

    private final List<AbstractClassifier> ensemble;
    private final List<Map<String, Integer>> syntheticClassMaps;
    private Instances header;
    private int numClasses;

    public EnsembleClassifier(List<AbstractClassifier> ensemble, List<Map<String, Integer>> syntheticClassMaps) {
        this.ensemble = ensemble;
        this.syntheticClassMaps = syntheticClassMaps;
    }

    @Override
    public void buildClassifier(Instances data) {
        this.header = data;
        this.numClasses = data.classAttribute().numValues();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] yVotes = new double[numClasses];
        ArrayList<Object> classValues = Collections.list(header.classAttribute().enumerateValues());

        for (int j = 0; j < ensemble.size(); j++) {
            AbstractClassifier clf = ensemble.get(j);
            Map<String, Integer> classMap = syntheticClassMaps.get(j);
            ArrayList<String> synthValues = new ArrayList<>(classMap.keySet());

            Instances syntheticHeader = new Instances(header, 0);
            syntheticHeader.setClassIndex(-1);
            syntheticHeader.deleteAttributeAt(header.classIndex());
            syntheticHeader.insertAttributeAt(new Attribute("synthetic_class", synthValues), syntheticHeader.numAttributes());
            syntheticHeader.setClassIndex(syntheticHeader.numAttributes() - 1);

            Instance synthetic = new weka.core.DenseInstance(syntheticHeader.numAttributes());
            synthetic.setDataset(syntheticHeader);

            int aIdx = 0;
            for (int a = 0; a < instance.numAttributes(); a++) {
                if (a == header.classIndex()) continue;
                synthetic.setValue(aIdx++, instance.value(a));
            }

            double[] dist = clf.distributionForInstance(synthetic);
            double[] syntheticVotes = new double[numClasses];
            double sum = 0;

            for (int s = 0; s < dist.length; s++) {
                String synthLabel = synthValues.get(s);
                String origLabel = synthLabel.contains("|||")
                        ? synthLabel.substring(synthLabel.lastIndexOf("|||") + 3)
                        : synthLabel;
                int y = classValues.indexOf(origLabel);
                syntheticVotes[y] += dist[s];
                sum += dist[s];
            }

            if (sum > 0) {
                for (int y = 0; y < numClasses; y++) {
                    yVotes[y] += syntheticVotes[y] / sum;
                }
            }
        }

        // Normalize votes
        double total = Arrays.stream(yVotes).sum();
        if (total > 0) {
            for (int i = 0; i < yVotes.length; i++) {
                yVotes[i] /= total;
            }
        }

        return yVotes;
    }
}

