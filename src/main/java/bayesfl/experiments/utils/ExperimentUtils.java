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
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;

import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import static weka.core.Utils.maxIndex;
import static weka.core.Utils.mean;

public class ExperimentUtils {

    public static int experimentID = -1;

    public static void saveExperiment(String path, String header, String data) {
        // Create the directory if it does not exist
        File directory = new File(path.substring(0, path.lastIndexOf("/")));
        if (!directory.exists()){
            directory.mkdirs();
        }

        File file = new File(path);
        BufferedWriter csvWriter;
        try {
            csvWriter = new BufferedWriter(new FileWriter(path, true));
        
            if(file.length() == 0) {
                csvWriter.append(header);
            }

            csvWriter.append(data);
            csvWriter.flush();
            csvWriter.close();
            
        } catch (IOException ex) {
            Logger.getLogger(ExperimentUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

        // Save the name of the results csv on ./results/done/
        if (experimentID != -1) {
            String nameOfResultscsv = path.substring(path.lastIndexOf("/") + 1);
            try {
                // if "./results/done/experimentID.done" is empty, add a line
                File doneFile = new File("./results/done/" + experimentID + ".done");

                if (doneFile.length() == 0) {
                    BufferedWriter doneWriter = new BufferedWriter(new FileWriter(doneFile, true));
                    doneWriter.append(nameOfResultscsv + "\n");
                    doneWriter.flush();
                    doneWriter.close();
                }
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
                BDeuScore bdeu = new BDeuScore(dat.getData());
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
                    MlBayesIm originalBN = readOriginalBayesianNetwork(dat.getOriginalBNPath());
                    return Utils.SMHD(Utils.removeInconsistencies(originalBN.getDag()), dag);
                } catch (Exception e) { e.printStackTrace(); }
            }
        }
        throw new IllegalArgumentException("The data must be object of the BN_DataSet class");
    }

    /**
     * Get the metrics of the model. The metrics are accuracy, precision, recall, F1-score, and prediction time.
     *
     * @param instances The instances.
     * @return The metrics of the model in the form of a string.
     */
    public static String getClassificationMetrics(AbstractClassifier classifier, Instances instances) {
        int numClasses = instances.numClasses();

        Evaluation evaluation;
        try {
            evaluation = new Evaluation(instances);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Initialize the start time of the evaluation
        double time = System.currentTimeMillis();

        try {
            evaluation.evaluateModel(classifier, instances);
        }
        catch (Exception e) {
            e.printStackTrace();
        }

        // Get the time of the evaluation and convert it to seconds
        time = System.currentTimeMillis() - time;
        time /= 1000;

        double accuracy = evaluation.pctCorrect() / 100;
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
     * Read the original Bayesian Network from the BIF file in the netPath.
     * @return The original Bayesian Network.
     * @throws Exception If the file is not found.
     */
    private static MlBayesIm readOriginalBayesianNetwork(String netPath) throws Exception {
        final PrintStream err = new PrintStream(System.err);
        System.setErr(new PrintStream(OutputStream.nullOutputStream()));

        BIFReader bayesianReader = new BIFReader();
        bayesianReader.processFile(netPath);

        System.setErr(err);

        // Transforming the BayesNet into a BayesPm
        BayesPm bayesPm = Utils.transformBayesNetToBayesPm(bayesianReader);

        return new MlBayesIm(bayesPm);
    }
}
