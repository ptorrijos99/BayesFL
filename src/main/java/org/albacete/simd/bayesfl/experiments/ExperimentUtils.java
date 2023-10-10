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

package org.albacete.simd.bayesfl.experiments;

import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.search.BDeuScore;
import edu.cmu.tetrad.search.Fges;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.albacete.simd.bayesfl.data.BN_DataSet;
import org.albacete.simd.bayesfl.data.Data;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

public class ExperimentUtils {
    
    final static String SAVEPATH = "results/";
    
    public static void saveExperiment(String restPath, String header, String data) {
        String path = SAVEPATH + restPath;
        
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
        System.out.println("Results saved at: " + path);
    }
    
    public static double calculateBDeu(Data data, Dag_n dag) {
        if (data.getData() != null) {
            BDeuScore bdeu = new BDeuScore((DataSet) data.getData());
            Fges fges = new Fges(bdeu);
            return fges.scoreDag(dag);
        } 
        return -1;
    }
    
    public static int calculateSMHD(Data data, Dag_n dag) {
        if (((BN_DataSet) data).getOriginalBNPath() != null) {
            try {
                MlBayesIm originalBN = readOriginalBayesianNetwork(((BN_DataSet) data).getOriginalBNPath());
                return Utils.SHD(Utils.removeInconsistencies(originalBN.getDag()), dag);
            } catch (Exception e) { e.printStackTrace(); }
        }
        return -1;
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
