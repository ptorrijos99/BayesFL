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
 *    LocalExperiment.java
 *    Copyright (C) 2023 Universidad de Castilla-La Mancha, España
 *
 * @author Pablo Torrijos Arenas
 *
 */

package org.albacete.simd.bayesfl.experiments;

import java.io.BufferedReader;
import java.io.FileReader;
import static org.albacete.simd.bayesfl.experiments.LocalExperiment.launchExperiment;

public class ClusterExperiment {
    
    public static void main(String[] args) {
        int index = Integer.parseInt(args[0]);
        String paramsFileName = args[1];
        //int threads = Integer.parseInt(args[2]);
        
        // Read the parameters from args       
        String[] parameters = null;
        try (BufferedReader br = new BufferedReader(new FileReader(paramsFileName))) {
            String line;
            for (int i = 0; i < index; i++)
                br.readLine();
            line = br.readLine();
            parameters = line.split(" ");
        }
        catch(Exception e){ System.out.println(e); } 
        
        System.out.println("Number of hyperparams: " + parameters.length);
        int i=0;
        for (String string : parameters) {
            System.out.println("Param[" + i + "]: " + string);
            i++;
        }
        
        // Read the parameters from file
        String net = parameters[0];
        String bbdd = parameters[1];
        String algorithm = parameters[2];
        String refinement = parameters[3];
        String fusionC = parameters[4];
        String fusionS = parameters[5];
        int nClients = Integer.parseInt(parameters[6]);
        int maxEdgesIt = Integer.parseInt(parameters[7]);
        int nIterations = Integer.parseInt(parameters[8]);
        
        LocalExperiment.PATH = "/tmp/pablo.torrijos/";
        
        // Launch the experiment
        launchExperiment(net, algorithm, refinement, fusionC, fusionS, bbdd, nClients, maxEdgesIt, nIterations);
    }

}
