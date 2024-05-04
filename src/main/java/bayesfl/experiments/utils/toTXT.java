package bayesfl.experiments.utils;

import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.*;

public class toTXT {

    public static void main(String[] args) throws Exception {
        String PATH = "./res/networks/";

        // For each .xbif file in PATH
        for (File file : new File(PATH).listFiles()) {
            if (file.isFile() && file.getName().endsWith(".xbif")) {
                // Read the network
                MlBayesIm originalBN = readOriginalBayesianNetwork(file.getAbsolutePath());
                
                for (int i = 0; i < originalBN.getNumNodes(); i++) {
                    originalBN.getNode(i).setName("X" + (i+1));
                }
                
                // Open a file with the same name but .txt extension, in PATH + "txt" folder
                String txtPath = PATH + "txt/" + file.getName().replace(".xbif", ".txt");

                // Write the network in the file, as a toString
                BufferedWriter writer;
                
                System.out.println(txtPath);
                try {
                    writer = new BufferedWriter(new FileWriter(txtPath, true));
                    writer.append(originalBN.getDag().toString());
                    writer.flush();
                    writer.close();
                } catch (IOException e) { System.out.println(e); }
            }
        }
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
