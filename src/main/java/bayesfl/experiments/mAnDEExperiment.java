package bayesfl.experiments;

import bayesfl.Client;
import bayesfl.Server;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.algorithms.mAnDETree_mAnDE;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.Weka_Instances;
import bayesfl.experiments.utils.ExperimentUtils;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.FusionPosition;
import bayesfl.fusion.mAnDETree_Fusion;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import static bayesfl.data.Weka_Instances.divide;
import static bayesfl.experiments.utils.ExperimentUtils.readParametersFromArgs;


public class mAnDEExperiment {
    public static String PATH = "./";

    public static void main(String[] args) {
        args = readParametersFromArgs(args);

        // Read the parameters from file
        String folder = args[0];
        String bbdd = args[1];
        int nClients = Integer.parseInt(args[2]);
        int seed = Integer.parseInt(args[3]);
        int folds = Integer.parseInt(args[4]);
        int n = Integer.parseInt(args[5]);
        int nTrees = Integer.parseInt(args[6]);
        double bagSize = Double.parseDouble(args[7]);
        String ensemble = args[8];
        double addNB = Double.parseDouble(args[9]);

        experimentmAnDE(true, folder, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble, addNB);
    }

    /*public static void main(String[] args) {
        String folder = "AnDE";
        String bbdd = "Adult";
        int nClients = 100;
        int seed = 2;
        int folds = 5;
        int n = 1;
        int nTrees = 1;
        double bagSize = 1;
        String ensemble = "RF";
        double addNB = 0;

        ExperimentUtils.experimentID = 0;
        //experimentmAnDE(false, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble, addNB);
        experimentmAnDE(true, folder, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble, addNB);
    }*/

    public static void experimentmAnDE(boolean federated, String folder, String bbdd, int nClients, int seed, int nFolds, int n, int nTrees, double bagSize, String ensemble, double addNB) {
        String bbddPath = PATH + "res/classification/" + folder + "/" + bbdd + ".arff";

        String operation = "mA" + n + "DE";
        if (federated) operation += "-FED";
        operation += "," + seed + "," + nTrees + "," + bagSize + "," + ensemble + "," + addNB;

        // Read the data and stratify it to the number of clients
        Instances[][][] splits = divide(bbdd, bbddPath, nFolds, nClients, seed);

        // Repetitions of cross-validation
        for (int cv = 0; cv < nFolds; cv++){
            ArrayList<Client> clients = new ArrayList<>();

            for (int i = 0; i < nClients; i++) {
                // Divide data in train and test
                Instances train = splits[cv][i][0];
                Instances test = splits[cv][i][1];

                Weka_Instances data = new Weka_Instances((bbdd + "," + i + "," + cv), train, test);
                LocalAlgorithm algorithm = new mAnDETree_mAnDE(n, nTrees, bagSize, ensemble, addNB);

                int position = 0;
                if (federated) position = -1;
                Fusion fusionClient = new FusionPosition(position);

                Client client = new Client(fusionClient, algorithm, data);
                client.setStats(true, true, PATH);
                client.setExperimentName(operation);
                client.setID(i);
                clients.add(client);
            }

            Fusion fusionServer = new mAnDETree_Fusion();
            Convergence convergence = new NoneConvergence();
            Server server = new Server(fusionServer, convergence, clients);

            server.setStats(false, PATH);
            server.setExperimentName(operation);
            server.setnIterations(1);

            server.run();
        }
    }
}
