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


public class mAnDEExperiment {
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

        // Read the parameters from file
        String folder = parameters[0];
        String bbdd = parameters[1];
        int nClients = Integer.parseInt(parameters[2]);
        int seed = Integer.parseInt(parameters[3]);
        int folds = Integer.parseInt(parameters[4]);
        int n = Integer.parseInt(parameters[5]);
        int nTrees = Integer.parseInt(parameters[6]);
        double bagSize = Double.parseDouble(parameters[7]);
        String ensemble = parameters[8];
        double addNB = Double.parseDouble(parameters[9]);

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
