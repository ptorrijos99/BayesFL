package bayesfl.experiments;

import bayesfl.Client;
import bayesfl.Server;
import bayesfl.algorithms.LocalAlgorithm;
import bayesfl.algorithms.mAnDETree_mAnDE;
import bayesfl.convergence.Convergence;
import bayesfl.convergence.NoneConvergence;
import bayesfl.data.CPT_Instances;
import bayesfl.fusion.Fusion;
import bayesfl.fusion.FusionPosition;
import bayesfl.fusion.mAnDETree_Fusion;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;


public class mAnDEExpermient {
    public static String PATH = "./";
    public static String localPath = "res/classification/AnDE/";

    public static void main(String[] args) {
        String bbdd = "Covertype";
        int nClients = 5;
        int seed = 42;
        int folds = 2;
        int n = 1;
        int nTrees = 1;
        double bagSize = 0.1;
        String ensemble = "RF";
        int addNB = 0;

        //experimentmAnDE(false, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble, addNB);
        experimentmAnDE(true, bbdd, nClients, seed, folds, n, nTrees, bagSize, ensemble, addNB);
    }

    public static void experimentmAnDE(boolean federated, String bbdd, int nClients, int seed, int folds, int n, int nTrees, double bagSize, String ensemble, int addNB) {
        String bbddPath = PATH + localPath + bbdd + ".arff";

        Random random = new Random(seed);

        String operation = "mA" + n + "DE";
        if (federated) operation += "-FED";
        operation += "," + seed + "," + nTrees + "," + bagSize + "," + ensemble + "," + addNB;

        // Read the data and stratify it to the number of clients
        Instances allData = CPT_Instances.readData(bbddPath);

        // Stratify each of the divisions
        ArrayList<Instances> divisionData = CPT_Instances.divideDataSet(allData, nClients, folds, random);

        System.out.println(" N instances divided: " + allData.numInstances() + " N clients: " + nClients + " N folds: " + folds);

        // Repetitions of cross-validation
        for (int cv = 0; cv < folds; cv++){
            ArrayList<Client> clients = new ArrayList<>();

            for (int i = 0; i < nClients; i++) {
                // Divide data in train and test
                System.out.println("Client: " + i + " Fold: " + cv + " Train: " + divisionData.get(i).numInstances());

                Instances train = divisionData.get(i).trainCV(folds, cv, random);
                Instances test = divisionData.get(i).testCV(folds, cv);

                CPT_Instances data = new CPT_Instances(train, test, (bbdd + "," + i + "," + cv));
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

            //LocalExperiment.writeExistentFile(savePath);
        }
    }




}
