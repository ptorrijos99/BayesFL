package org.bayesfl;

import org.bayesfl.algorithms.BN_GES;
import org.bayesfl.algorithms.LocalAlgorithm;
import org.bayesfl.data.BN_DataSet;
import org.bayesfl.data.Data;
import org.bayesfl.fusion.BN_FusionUnion;
import org.bayesfl.fusion.Fusion;

import java.util.HashSet;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Fusion fusionClient = new BN_FusionUnion();
        Fusion fusionServer = new BN_FusionUnion();
        BN_DataSet data = new BN_DataSet("./res/networks/BBDD/andes.xbif50003_.csv", "andes_50003");
        data.setOriginalBNPath("./res/networks/andes.xbif");
        LocalAlgorithm algorithm = new BN_GES("pGES", "FES");

        Set<Client> clients = new HashSet<>();
        for (int i = 0; i < 2; i++) {
            Client client = new Client(fusionClient, algorithm, data);
            client.setStats(true);
            clients.add(client);
        }

        Server server = new Server(fusionServer, clients);
        server.run();
    }
}