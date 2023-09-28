package org.bayesfl;

import org.bayesfl.algorithms.BN_GES;
import org.bayesfl.algorithms.LocalAlgorithm;
import org.bayesfl.data.BN_DataSet;
import org.bayesfl.fusion.BN_FusionUnion;
import org.bayesfl.fusion.BN_FusionIntersection;
import org.bayesfl.fusion.Fusion;

import java.util.HashSet;
import java.util.Set;


public class Main {
    public static void main(String[] args) {
        String net = "andes";

        Set<Client> clients = new HashSet<>();
        for (int i = 1; i < 5; i++) {
            Fusion fusionClient = new BN_FusionIntersection();
            BN_DataSet data = new BN_DataSet("./res/networks/BBDD/" + net + ".xbif5000" + i + "_.csv", net + "_5000" + i);
            data.setOriginalBNPath("./res/networks/" + net + ".xbif");
            LocalAlgorithm algorithm = new BN_GES("pGES", "FES");
            
            Client client = new Client(fusionClient, algorithm, data);
            client.setStats(true);
            clients.add(client);
        }

        Fusion fusionServer = new BN_FusionIntersection();
        Server server = new Server(fusionServer, clients);
        server.setStats(true);
        server.setOriginalBNPath("./res/networks/" + net + ".xbif");
        server.run();
    }
}