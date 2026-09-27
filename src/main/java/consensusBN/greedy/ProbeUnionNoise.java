package consensusBN.greedy;

import consensusBN.Experiments.ExperimentMinCutGES;
import edu.cmu.tetrad.graph.Dag;

import java.io.File;
import java.util.*;

/**
 * A data-free, G+-free predictor of whether to build forward or prune backward.
 *
 * The experiments show the choice is not really about network size: it is about
 * how much of the unrestricted fusion is noise. Every pair adjacent in at least
 * one input becomes an edge of G+, but only pairs adjacent in a majority are
 * worth keeping, so
 *
 *     noise = 1 - |majority pairs| / |supported pairs|
 *
 * is the fraction of G+ that a good consensus will have to delete. When it is
 * high, G+ is a bad place to start and building forward is cheaper and better.
 *
 * Both counts come straight out of PairWeights, so this costs O(r n^2) and never
 * builds G+ — which matters, because on the networks where the answer is "do not
 * build G+" building it to find that out would defeat the purpose.
 *
 * Usage: ProbeUnionNoise [nets] [rs]
 */
public class ProbeUnionNoise {

    public static void main(String[] args) {
        String[] nets = (args.length > 0 ? args[0]
                : "child,mildew,barley,hailfinder,water,insurance,alarm,hepar2,win95pts,pathfinder,andes,diabetes,pigs,link").split(",");
        String[] rs = (args.length > 1 ? args[1] : "10,30,100").split(",");

        System.out.println("net,n,r,supportedPairs,majoritySkelPairs,majorityMoralPairs,noiseSkel,noiseMoral");
        for (String net : nets) {
            for (String rStr : rs) {
                int r = Integer.parseInt(rStr);
                List<Dag> dags = CacheDir.clients(net, r);
                if (dags == null) continue;

                PairWeights w = new PairWeights(dags);
                int supported = w.supportedPairs().size();
                int majSkel = 0, majMoral = 0;
                for (int i = 0; i < w.n(); i++)
                    for (int j = i + 1; j < w.n(); j++) {
                        if (2 * w.countSkel(i, j) > r) majSkel++;
                        if (2 * w.countMoral(i, j) > r) majMoral++;
                    }
                System.out.printf(Locale.US, "%s,%d,%d,%d,%d,%d,%.4f,%.4f%n",
                        net, w.n(), r, supported, majSkel, majMoral,
                        1.0 - majSkel / (double) supported,
                        1.0 - majMoral / (double) supported);
            }
        }
    }
}
