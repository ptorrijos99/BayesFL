package consensusBN.Experiments;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.*;
import org.albacete.simd.utils.Utils;
import weka.classifiers.bayes.net.BIFReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Distance from the empty network to each gold standard, used as the lower
 * reference line in the adversarial robustness figures. An empty DAG is feasible
 * for every treewidth bound, so it is the score any useful method must beat.
 */
public class EmptyNetworkReference {

    static final String PATH = System.getProperty("user.dir") + "/";

    public static void main(String[] args) throws IOException {
        String[] nets = args.length > 0 ? args : new String[]{"alarm", "insurance", "water", "barley"};
        String out = PATH + "results/adversarial/empty_reference.csv";
        new File(out).getParentFile().mkdirs();

        int r = 10;
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(out))) {
            bw.write("net,emptySMHDgold,emptySHDgold,gesSMHDgold,gesSHDgold,twGold,goldEdges\n");
            for (String net : nets) {
                DataSet data = Utils.readData(PATH + "res/networks/BBDD/" + net + "/" + net + ".0.csv");
                BIFReader reader = new BIFReader();
                try { reader.processFile(PATH + "res/networks/xbif/" + net + ".xbif"); }
                catch (Exception e) { throw new RuntimeException(e); }

                RandomBN rbn = new RandomBN(reader, data, 0, 1, 5);
                Dag gold = new Dag(rbn.originalBayesIm.getBayesPm().getDag());
                Dag empty = new Dag(gold.getNodes());

                Graph cpdagGold = GraphTransforms.dagToCpdag(new EdgeListGraph(gold));

                int smhd = Utils.SMHD(empty, gold);
                int shd = Utils.SHDundir(
                        GraphTransforms.dagToCpdag(new EdgeListGraph(empty)), cpdagGold);

                // Mean quality of an individual honest participant, which is what the
                // consensus has to improve on for the fusion to be worth performing.
                double gesSmhd = 0, gesShd = 0;
                for (Dag d : ExperimentsJournal.loadGESDags(net + ".0", net, r)) {
                    gesSmhd += Utils.SMHD(d, gold);
                    gesShd += Utils.SHDundir(
                            GraphTransforms.dagToCpdag(new EdgeListGraph(d)), cpdagGold);
                }
                gesSmhd /= r;
                gesShd /= r;

                bw.write(String.format(java.util.Locale.US, "%s,%d,%d,%.2f,%.2f,%d,%d%n",
                        net, smhd, shd, gesSmhd, gesShd,
                        Utils.getTreeWidth(gold), gold.getNumEdges()));
                System.out.printf("%s  empty SMHD=%d SHD=%d | mean GES SMHD=%.1f SHD=%.1f%n",
                        net, smhd, shd, gesSmhd, gesShd);
            }
        }
        System.out.println("Written: " + out);
    }
}
