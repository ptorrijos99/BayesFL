package consensusBN.greedy;

import edu.cmu.tetrad.graph.*;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.FileInputStream;
import java.util.*;

/**
 * Loads a gold-standard DAG from an XMLBIF file, structure only.
 *
 * The existing route went through weka's BIFReader and
 * Utils.transformBayesNetToBayesIm, which aligns the network's categories with
 * the categories present in a sampled CSV. That coupling is unnecessary here —
 * every harness in this package uses the gold standard purely as a DAG, for
 * SMHD and SHD — and it actively breaks in the two regimes this study needs:
 *
 *   - small samples, where a rare state may never appear in 1000 rows, so the
 *     data has fewer categories than the network and the alignment throws
 *     ArrayIndexOutOfBounds;
 *   - imported networks, whose XMLBIF may carry no DOCTYPE for weka to validate
 *     against.
 *
 * Reading only VARIABLE/NAME and DEFINITION/FOR+GIVEN sidesteps both. In XMLBIF
 * a DEFINITION lists the node in FOR and each of its parents in GIVEN.
 */
public class GoldStandard {

    private static final Map<String, Dag> CACHE = new HashMap<>();

    public static Dag load(String net) {
        return CACHE.computeIfAbsent(net, GoldStandard::parse);
    }

    /**
     * The gold standard rebuilt on the caller's own Node objects.
     *
     * Tetrad indexes adjacency by Node instance in a HashMap, and equality
     * between a GraphNode and the DiscreteVariable that a DataSet-derived DAG
     * carries is not symmetric. Comparing a freshly parsed gold against client
     * DAGs therefore throws inside isAdjacentTo rather than returning a wrong
     * answer — which is at least loud, but only if something actually compares
     * them. Always align before any SMHD/SHD call.
     */
    public static Dag loadAlignedTo(String net, List<Node> nodes) {
        Dag parsed = load(net);
        Map<String, Node> byName = new HashMap<>();
        for (Node n : nodes) byName.put(n.getName(), n);

        Dag out = new Dag(new ArrayList<>(nodes));
        for (Edge e : parsed.getEdges()) {
            Node a = e.getNode1(), b = e.getNode2();
            if (!parsed.isParentOf(a, b)) { Node t = a; a = b; b = t; }
            Node pa = byName.get(a.getName()), ch = byName.get(b.getName());
            if (pa == null || ch == null)
                throw new IllegalStateException("gold standard for " + net
                        + " mentions " + (pa == null ? a.getName() : b.getName())
                        + ", which the data does not contain");
            out.addEdge(Edges.directedEdge(pa, ch));
        }
        return out;
    }

    private static Dag parse(String net) {
        File f = new File("./res/networks/xbif/" + net + ".xbif");
        if (!f.exists()) throw new IllegalArgumentException("no xbif for " + net + ": " + f.getPath());
        try (FileInputStream in = new FileInputStream(f)) {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setValidating(false);
            factory.setNamespaceAware(false);
            // XMLBIF files reference a DTD that is usually absent; never fetch it.
            factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
            DocumentBuilder builder = factory.newDocumentBuilder();
            builder.setEntityResolver((publicId, systemId) ->
                    new InputSource(new java.io.StringReader("")));
            Document doc = builder.parse(new InputSource(in));

            Map<String, Node> nodes = new LinkedHashMap<>();
            NodeList vars = doc.getElementsByTagName("VARIABLE");
            for (int i = 0; i < vars.getLength(); i++) {
                String name = firstChildText((Element) vars.item(i), "NAME");
                if (name != null) nodes.put(name, new GraphNode(name));
            }
            if (nodes.isEmpty()) throw new IllegalStateException("no VARIABLE elements in " + f);

            Dag dag = new Dag(new ArrayList<>(nodes.values()));
            NodeList defs = doc.getElementsByTagName("DEFINITION");
            for (int i = 0; i < defs.getLength(); i++) {
                Element def = (Element) defs.item(i);
                String child = firstChildText(def, "FOR");
                if (child == null || !nodes.containsKey(child)) continue;
                NodeList givens = def.getElementsByTagName("GIVEN");
                for (int j = 0; j < givens.getLength(); j++) {
                    String parent = givens.item(j).getTextContent().trim();
                    if (!nodes.containsKey(parent) || parent.equals(child)) continue;
                    Edge e = Edges.directedEdge(nodes.get(parent), nodes.get(child));
                    if (!dag.isAdjacentTo(nodes.get(parent), nodes.get(child))) dag.addEdge(e);
                }
            }
            return dag;
        } catch (Exception e) {
            throw new RuntimeException("failed to read gold standard for " + net, e);
        }
    }

    private static String firstChildText(Element parent, String tag) {
        NodeList l = parent.getElementsByTagName(tag);
        return l.getLength() == 0 ? null : l.item(0).getTextContent().trim();
    }
}
