package org.albacete.simd.utils;

import consensusBN.BetaToAlpha;
import consensusBN.ConsensusUnion;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesIm;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.SearchGraphUtils;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.DiscreteEstimatorBayes;
import weka.estimators.Estimator;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Utils {

    private static Random random = new Random();
    
    /**
     * Transforms a maximally directed pattern (PDAG) represented in graph
     * <code>g</code> into an arbitrary DAG by modifying <code>g</code> itself.
     * Based on the algorithm described in </p> Chickering (2002) "Optimal
     * structure identification with greedy search" Journal of Machine Learning
     * Research. </p> R. Silva, June 2004
     */
    public static void pdagToDag(Graph g) {
        Graph p = new EdgeListGraph(g);
        List<Edge> undirectedEdges = new ArrayList<>();

        for (Edge edge : g.getEdges()) {
            if (edge.getEndpoint1() == Endpoint.TAIL
                    && edge.getEndpoint2() == Endpoint.TAIL
                    && !undirectedEdges.contains(edge)) {
                undirectedEdges.add(edge);
            }
        }
        g.removeEdges(undirectedEdges);
        List<Node> pNodes = p.getNodes();

        do {
            Node x = null;

            for (Node pNode : pNodes) {
                x = pNode;

                if (!p.getChildren(x).isEmpty()) {
                    continue;
                }

                Set<Node> neighbors = new HashSet<>();

                for (Edge edge : p.getEdges()) {
                    if (edge.getNode1() == x || edge.getNode2() == x) {
                        if (edge.getEndpoint1() == Endpoint.TAIL
                                && edge.getEndpoint2() == Endpoint.TAIL) {
                            if (edge.getNode1() == x) {
                                neighbors.add(edge.getNode2());
                            } else {
                                neighbors.add(edge.getNode1());
                            }
                        }
                    }
                }
                if (!neighbors.isEmpty()) {
                    Collection<Node> parents = p.getParents(x);
                    Set<Node> all = new HashSet<>(neighbors);
                    all.addAll(parents);
                    if (!GraphUtils.isClique(all, p)) {
                        continue;
                    }
                }

                for (Node neighbor : neighbors) {
                    Node node1 = g.getNode(neighbor.getName());
                    Node node2 = g.getNode(x.getName());

                    g.addDirectedEdge(node1, node2);
                }
                p.removeNode(x);
                break;
            }
            pNodes.remove(x);
        } while (!pNodes.isEmpty());
    }

    /**
     * Separates the set of possible arcs into as many subsets as threads we use to solve the problem.
     *
     * @param listOfArcs List of {@link Edge Edges} containing all the possible edges for the actual problem.
     * @param numSplits  The number of splits to do in the listOfArcs.
     * @return The subsets of the listOfArcs in an ArrayList of TupleNode.
     */
    public static List<Set<Edge>> split(Set<Edge> listOfArcs, int numSplits) {
        List<Set<Edge>> subSets = new ArrayList<>(numSplits);

        // Shuffling arcs
        List<Edge> shuffledArcs = new ArrayList<>(listOfArcs);
        Collections.shuffle(shuffledArcs, random);

        // Splitting Arcs into subsets
        int n = 0;
        for(int s = 0; s< numSplits-1; s++){
            Set<Edge> sub = new HashSet<>();
            for(int i = 0; i < Math.floorDiv(shuffledArcs.size(),numSplits) ; i++){
                sub.add(shuffledArcs.get(n));
                n++;
            }
            subSets.add(sub);
        }

        // Adding leftovers
        Set<Edge> sub = new HashSet<>();
        for(int i = n; i < shuffledArcs.size(); i++ ){
            sub.add(shuffledArcs.get(i));
        }
        subSets.add(sub);

        return subSets;
    }

    public static void setSeed(long seed){
        random = new Random(seed);
    }

    /**
     * Calculates the amount of possible arcs between the variables of the dataset and stores it.
     *
     * @param data DataSet used to calculate the arcs between its columns (nodes).
     */
    public static Set<Edge> calculateArcs(DataSet data) {
        return calculateArcs(data.getVariables());
    }

    /**
     * Calculates the amount of possible arcs between the variables of the dataset and stores it.
     *
     * @param variables List of nodes used to calculate the arcs between its nodes.
     */
    public static Set<Edge> calculateArcs(List<Node> variables) {
        int N = variables.size();

        //0. Accumulator
        Set<Edge> setOfArcs = new HashSet<>(N * (N - 1));

        //1. Iterate over variables and save pairs
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                // Getting pair of variables (Each variable is different)
                Node var_A = variables.get(i);
                Node var_B = variables.get(j);

                //3. Storing both pairs
                setOfArcs.add(Edges.directedEdge(var_A, var_B));
                setOfArcs.add(Edges.directedEdge(var_B, var_A));
            }
        }
        return setOfArcs;
    }
    
    /**
     * Calculates the amount of possible edges between the variables of the dataset and stores it.
     *
     * @param data DataSet used to calculate the edges between its columns (nodes).
     */
    public static Set<Edge> calculateEdges(DataSet data) {
        //0. Accumulator
        Set<Edge> setOfArcs = new HashSet<>(data.getNumColumns() * (data.getNumColumns() - 1));
        //1. Get edges (variables)
        List<Node> variables = data.getVariables();
        //2. Iterate over variables and save pairs
        for (int i = 0; i < data.getNumColumns() - 1; i++) {
            for (int j = i + 1; j < data.getNumColumns(); j++) {
                // Getting pair of variables (Each variable is different)
                Node var_A = variables.get(i);
                Node var_B = variables.get(j);

                //3. Storing both pairs
                setOfArcs.add(Edges.directedEdge(var_A, var_B));
            }
        }
        return setOfArcs;
    }

    /**
     * Stores the data from a csv as a DataSet object.
     * @param path
     * Path to the csv file.
     * @return DataSet containing the data from the csv file.
     */
    public static DataSet readData(String path){
        // Initial Configuration
        DataReader reader = new DataReader();
        reader.setDelimiter(DelimiterType.COMMA);
        reader.setMaxIntegralDiscrete(100);
        DataSet dataSet = null;
        // Reading data
        try {
            dataSet = reader.parseTabular(new File(path));
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dataSet;
    }


    public static Node getNodeByName(List<Node> nodes, String name){
        for(Node n : nodes){
            if (n.getName().equals(name)){
                return n;
            }
        }
        return null;
    }

    public static int getIndexOfNodeByName(List<Node> nodes, String name){
        for(int i = 0; i < nodes.size(); i++){
            Node n = nodes.get(i);
            if(n.getName().equals(name)){
                return i;
            }
        }
        return -1;
    }

    private static void ensureVariables(ArrayList<Dag> setofbns){
        List<Node> nodes = setofbns.get(0).getNodes();
        //System.out.println("Nodes: " + nodes);
        for(int i = 1 ; i< setofbns.size(); i++) {
            Dag oldDag = setofbns.get(i);
            Set<Edge> oldEdges = oldDag.getEdges();
            Dag newdag = new Dag(nodes);
            for(Edge e: oldEdges){
                /*
                System.out.println("Node1");
                System.out.println(e.getNode1());
                System.out.println("Node2");
                System.out.println(e.getNode2());
                */
                //int tailIndex = nodes.indexOf(e.getNode1());
                //int headIndex = nodes.indexOf(e.getNode2());

                int tailIndex = getIndexOfNodeByName(nodes, e.getNode1().getName());
                int headIndex = getIndexOfNodeByName(nodes, e.getNode2().getName());

                //System.out.println("tail: " + tailIndex);
                //System.out.println("head: "  + headIndex);
                Edge newEdge = new Edge(nodes.get(tailIndex),nodes.get(headIndex), Endpoint.TAIL, Endpoint.ARROW);
                newdag.addEdge(newEdge);
            }
            setofbns.remove(i);
            setofbns.add(i, newdag);
        }
    }

    public static int SMHD(Graph bn1, Graph bn2) {
        Graph g1 = moralize(bn1);
        Graph g2 = moralize(bn2);

        int sum = 0;
        for(Edge e: g1.getEdges()) {
            if(!g2.isAdjacentTo(e.getNode1(), e.getNode2())) sum++;
        }

        for(Edge e: g2.getEdges()) {
            if(!g1.isAdjacentTo(e.getNode1(), e.getNode2())) sum++;
        }
        return sum;
    }

    // This function was used in the SMHD instead of moralize
    private static Graph connectParents(Graph bn) {
        EdgeListGraph g = new EdgeListGraph(bn);
        for(Node n: bn.getNodes()) {
            List<Node> p = bn.getParents(n);
            for (int i=0; i<p.size()-1;i++) {
                for (int j = i + 1; j < p.size(); j++) {
                    if (!g.isAdjacentTo(p.get(i), p.get(j))) {
                        Edge e = new Edge(p.get(i), p.get(j), Endpoint.TAIL, Endpoint.TAIL);
                        g.addEdge(e);
                    }
                }
            }
        }
        return g;
    }

    /**
     * SHD for DAGs: number of arcs added, deleted and reversed to make the two DAGs match.
     * Reversed arcs are counted only once.
     */
    public static int SHD(Dag bn1, Dag bn2) {
        int sum = countDifferences(bn1, bn2);

        for (Edge e : bn2.getEdges()) {
            if (!bn1.containsEdge(e) && !bn1.containsEdge(e.reverse())) {
                sum++;  // Edge in bn2 but not in bn1 (without counting reverse edges, as they are already counted)
            }
        }
        return sum;
    }



    /**
     * SHD for PDAGs: number of operations (add or delete an undirected edge; and add, remove, or reverse the
     * orientation of an arc). All operations have a cost of 1.
     */
    public static int SHDundir(Graph bn1, Graph bn2) {
        int sum = countDifferences(bn1, bn2);

        for (Edge e : bn2.getEdges()) {
            if (e.isDirected()) {
                if (!bn1.containsEdge(e) && !bn1.containsEdge(e.reverse())) {
                    sum++;  // Directed edge in bn2 but not in bn1 (without counting reverse edges, as they are already counted)
                }
            } else {
                if (!bn1.containsEdge(e)) {
                    sum++;  // Undirected edge in bn2 but not in bn1
                }
            }
        }

        return sum;
    }


    public static int fusionSimilarity(Dag g1, Dag g2) {
        // 𝐺+ = GESℎ𝑑({𝐺1, 𝐺2})  // Optimal fusion, Algorithm 6  (here we use approximate fusion for efficiency)
        Dag gPlus = ConsensusUnion.fusionUnion(Arrays.asList(g1, g2));

        // 𝜎 = a topological order for 𝐺+
        List<Node> sigma = gPlus.getTopologicalOrder();

        // 𝐺𝜎 = MethodA(𝐺,𝜎)  // Minimal 𝐼-map, Algorithm 1
        Dag gSigma1 = BetaToAlpha.transformToAlpha(g1, sigma);
        Dag gSigma2 = BetaToAlpha.transformToAlpha(g2, sigma);

        // Get the undirected graphs for each DAG
        Graph undirectedG1 = undirectedGraphFromDag(g1);
        Graph undirectedGSigma1 = undirectedGraphFromDag(gSigma1);
        Graph undirectedG2 = undirectedGraphFromDag(g2);
        Graph undirectedGSigma2 = undirectedGraphFromDag(gSigma2);

        // Calculate the structural differences between the sets of edges
        int diff = 0;
        diff += countDifferences(undirectedGSigma1, undirectedGSigma2);  // |Eσ1' ⧵ Eσ2'|
        diff += countDifferences(undirectedGSigma2, undirectedGSigma1);  // |Eσ2' ⧵ Eσ1'|
        diff += countDifferences(undirectedGSigma1, undirectedG1);       // |Eσ1' ⧵ E′1|
        diff += countDifferences(undirectedGSigma2, undirectedG2);       // |Eσ2' ⧵ E′2|

        return diff;
    }

    public static Graph undirectedGraphFromDag(Dag dag) {
        Graph undirectedGraph = new EdgeListGraph(dag.getNodes());
        dag.getEdges().forEach(e -> undirectedGraph.addUndirectedEdge(e.getNode1(), e.getNode2()));
        return undirectedGraph;
    }

    /**
     * Counts the number of edges that are in g1 but not in g2.
     */
    public static int countDifferences(Graph g1, Graph g2) {
        int diff = 0;
        for (Edge e : g1.getEdges()) {
            if (!g2.containsEdge(e)) {
                diff++;
            }
        }
        return diff;
    }


    public static List<Node> getMarkovBlanket(Dag bn, Node n){
        List<Node> mb = new ArrayList<>();

        // Adding children and parents to the Markov's Blanket of this node
        List<Node> children = bn.getChildren(n);
        List<Node> parents = bn.getParents(n);

        mb.addAll(children);
        mb.addAll(parents);

        for(Node child : children){
            for(Node father : bn.getParents(child)){
                if (!father.equals(n)){
                    mb.add(father);
                }
            }
        }
        return mb;
    }

    /**
     * Gives back the percentages of markov's blanquet difference with the original bayesian network. It gives back the
     * percentage of difference with the blanquet of the original bayesian network, the percentage of extra nodes added
     * to the blanquet and the percentage of missing nodes in the blanquet compared with the original.
     * @param original
     * @param created
     * @return
     */
    public static double [] avgMarkovBlanquetdif(Dag original, Dag created) {
        if (original.getNodes().size() != created.getNodes().size())
            return null;

        for (String originalNodeName : original.getNodeNames()) {
            if (!created.getNodeNames().contains(originalNodeName))
                return null;
        }

        // First number is the average dfMB, the second one is the amount of more variables in each MB, the last number is the the amount of missing variables in each MB
        double[] result = new double[3];
        double differenceNodes = 0;
        double plusNodes = 0;
        double minusNodes = 0;

        for (Node e1 : original.getNodes()) {
            Node e2 = created.getNode(e1.getName());

            // Creating Markov's Blanket
            List<Node> mb1 = getMarkovBlanket(original, e1);
            List<Node> mb2 = getMarkovBlanket(created, e2);

            ArrayList<String> names1 = new ArrayList<String>();
            ArrayList<String> names2 = new ArrayList<String>();
            // Nodos de más en el manto creado
            for (Node n1 : mb1) {
                String name1 = n1.getName();
                names1.add(name1);
            }
            for (Node n2 : mb2) {
                String name2 = n2.getName();
                names2.add(name2);
            }

            //Variables de más
            for(String s2: names2) {
                if(!names1.contains(s2)) {
                    differenceNodes++;
                    plusNodes++;
                }
            }
            // Variables de menos
            for(String s1: names1) {
                if(!names2.contains(s1)) {
                    differenceNodes++;
                    minusNodes++;
                }
            }
        }

        // Differences of MM
        result[0] = differenceNodes;
        result[1] = plusNodes;
        result[2] = minusNodes;

        return result;

    }

    /**
     * Transforms a graph to a DAG, and removes any possible inconsistency found throughout its structure.
     * @param g Graph to be transformed.
     * @return Resulting DAG of the inserted graph.
     */
    public static Dag removeInconsistencies(Graph g){
        // Transforming the current graph into a DAG
        pdagToDag(g);

        // Checking Consistency
        Node nodeT, nodeH;
        for (Edge e : g.getEdges()){
            if(!e.isDirected()) continue;
            //System.out.println("Undirected Edge: " + e);
            Endpoint endpoint1 = e.getEndpoint1();
            if (endpoint1.equals(Endpoint.ARROW)){
                nodeT = e.getNode1();
                nodeH = e.getNode2();
            }else{
                nodeT = e.getNode2();
                nodeH = e.getNode1();
            }

            if(g.existsDirectedPathFromTo(nodeT, nodeH)){
                System.out.println("Directed path from " + nodeT + " to " + nodeH +"\t Deleting Edge...");
                g.removeEdge(e);
            }
        }
        // Adding graph from each thread to the graphs array
        return new Dag(g);
    }

    public static double LL(BayesIm bn, DataSet data) {
        BayesIm bayesIm;
        int[][][] observedCounts;

        Graph graph = bn.getDag();
        Node[] nodes = new Node[graph.getNumNodes()];

        observedCounts = new int[nodes.length][][];
        int[][] observedCountsRowSum = new int[nodes.length][];

        bayesIm = new MlBayesIm(bn);

        for (int i = 0; i < nodes.length; i++) {
            int numRows = bayesIm.getNumRows(i);
            observedCounts[i] = new int[numRows][];
            observedCountsRowSum[i] = new int[numRows];

            for (int j = 0; j < numRows; j++) {
                observedCountsRowSum[i][j] = 0;
                int numCols = bayesIm.getNumColumns(i);
                observedCounts[i][j] = new int[numCols];
            }
        }

        //At this point set values in observedCounts
        for (int j = 0; j < data.getNumColumns(); j++) {
            DiscreteVariable var = (DiscreteVariable) data.getVariables().get(j);
            String varName = var.getName();
            Node varNode = bn.getDag().getNode(varName);
            int varIndex = bayesIm.getNodeIndex(varNode);

            int[] parentVarIndices = bayesIm.getParents(varIndex);

            if (parentVarIndices.length == 0) {
                //System.out.println("No parents");
                for (int col = 0; col < var.getNumCategories(); col++) {
                    observedCounts[varIndex][0][col] = 0;
                }

                for (int i = 0; i < data.getNumRows(); i++) {
                    observedCounts[varIndex][0][data.getInt(i, j)] += 1.0;
                }
            }
            else {    //For variables with parents:
                int numRows = bayesIm.getNumRows(varIndex);

                for (int row = 0; row < numRows; row++) {
                    int[] parValues = bayesIm.getParentValues(varIndex, row);

                    for (int col = 0; col < var.getNumCategories(); col++) {
                        try{
                            observedCounts[varIndex][row][col] = 0;
                        }catch(Exception ex) {}
                    }

                    for (int i = 0; i < data.getNumRows(); i++) {
                        //for a case where the parent values = parValues increment the estCount
                        boolean parentMatch = true;

                        for (int p = 0; p < parentVarIndices.length; p++) {
                            if (parValues[p] != data.getInt(i, parentVarIndices[p])) {
                                parentMatch = false;
                                break;
                            }
                        }

                        if (!parentMatch) {
                            continue;  //Not a matching case; go to next.
                        }
                        observedCounts[varIndex][row][data.getInt(i, j)] += 1;
                    }
                }
            }
        }

        for (int i = 0; i < nodes.length; i++) {
            for (int j = 0; j < bayesIm.getNumRows(i); j++) {
                for (int k = 0; k < bayesIm.getNumColumns(i); k++) {
                    observedCountsRowSum[i][j] += observedCounts[i][j][k];
                }
            }
        }

        double sum = 0.0;
        int n = nodes.length;

        for (int i = 0; i < n; i++) {
            int qi = bayesIm.getNumRows(i);
            for (int j = 0; j < qi; j++) {
                int ri = bayesIm.getNumColumns(i);
                for (int k = 0; k < ri; k++) {
                    try {
                        double p1 = observedCounts[i][j][k];
                        double p2 = observedCountsRowSum[i][j];
                        double p3 = Math.log(p1/p2);
                        if(p1 != 0.0) sum += p1 * p3;
                    }
                    catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return sum / data.getNumRows() / data.getNumColumns();
    }

    public static double LL(Dag g, DataSet data) {
        BayesPm bnaux = new BayesPm(g);
        MlBayesIm bnOut = new MlBayesIm(bnaux, MlBayesIm.MANUAL);
        return LL(bnOut, data);
    }

    /**
     * Transforms a BayesNet read from a xbif file into a BayesPm object for tetrad
     *
     * @param wekabn BayesNet read from an xbif file
     * @return The BayesPm of the BayesNet
     */
    public static BayesPm transformBayesNetToBayesPm(BayesNet wekabn) {
        ArrayList<Node> nodes = new ArrayList<>();

        // Getting nodes from weka network and adding them to a GraphNode
        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            nodes.add(new DiscreteVariable(wekabn.getNodeName(indexNode)));
        }
        Dag graph = new Dag(nodes);

        // Adding all of the edges from the wekabn into the new Graph
        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            int nParent = wekabn.getNrOfParents(indexNode);
            for (int np = 0; np < nParent; np++) {
                int indexp = wekabn.getParent(indexNode, np);
                Edge ed = new Edge(graph.getNode(wekabn.getNodeName(indexp)), graph.getNode(wekabn.getNodeName(indexNode)), Endpoint.TAIL, Endpoint.ARROW);
                graph.addEdge(ed);
            }
        }
        //System.out.println(graph);
        return new BayesPm(graph);
    }

    /**
     * Transforms a BayesNet read from a xbif file into a BayesIm object for tetrad, with probability tables
     *
     * @param wekabn BayesNet read from an xbif file
     * @return The BayesIm of the BayesNet
     */
    public static MlBayesIm transformBayesNetToBayesIm(BayesNet wekabn) {
        BayesPm bayesPm = transformBayesNetToBayesPm(wekabn);

        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            Node node = bayesPm.getNode(wekabn.getNodeName(indexNode));
            bayesPm.setNumCategories(node, wekabn.getCardinality(indexNode));
        }

        MlBayesIm bayesIm = new MlBayesIm(bayesPm);

        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            double[][] probTable = getProbabilityTable(wekabn.m_Distributions[indexNode]);
            int indexInIm = bayesIm.getNodeIndex(bayesIm.getNode(wekabn.getNodeName(indexNode)));

            bayesIm.setProbability(indexInIm, probTable);
        }
        return bayesIm;
    }

    /**
     * Transforms a BayesNet read from a xbif file into a BayesIm object for tetrad, with variables and probability tables
     * with a specific order of the categories
     *
     * @param wekabn BayesNet read from an xbif file
     * @return The BayesIm of the BayesNet
     */
    public static MlBayesIm transformBayesNetToBayesIm(BayesNet wekabn, ArrayList<String>[] categories) {
        BayesPm bayesPm = transformBayesNetToBayesPm(wekabn);

        String[][] classWekaOrder = new String[wekabn.getNrOfNodes()][];

        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            Node node = bayesPm.getNode(wekabn.getNodeName(indexNode));
            bayesPm.setNumCategories(node, wekabn.getCardinality(indexNode));

            classWekaOrder[indexNode] = new String[wekabn.getCardinality(indexNode)];

            for (int j=0; j < wekabn.getCardinality(indexNode); j++){
                classWekaOrder[indexNode][j] = wekabn.m_Instances.attribute(indexNode).value(j);
            }
        }

        MlBayesIm bayesIm = new MlBayesIm(bayesPm);

        for (int indexNode = 0; indexNode < wekabn.getNrOfNodes(); indexNode++) {
            double[][] probTableWeka = getProbabilityTable(wekabn.m_Distributions[indexNode]);
            double[][] probTable = new double[probTableWeka.length][];
            int indexInIm = bayesIm.getNodeIndex(bayesIm.getNode(wekabn.getNodeName(indexNode)));

            for (int i=0; i < probTableWeka.length; i++){
                probTable[i] = new double[probTableWeka[i].length];
                for (int j=0; j < probTableWeka[i].length; j++){
                    int index = categories[indexNode].indexOf(classWekaOrder[indexNode][j]);

                    if (index == -1) {
                        categories[indexNode].add(classWekaOrder[indexNode][j]);
                        index = categories[indexNode].indexOf(classWekaOrder[indexNode][j]);
                    }

                    probTable[i][index] = probTableWeka[i][j];
                }
            }

            bayesIm.setProbability(indexInIm, probTable);
        }
        return bayesIm;
    }

    /**
     * Gets the probability table of an array of WEKA estimators
     *
     * @param estimators Array of WEKA estimators
     * @return The probability table of the estimators
     */
    public static double[][] getProbabilityTable(Estimator[] estimators) {
        int nRows = estimators.length;
        int nCols = ((DiscreteEstimatorBayes)estimators[0]).getNumSymbols();
        double[][] table = new double[nRows][nCols];

        for (int i = 0; i < nRows; i++) {
            Estimator estimator = estimators[i];
            for (int j = 0; j < nCols; j++) {
                table[i][j] = estimator.getProbability(j);
            }
        }
        return table;
    }

    public static Map<Node, Set<Node>> getMoralTriangulatedCliques(Dag dag) {
        // 1. Moralize the graph
        EdgeListGraph undirectedGraph = (EdgeListGraph) moralize(dag);

        // 2. Triangulate the graph
        // tetrad-lib/src/main/java/edu/cmu/tetrad/bayes/JunctionTreeAlgorithm.java#L106
        Node[] maximumCardinalityOrdering = getMaximumCardinalityOrdering(undirectedGraph);
        fillIn(undirectedGraph, maximumCardinalityOrdering);

        // 3. Find the maximum clique size
        maximumCardinalityOrdering = getMaximumCardinalityOrdering(undirectedGraph);
        return getCliques(maximumCardinalityOrdering, undirectedGraph);
    }

    public static int getTreeWidth(Dag dag) {
        Map<Node, Set<Node>> cliques = getMoralTriangulatedCliques(dag);
        return cliques.values().stream().mapToInt(Set::size).max().orElse(0);
    }

    /**
     * Get cliques in a decomposable graph. A clique is a fully-connected
     * subgraph.
     *
     * @param graph    decomposable graph
     * @param ordering maximum cardinality ordering
     * @return set of cliques
     */
    public static Map<Node, Set<Node>> getCliques(Node[] ordering, Graph graph) {
        Map<Node, Set<Node>> cliques = new HashMap<>();
        for (int i = ordering.length - 1; i >= 0; i--) {
            Node v = ordering[i];

            Set<Node> clique = new HashSet<>();
            clique.add(v);

            for (int j = 0; j < i; j++) {
                Node w = ordering[j];
                if (graph.isAdjacentTo(v, w)) {
                    clique.add(w);
                }
            }

            cliques.put(v, clique);
        }

        // remove subcliques
        cliques.forEach((k1, v1) -> cliques.forEach((k2, v2) -> {
            if ((k1 != k2) && !(v1.isEmpty() || v2.isEmpty()) && v1.containsAll(v2)) {
                v2.clear();
            }
        }));

        // remove empty sets from map
        while (cliques.values().remove(Collections.EMPTY_SET)) {
            // empty.
        }

        return cliques;
    }

    /**
     * Apply Tarjan and Yannakakis (1984) fill in algorithm for graph
     * triangulation. An undirected graph is triangulated if every cycle of
     * length greater than 4 has a chord.
     *
     * @param graph    moral graph
     * @param ordering maximum cardinality ordering
     */
    public static void fillIn(Graph graph, Node[] ordering) {
        int numOfNodes = ordering.length;

        // in reverse order, insert edges between any non-adjacent neighbors that are lower numbered in the ordering.
        for (int i = numOfNodes - 1; i >= 0; i--) {
            Node v = ordering[i];

            // find pairs of neighbors with lower order
            for (int j = 0; j < i; j++) {
                Node w = ordering[j];
                if (graph.isAdjacentTo(v, w)) {
                    for (int k = j + 1; k < i; k++) {
                        Node x = ordering[k];
                        if (graph.isAdjacentTo(x, v)) {
                            graph.addUndirectedEdge(x, w); // fill in edge
                        }
                    }
                }
            }
        }
    }

    /**
     * Perform Tarjan and Yannakakis (1984) maximum cardinality search (MCS) to
     * get the maximum cardinality ordering.
     *
     * CHANGED: getAdjacentNodesSet() instead of getAdjacentNodes()
     *
     * @param graph moral graph
     * @return maximum cardinality ordering of the graph
     */
    public static Node[] getMaximumCardinalityOrdering(EdgeListGraph graph) {
        int numOfNodes = graph.getNumNodes();
        if (numOfNodes == 0) {
            return new Node[0];
        }

        Node[] ordering = new Node[numOfNodes];
        Set<Node> numbered = new HashSet<>(numOfNodes);
        for (int i = 0; i < numOfNodes; i++) {
            // find an unnumbered node that is adjacent to the most number of numbered nodes
            Node maxCardinalityNode = null;
            int maxCardinality = -1;
            for (Node v : graph.getNodesSet()) {
                if (!numbered.contains(v)) {
                    // count the number of times node v is adjacent to numbered node w
                    int cardinality = (int) graph.getAdjacentNodesSet(v).stream()
                            .filter(numbered::contains)
                            .count();

                    // find the maximum cardinality
                    if (cardinality > maxCardinality) {
                        maxCardinality = cardinality;
                        maxCardinalityNode = v;
                    }
                }
            }

            // add the node with maximum cardinality to the ordering and number it
            ordering[i] = maxCardinalityNode;
            numbered.add(maxCardinalityNode);
        }

        return ordering;
    }

    /**
     * Create a moral graph. A graph is moralized if an edge is added between
     * two parents with common a child and the edge orientation is removed,
     * making an undirected graph.
     *
     * @param graph to moralized
     * @return a moral graph
     */
    public static Graph moralize(Graph graph) {
        Graph moralGraph = new EdgeListGraph(graph.getNodes());

        // make skeleton
        graph.getEdges()
                .forEach(e -> moralGraph.addUndirectedEdge(e.getNode1(), e.getNode2()));

        // add edges to connect parents with common child
        graph.getNodes()
                .forEach(node -> {
                    List<Node> parents = graph.getParents(node);
                    if (!(parents == null || parents.isEmpty()) && parents.size() > 1) {
                        Node[] p = parents.toArray(new Node[0]);
                        for (int i = 0; i < p.length; i++) {
                            for (int j = i + 1; j < p.length; j++) {
                                Node node1 = p[i];
                                Node node2 = p[j];
                                if (!moralGraph.isAdjacentTo(node1, node2)) {
                                    moralGraph.addUndirectedEdge(node1, node2);
                                }
                            }
                        }
                    }
                });

        return moralGraph;
    }
}
