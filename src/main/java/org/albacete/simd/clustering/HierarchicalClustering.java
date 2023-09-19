package org.albacete.simd.clustering;

import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;
import org.albacete.simd.utils.Utils;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class HierarchicalClustering extends Clustering{

    private final Object lock = new Object();
    private Set<Edge> allEdges;
    private double[][] simMatrix;
    private boolean isParallel = false;
    private boolean isJoint = false;
    private final Map<Edge, Double> edgeScores = new ConcurrentHashMap<>();
    private List<Set<Node>> clusters;
    private final Map<Node, Set<Integer>> nodeClusterMap = new HashMap<>();

    public HierarchicalClustering(){

    }

    public HierarchicalClustering(Problem problem) {
        super(problem);
    }


    public HierarchicalClustering(Problem problem, boolean isParallel) {
        this(problem);
        this.isParallel = isParallel;
    }
    public HierarchicalClustering(boolean isParallel, boolean isJoint) {
        this.isParallel = isParallel;
        this.isJoint = isJoint;
    }
    

    public List<Set<Node>> generateNodeClusters(int numClusters) {
        //Initial setup
        System.out.println("Generating node clusters");
        if(edgeScores.isEmpty()) {
            System.out.println("Generating edge scores");
            calculateEdgesScore();
        }
        List<Node> nodes = problem.getVariables();

        //Initializing clusters
        System.out.println("Initializing clusters");
        clusters = new ArrayList<>(nodes.size());
        for (Node n : nodes) {
            Set<Node> s = new HashSet<>();
            s.add(n);
            clusters.add(s);
        }

        // Initializing Similarity Matrix
        System.out.println("Initializing similarity matrix");
        initializeSimMatrix();

        // Calculating clusters
        System.out.println("Calculating clusters");
        while (clusters.size() > numClusters) {

            // Initializing scores and initial positions of indexes
            double maxValue = Double.NEGATIVE_INFINITY;
            int posI = -1;
            int posJ = -1;

            //Checking which two clusters are better to merge together in parallel
            for (int i = 0; i < (clusters.size() - 1); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    if (simMatrix[i][j] > maxValue) {
                        maxValue = simMatrix[i][j];
                        posI = i;
                        posJ = j;
                    }
                }
            }
            // Merging the chosen clusters into posI of the clusters list.
            Set<Node> mergeCluster = new HashSet<>();
            mergeCluster.addAll(clusters.get(posI));
            mergeCluster.addAll(clusters.get(posJ));
            clusters.set(posI, mergeCluster);

            //Recalculating simMatrix
            for (int j = posI + 1; j < clusters.size(); j++) {
                if (j != posJ) {
                    simMatrix[posI][j] = getScoreClusters(clusters.get(posI), clusters.get(j));
                }
            }
            for (int i = 0; i < posI; i++) {
                simMatrix[i][posI] = getScoreClusters(clusters.get(i), clusters.get(posI));
            }

            // Deleting cluster and the information of posJ in simMatrix
            deleteClusterInSimMatrix(posJ);

        }

        // Creating joint clusters if necessary
        if(this.isJoint){
            System.out.println("Creating joint clusters");
            createJointClusters();
        }

        //Indexing the cluster nodes
        System.out.println("Indexing the cluster nodes");
        indexClusters();

        return clusters;
    }

    private void calculateEdgesScore(){
        // Calculating all the edges
        allEdges = Utils.calculateArcs(problem.getData());

        // Getting hashmap of indexes
        HashMap<Node, Integer> index = problem.getHashIndices();

        // Parallel for loop to calculate the scores of each edge with no other subset of parents
        allEdges.parallelStream().forEach((edge) -> {
            Node parent = edge.getNode1();
            Node child = edge.getNode2();
            int parentIndex = index.get(parent);
            int childIndex = index.get(child);

            HashSet<Node> hashParent = new HashSet<>();
            hashParent.add(parent);
            double score = GESThread.localBdeuScore(childIndex, new int[]{parentIndex}, hashParent, problem) -
                    GESThread.localBdeuScore(childIndex, new int[]{}, new HashSet<>(), problem);
            edgeScores.put(edge, score);
        });
    }

    private void initializeSimMatrix() {
        this.simMatrix = new double[problem.getVariables().size()][problem.getVariables().size()];
        if (isParallel)
            initializeSimMatrixParallel();
        else
            initializeSimMatrixSequential();

    }

    private void initializeSimMatrixParallel() {
        // Similarity matrix
        // Getting hashmap of indexes
        Map<Node, Integer> index = problem.getHashIndices();
        // Calculating similarity
        edgeScores.entrySet().parallelStream().forEach(edgeDoubleEntry -> {
                    int i = index.get(edgeDoubleEntry.getKey().getNode1()); //Getting node 1 index
                    int j = index.get(edgeDoubleEntry.getKey().getNode2()); // Getting node 2 index
                    double score = edgeDoubleEntry.getValue();              // Getting score of edge
                    addValueSimMatrix(i, j, score);                         //Adding value to the sim matrix
                }
        );
    }

    private void addValueSimMatrix(int i, int j, double score) {
        synchronized (lock) {
            simMatrix[i][j] = score;
        }
    }

    private void initializeSimMatrixSequential() {
        List<Node> nodes = problem.getVariables();
        int numNodes = problem.getVariables().size();
        for (int i = 0; i < (numNodes - 1); i++) {
            for (int j = i + 1; j < numNodes; j++) {
                // Getting Nodes
                Node x = nodes.get(i);
                Node y = nodes.get(j);
                // Getting Edge
                Edge edge = new Edge(x, y, Endpoint.TAIL, Endpoint.ARROW);
                // Getting score
                simMatrix[i][j] = edgeScores.get(edge);
            }
        }
    }

    private double getScoreClusters(Set<Node> cluster1, Set<Node> cluster2) {
        // Merged cluster
        Set<Node> mergeCluster = ConcurrentHashMap.newKeySet();//new HashSet<>();
        mergeCluster.addAll(cluster1);
        mergeCluster.addAll(cluster2);
        // initializing score
        double score = 0.0;
        // Sequential for loop

        List<Node> mergeClusterList = new ArrayList<>(mergeCluster);
        for (int i = 0; i < (mergeClusterList.size() - 1); i++) {
            for (int j = i + 1; j < mergeClusterList.size(); j++) {
                Node nodeI = mergeClusterList.get(i);
                Node nodeJ = mergeClusterList.get(j);
                Edge edge = new Edge(nodeI, nodeJ, Endpoint.TAIL, Endpoint.ARROW);
                //System.out.println(edge);
                score += edgeScores.get(edge);
            }
        }

        score /= ((double) mergeCluster.size() * (mergeCluster.size() - 1) / 2);
        return score;
    }

    public double getScoreDifference(Set<Node> cluster, Node node){
        double score = 0;
        for (Node n: cluster) {
            Edge edge = new Edge(n, node, Endpoint.TAIL, Endpoint.ARROW);
            score+= edgeScores.get(edge);
        }
        return score;
    }

    private void deleteClusterInSimMatrix(int posJ){
        // Deleting cluster in position posJ
        clusters.remove(posJ);
        // Making copy
        double[][]auxMatrix = Arrays.stream(simMatrix)
                .map(double[]::clone)
                .toArray(double[][]::new);
        // Reducing by one the size of the simMatrix since the size of the clusters is one less
        simMatrix = new double[clusters.size()][clusters.size()];

        // Row posJ and column posJ will be deleted
        int p = 0;
        for (int i = 0; i < auxMatrix.length; i++) {
            if(i == posJ)
                continue;
            int q = 0;
            for (int j = 0; j < auxMatrix.length; j++) {
                if(j == posJ)
                    continue;
                simMatrix[p][q] = auxMatrix[i][j];
                q++;
            }
            p++;
        }
    }



    private void createJointClusters(){
        double start = System.currentTimeMillis();
        //1. Calculating the number of variables that need to be in each cluster
        int maxVarsClusters = clusters.parallelStream().map(Set::size).max(Integer::compare).orElse(-1);

        //2. For each cluster, find the best nodes to add until the cluster size is equal to maxVarsClusters
        clusters.forEach(cluster -> {
            while(cluster.size() < maxVarsClusters){
                //2.1. Find the best node to add to the cluster
                Node bestNode = null;
                double bestScore = Double.NEGATIVE_INFINITY;
                for(Node node : problem.getVariables()){
                    if(!cluster.contains(node)){
                        //Set<Node> auxCluster = new HashSet<>(cluster);
                        //auxCluster.add(node);
                        //double score = getScoreClusters(auxCluster, cluster);
                        double score = getScoreDifference(cluster, node);
                        if(score > bestScore){
                            bestScore = score;
                            bestNode = node;
                        }
                    }
                }
                //2.2. Add the best node to the cluster
                cluster.add(bestNode);
            }
        });

        double end = System.currentTimeMillis();
        System.out.println("Time to create joint clusters: " + (end - start)/1000 + " seconds");
    }

    private void indexClusters() {
        for (int i = 0; i < clusters.size(); i++) {
            Set<Node> nodesCluster = clusters.get(i);
            for (Node node : nodesCluster) {
                Set<Integer> clusterIndexes;
                if(!nodeClusterMap.containsKey(node)) {
                    clusterIndexes = new HashSet<>();
                }
                else{
                    clusterIndexes = nodeClusterMap.get(node);
                }
                clusterIndexes.add(i);
                nodeClusterMap.put(node, clusterIndexes);
            }
        }
    }

    @Override
    public List<Set<Edge>> generateEdgeDistribution(int numClusters) {
        // Generating node clusters
        if(clusters == null) {
            generateNodeClusters(numClusters);
        }
        // Generating edge distribution
        System.out.println("Generating edge distribution");
        List<Set<Edge>> edgeDistribution = new ArrayList<>(clusters.size());
        for (int i = 0; i < clusters.size(); i++) {
            Set<Edge> edges = new HashSet<>();
            edgeDistribution.add(edges);
        }

        // Generating the Inner and Outer edges
        System.out.println("Generating inner and outer edges");
        allEdges.forEach(edge -> {
            Node node1 = edge.getNode1();
            Node node2 = edge.getNode2();

            // Both sets indicate where each node is located in each cluster.
            Set<Integer> clusterIndexes1 = nodeClusterMap.get(node1);
            Set<Integer> clusterIndexes2 = nodeClusterMap.get(node2);

            // To calculate the inner edges, we need to check where the nodes repeat themselves in the clusters by means of an intersection.
            Set<Integer> innerClusterIndexes = new HashSet<>(clusterIndexes1);
            innerClusterIndexes.retainAll(clusterIndexes2);
            // To calculate the outer edges, we need to check where the nodes don't repeat themselves in the clusters by means of a difference.
            Set<Integer> outerClusterIndexes = new HashSet<>(clusterIndexes1);
            outerClusterIndexes.addAll(clusterIndexes2);
            outerClusterIndexes.removeAll(innerClusterIndexes);

            // Adding the edge to each cluster in the inner-cluster indexes to create inner edges:
            for (Integer innerClusterIndex : innerClusterIndexes) {
                edgeDistribution.get(innerClusterIndex).add(edge);
            }

            // This adds the edge as an outer edge to only one cluster.
            // If the edge is an outer edge, now we add it to the smallest edgeDistribution cluster.
            if(outerClusterIndexes.size() > 0) {
                int minSize = Integer.MAX_VALUE;
                int minIndex = -1;
                for (Integer outerClusterIndex : outerClusterIndexes) {
                    if (edgeDistribution.get(outerClusterIndex).size() < minSize) {
                        minSize = edgeDistribution.get(outerClusterIndex).size();
                        minIndex = outerClusterIndex;
                    }
                }
                edgeDistribution.get(minIndex).add(edge);
            }

        });
        System.out.println("Finished generating edge distribution");

        return edgeDistribution;
    }


}
