package org.albacete.simd.mctsbn;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.BDeuScore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.albacete.simd.utils.ProblemMCTS;
import org.apache.commons.lang3.ArrayUtils;

public class HillClimbingEvaluator {

    private final ProblemMCTS problem;
    
    private final ConcurrentHashMap<String,Double> localScoreCache;

    private List<Integer> order;

    private Graph graph;

    private double finalScore = 0;

    private final static int MAX_ITERATIONS = 1000;

    private final BDeuScore metric;
    
    protected final double[] bestBDeuForNode;


    public HillClimbingEvaluator(ProblemMCTS problem, ConcurrentHashMap<String,Double> localScoreCache){
        this.problem = problem;
        this.localScoreCache = localScoreCache;
        this.graph = new EdgeListGraph_n();
        this.bestBDeuForNode = new double [problem.getVariables().size()];
        metric = new BDeuScore(problem.getData());
    }

    public HillClimbingEvaluator(ProblemMCTS problem, List<Integer> order, ConcurrentHashMap<String,Double> localScoreCache){
        this(problem, localScoreCache);
        setOrder(order);
    }



    public Pair evaluate(int child, Collection<Integer> candidates){
        int iteration = 0;

        //System.out.println("\n-------------- " + child + " --------------");
        
        Set<Integer> parents = new HashSet<>();
        double bdeuFinal = 0;

        while(iteration < MAX_ITERATIONS) {
            //System.out.println("\nITERATION " + iteration);
            AtomicReference<Double> bestScore = new AtomicReference<>(Double.NEGATIVE_INFINITY);
            AtomicInteger bestParent = new AtomicInteger();
            
            // Generate parents array
            Integer[] parentsArr = new Integer[parents.size()];
            parents.toArray(parentsArr);
            Arrays.sort(parentsArr);

            candidates.parallelStream().forEach(candidate -> {
                double score;

                // OPERATION ADD
                if (!parents.contains(candidate)) {
                    //System.out.println("ADD: " + candidate + " -> " + child + ", " + parents);
                    score = getAdditionScore(child, candidate, new HashSet<>(parents), parentsArr);
                }
                    
                // OPERATION DELETE
                else {
                    //System.out.println("DELETE: " + candidate + " -> " + child + ", " + parents);
                    score = getDeleteScore(child, candidate, new HashSet<>(parents), parentsArr);
                }
                if(score > bestScore.get()){
                    bestScore.set(score);
                    bestParent.set(candidate);
                }
                 
            });
            
            // Updating graph
            if(bestScore.get() > 0){
                int bp = bestParent.get();
                if(parents.contains(bp)) {
                    parents.remove(bp);
                    //System.out.println("   BEST PARENT DEL: " + bp + ", SCORE: " + bestScore.get());
                }
                else {
                    parents.add(bp);
                    //System.out.println("   BEST PARENT ADD: " + bp + ", SCORE: " + bestScore.get());
                }
                iteration++;
                bdeuFinal += bestScore.get();
            } 
            else {
                //System.out.println("     FIN DE LA ITERACIÓN: " + bestParent.get() + ", SCORE: " + bestScore.get());
                break;
            }
        }
        
        // Updating best BDeu for Node
        if (bdeuFinal > bestBDeuForNode[child]) {
            bestBDeuForNode[child] = bdeuFinal;
        }
        
        return new Pair(child, parents, bdeuFinal);
    }

    
    public double getAdditionScore(int indexChild, int indexParent, Set<Integer> parents, Integer[] indexParents) {
        // Creating an array adding the index of the parent
        int[] indexUnion = new int[indexParents.length + 1];
        for (int i = 0; i < indexParents.length; i++) {
            indexUnion[i] = indexParents[i];
        }
        indexUnion[indexUnion.length - 1] = indexParent;
        parents.add(indexParent);
        if (indexUnion.length > 1)
            Arrays.sort(indexUnion);
        
        double scorePart1 = localBdeuScore(indexChild, indexUnion);
        
        // Removing again the parent to the set
        parents.remove(indexParent);
        double scorePart2 = localBdeuScore(indexChild, ArrayUtils.toPrimitive(indexParents));
        
        // Score = localbdeu(x,P(G) + {x_p}) - localbdeu(x,P(G))
        return scorePart1 - scorePart2;
    }


    public double getDeleteScore(int indexChild, int indexParent, Set<Integer> parents, Integer[] indexParents) {
        // Calculating indexes for the difference set of parents
        parents.remove(indexParent);    
        Integer[] indexParentsAux = new Integer[parents.size()];
        parents.toArray(indexParentsAux);
        if (indexParentsAux.length > 1)
            Arrays.sort(indexParentsAux);

        double scorePart1 = localBdeuScore(indexChild, ArrayUtils.toPrimitive(indexParentsAux));
        
        // Adding again the parent to the set
        parents.add(indexParent);
        double scorePart2 = localBdeuScore(indexChild, ArrayUtils.toPrimitive(indexParents));
        
        // Score = localbdeu(x,P(G) - {x_p}) - localbdeu(x,P(G))
        return scorePart1 - scorePart2;
    }


    public double localBdeuScore(int nNode, int[] nParents) {
        Double oldScore = localScoreCache.get(nNode + Arrays.toString(nParents));
        //problem.counter.getAndIncrement();

        if (oldScore != null) {
            return oldScore;
        }
        
        //problem.counterSinDict.getAndIncrement();
        
        double fLogScore = metric.localScore(nNode, nParents);
        localScoreCache.put(nNode + Arrays.toString(nParents), fLogScore);

        return fLogScore;
    }

    public double search(){
        graph = new EdgeListGraph_n(problem.getVariables());
        finalScore = 0;
        
        Set<Integer> candidates = new HashSet<>();
        for (int node : order) {
            Set<Integer> parents = evaluate(node, candidates).set;
            for (int parent : parents) {
                Edge edge = Edges.directedEdge(problem.getNode(parent), problem.getNode(node));
                graph.addEdge(edge);
            }
            
            Integer[] arr = parents.toArray(new Integer[0]);
            finalScore += localBdeuScore(node, ArrayUtils.toPrimitive(arr));
            
            candidates.add(node);
        }
        
        //System.out.println("Obtained result: " + finalScore + "\t-> " + order);

        
        //System.out.println("\n TOTAL CALCULATIONS:  " + (problem.counter.get() - c1));
        //System.out.println(" TOTAL CALCULATIONS no dictionary:  " + (problem.counterSinDict.get() - c2));
        //System.out.println(" SIZE OF CONCURRENT: " + localScoreCache.size());
        
        return finalScore;
    }
    

    public double getScore() {
        return finalScore;
    }

    public Graph getGraph() {
        return graph;
    }
    
    public final void setOrder(List<Integer> order) {
        this.order = order;
    }
     
    public ArrayList<Integer> nodeToIntegerList(List<Node> nodes){
        ArrayList<Integer> integers = new ArrayList(nodes.size());
        for (Node node : nodes) {
            integers.add(nodeToInteger(node));
        }
        return integers;
    }
    
    public int nodeToInteger(Node node){
        return problem.getHashIndices().get(node);
    }

    public int[] nodesToInteger(List<Node> nodes){
        int[] integers = new int[nodes.size()];
        for (int i = 0; i < nodes.size(); i++) {
            integers[i] = nodeToInteger(nodes.get(i));
        }
        return integers;
    }
    
    public static class Pair implements Comparable<Pair> {
        public final int node;
        public final Set set;
        public final double bdeu;

        public Pair(int node, Set a, double b) {
            this.node = node;
            this.set = a;
            this.bdeu = b;
        }
        
        @Override
        public int compareTo(Pair o) {
            return Double.compare(this.bdeu, o.bdeu);
        }
    }
    
}
