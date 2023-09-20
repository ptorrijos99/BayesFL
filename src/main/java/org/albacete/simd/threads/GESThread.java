package org.albacete.simd.threads;

import consensusBN.SubSet;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.MeekRules;
import edu.cmu.tetrad.search.SearchGraphUtils;
import edu.cmu.tetrad.util.NumberFormatUtil;
import org.albacete.simd.framework.BackwardStage;
import org.albacete.simd.framework.ForwardStage;
import org.albacete.simd.utils.LocalScoreCacheConcurrent;
import org.albacete.simd.utils.Problem;

import java.text.NumberFormat;
import java.util.*;
import static org.albacete.simd.utils.Utils.pdagToDag;

/*
  GESThread is an abstract class that encapsulates the common attributes and methods of the threads executed in both the FES
  and BES stage. For future versions, there could be more types of threads other than {@link ThFES ThFES} and {@link ThBES ThBES}.
 */
//@SuppressWarnings({"DuplicatedCode", "unused"})
public abstract class GESThread implements Runnable{
    /**
     * Set of edges that will be checked over the GESThread
     */
    protected Set<Edge> S;
    
    protected Set<Edge> enlaces;
    
    /**
     * Set of edges that will be checked over the GESThread
     */
    protected Set<EdgeSearch> scores = new HashSet<>();

    /**
     * Problem the thread is solving
     */
    protected Problem problem = null;

    /**
     * Initial DAG this thread starts with. It can be null or a resulting DAG of an iteration.
     */
    protected Graph initialDag = null;
    /**
     * The current DAG with which the thread is working. This is a modified DAG of the initial DAG.
     */
    protected Graph currentGraph = null;

    /**
     * Flag indicating if the thread is working or not.
     */
    protected boolean flag = false;


    /**
     * For formatting printed numbers.
     */
    protected final NumberFormat nf = NumberFormatUtil.getInstance().getNumberFormat();

    /**
     * Total calls done
     */
    protected static int numTotalCalls=0;

    /**
     * Total calls done to non-cached information
     */
    protected static int numNonCachedCalls=0;

    /**
     * Elapsed time of the most recent search.
     */
    protected long elapsedTime;

    /**
     * Start time of the search
     */
    protected long startTime;

    /**
     * True if cycles are to be aggressively prevented. May be expensive
     * for large graphs (but also useful for large graphs).
     */
    protected boolean aggressivelyPreventCycles = false;

    /**
     * Maximum number of edges.
     */
    protected int maxNumEdges = -1;

    /**
     * Score of the bdeu model.
     */
    protected double modelBDeu;

    /**
     * Maximum number of iterations.
     */
    protected int maxIt;

    /**
     * Total number of iterations done.
     */
    protected int iterations = 0;
    /**
     * Node X from the edge X->Y that can be inserted in the graph.
     */
    protected Node x_i;

    /**
     * Node Y from the edge X->Y that can be inserted in the graph.
     */
    protected Node y_i;

    /**
     * Subset T of Nodes with which X and Y are inserted.
     */
    protected SubSet t_0;

    /**
     * Node X from the edge X->Y that can be deleted from the graph.
     */
    protected Node x_d;

    /**
     * Node Y from the edge X->Y that can be deleted from the graph.
     */
    protected Node y_d;

    /**
     * Subset H of Nodes with which X and Y are deleted.
     */
    protected SubSet h_0;

    /**
     * Id of the thread
     */
    protected int id = -1;

    /**
     * Boolean value that says if the thread is from a forward stage (true) or from a backwards stage (false)
     */
    protected boolean isForwards;

    /**
     * Evaluate the Insert(X, Y, T) operator (@see <a href="http://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf"> Definition 12 from Chickering 2002</a>,
     * ).
     * @param x First {@link Node Node} of the edge being considered to insert the graph.
     * @param y Second {@link Node Node} of the edge being considered to insert the graph.
     * @param t Set of {@link Node Nodes} to be considered when making the insertion.
     * @param graph Current {@link Graph Graph} of the stage.
     * @return Score difference of the insertion.
     */
    public static double insertEval(Node x, Node y, Set<Node> t, Graph graph, Problem problem) {
        // set1 contains x; set2 does not.
        Set<Node> set1;
//        if(t.isEmpty()){
//            set1 = new HashSet<>();
//        }
//        else{
            set1 = new HashSet<>(findNaYX(x, y, graph));
//        }
        set1.addAll(t);
        set1.addAll(graph.getParents(y));
        Set<Node> set2 = new HashSet<>(set1);
        set1.add(x);
        return scoreGraphChange(y, set1, set2, graph, problem);
    }



    /**
     * Do an actual insertion of an edge.
     * (@see <a href="http://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf"> Definition 12 from Chickering 2002</a>).
     * @param x First {@link Node Node} of the edge being inserted into the graph.
     * @param y Second {@link Node Node} of the edge being inserted into the graph.
     * @param subset Set of {@link Node Nodes} that allow the insertion.
     * @param graph Current {@link Graph Graph} of the stage where the edge will be inserted into.
     */
    public static void insert(Node x, Node y, Set<Node> subset, Graph graph) {
        //System.out.println("Insert: " + x + " -> " + y);
        graph.addDirectedEdge(x, y);

        for (Node node : subset) {
            //System.out.println("Delete: " + node + " -- " + y);
            graph.removeEdge(node, y);
            //System.out.println("Insert: " + node + " -> " + y);
            graph.addDirectedEdge(node, y);
        }
    }

    /**
     * Do an actual deletion (@see <a href="http://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf"> Definition 13 from Chickering 2002</a>).
     * @param x First {@link Node Node} of the edge being deleted from the graph.
     * @param y Second {@link Node Node} of the edge being deleted from the graph.
     * @param subset Set of {@link Node Nodes} that allow the deletion.
     * @param graph Current {@link Graph Graph} of the stage where the edge will be deleted from.
     */
    public static void delete(Node x, Node y, Set<Node> subset, Graph graph) {
        //System.out.println("Delete: " + x + " -- " + y);
        graph.removeEdges(x, y);

        for (Node aSubset : subset) {
            //System.out.println("Deleting Edges in subset");
            if (!graph.isParentOf(aSubset, x) && !graph.isParentOf(x, aSubset)) {
                //System.out.println("Delete: " + x + " -- " + aSubset);
                graph.removeEdge(x, aSubset);
                //System.out.println("Insert: " + x + " -> " + aSubset);
                graph.addDirectedEdge(x, aSubset);
            }
            //System.out.println("Delete: " + y + " -- " + aSubset);
            graph.removeEdge(y, aSubset);
            //System.out.println("Insert: " + y + " -> " + aSubset);
            graph.addDirectedEdge(y, aSubset);
        }
    }

    /**
     * Evaluate the Delete(X, Y, T) operator (@see <a href="http://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf"> Definition 13 from Chickering 2002</a>).
     * @param x First {@link Node Node} of the edge being considered for deletion.
     * @param y Second {@link Node Node} of the edge being considered for deletion.
     * @param h Set of {@link Node Nodes} to be considered when making the deletion.
     * @param graph Current {@link Graph Graph} of the stage.
     * @return Score difference of the deletion.
     */
    protected double deleteEval(Node x, Node y, Set<Node> h, Graph graph) {
        Set<Node> set1 = new HashSet<>(findNaYX(x, y, graph));
        set1.removeAll(h);
        set1.addAll(graph.getParents(y));
        Set<Node> set2 = new HashSet<>(set1);
        set1.remove(x);
        set2.add(x);
        return scoreGraphChange(y, set1, set2, graph, problem);
    }

    /**
     * Find all the nodes that are connected to Y by an undirected edge that are
     * adjacent to X (that is, by undirected or directed edge) NOTE: very
     * inefficient implementation, since the current library does not allow
     * access to the adjacency list/matrix of the graph.
     * @param x Node X
     * @param y Node Y
     * @param graph current graph of the problem
     * @return List of nodes that are connected to Y and adjacent to X.
     */
    protected static List<Node> findNaYX(Node x, Node y, Graph graph) {
        List<Node> adjY = graph.getAdjacentNodes(y);
        List<Node> naYX = new LinkedList<>(adjY);
        naYX.retainAll(graph.getAdjacentNodes(x));
        
        for (int i = naYX.size() - 1; i >= 0; i--) {
            Node z = naYX.get(i);
            Edge edge = graph.getEdge(y, z);

            if (!Edges.isUndirectedEdge(edge)) {
                naYX.remove(z);
            }
        }

        return naYX;
    }

    /**
     * Returns true if the given set forms a clique in the given graph.
     * @param set Set of Nodes that is being checked for a clique
     * @param graph Current graph of the stage
     * @return true if there is a clique, false otherwise.
     */
    protected static boolean isClique(List<Node> set, Graph graph) {
        List<Node> setNeighbors = new LinkedList<>(set);
        for (int i = 0; i < setNeighbors.size() - 1; i++) {
            for (int j = i + 1; j < setNeighbors.size(); j++) {
                if (!graph.isAdjacentTo(setNeighbors.get(i), setNeighbors.get(j))) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Verifies if every semi-directed path from y to x contains a node in naYXT.
     * @param x Starting node of the test.
     * @param y Ending node of the test
     * @param naYXT List of nodes being checked to be in the semi-directed path between x and y.
     * @param graph Current graph of the stage.
     * @param marked Set of Nodes being stored to avoid cycles.
     * @return true if there is a semi-directed path between x and y that contains a node stored in naYXT,
     * false if there isn't.
     */
    protected static boolean isSemiDirectedBlocked(Node x, Node y, List<Node> naYXT,
                                          Graph graph, Set<Node> marked) {
        if (naYXT.contains(y)) {
            return true;
        }

        if (y == x) {
            return false;
        }
        
        // nueva forma de calcular los caminos semidirigidos

        LinkedHashSet<Node> open = new LinkedHashSet<>();
        open.add(y);
        HashSet<Node> close = new HashSet<>(naYXT);
        
        while(!open.isEmpty()) {
            Node a = open.iterator().next();
            if(a.equals(x)) return false;
            open.remove(a);
            close.add(a);
            
            for(Edge e: graph.getEdges(a)) {
                Node n = e.getNode1();
                Node n2 = e.getNode2();
                if (n.equals(a)) n = n2;
                if (!close.contains(n) && !open.contains(n) && !e.pointsTowards(a)) 
                    open.add(n);
            }
        }
        return true;

        /*for (Node node1 : graph.getNodes()) {
            if (node1 == y || marked.contains(node1)) {
                continue;
            }
            
            if (graph.isAdjacentTo(y, node1) && !graph.isParentOf(node1, y)) {
                marked.add(node1);

                if (!isSemiDirectedBlocked(x, node1, naYXT, graph, marked)) {
                    return false;
                }

                marked.remove(node1);
            }
        }

        return true;*/
    }
    
    /**
     * Completes a pattern that was modified by an insertion/deletion operator
     * Based on the algorithm described on @see <a href="http://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf">Appendix C of (Chickering, 2002)</a>.
     * @param graph Graph being rebuilt.
     */
    protected void rebuildPattern(Graph graph) {
        SearchGraphUtils.basicCPDAG(graph);
        pdag(graph);
    }

    /**
     * Fully direct a graph with background knowledge. I am not sure how to
     * adapt Chickering's suggested algorithm above (dagToPdag) to incorporate
     * background knowledge, so I am also implementing this algorithm based on
     * Meek's 1995 UAI paper. Notice it is the same implemented in PcSearch.
     * </p> *IMPORTANT!* *It assumes all colliders are oriented, as well as
     * arrows dictated by time order.*
     *
     * ELIMINATING BACKGROUND KNOWLEDGE
     *
     * @param graph Graph being transformed into a PDAG.
     */
    protected void pdag(Graph graph) {
        MeekRules rules = new MeekRules();
        rules.setAggressivelyPreventCycles(this.aggressivelyPreventCycles);
        rules.orientImplied(graph);
    }




    //===========================SCORING METHODS===========================//

    /**
     * Scores a DAG using the BDeu score function
     * @param graph DAG graph being evaluated
     * @return score of the graph.
     */
    public static double scoreGraph(Graph graph, Problem problem) {

        if (graph == null){
            return Double.NEGATIVE_INFINITY;
        }

//        Graph dag = SearchGraphUtils.dagFromPattern(graph);
        Graph dag = new EdgeListGraph_n(graph);
        pdagToDag(dag);
        double score = 0.;

        for (Node next : dag.getNodes()) {
            Set<Node> parents = new HashSet<>(dag.getParents(next));
            int nextIndex = -1;
            for (int i = 0; i < problem.getVariables().size(); i++) {
                String[] varNames = problem.getVarNames();
                if (varNames[i].equals(next.getName())) {
                    nextIndex = i;
                    break;
                }
            }
            int[] parentIndices = new int[parents.size()];
            Iterator<Node> pi = parents.iterator();
            int count = 0;
            while (pi.hasNext()) {
                Node nextParent = pi.next();
                for (int i = 0; i < problem.getVariables().size(); i++) {
                    String[] varNames = problem.getVarNames();
                    if (varNames[i].equals(nextParent.getName())) {
                        parentIndices[count++] = i;
                        break;
                    }
                }
            }
                score += localBdeuScore(nextIndex, parentIndices, parents, problem);
            }
        return score;
    }
    
    public double scoreDag(Graph graph) {
        
        if (graph == null){
            return Double.NEGATIVE_INFINITY;
        }
        
        Graph dag = new EdgeListGraph_n(graph);
        pdagToDag(dag);
        HashMap<Node,Integer> hashIndices = problem.getHashIndices();

        double _score = 0;

        for (Node node : getVariables()) {
            List<Node> x = dag.getParents(node);

            int[] parentIndices = new int[x.size()];

            int count = 0;
            for (Node parent : x) {
                parentIndices[count++] = hashIndices.get(parent);
            }

            final double nodeScore = problem.getBDeu().localScore(hashIndices.get(node), parentIndices);

            _score += nodeScore;
        }
        return _score;
    }

    /**
     * Score difference of the node y when it is associated as child with one set of parents and when it is associated with another one.
     * @param y {@link Node Node} being considered for the score.
     * @param parents1 Set of {@link Node Node} of the first set of parents.
     * @param parents2 Set of {@link Node Node} of the second set of parents.
     * @return Score difference between both possibilities.
     */
    public static double scoreGraphChange(Node y, Set<Node> parents1,
                                   Set<Node> parents2, Graph graph, Problem problem) {

        /*
        // Creating hashindeces map
        HashMap<Node, Integer> index = new HashMap<>();

        // Building map
        for (Node next : graph.getNodes()) {
            for (int i = 0; i < varNames.length; i++) {
                if (varNames[i].equals(next.getName())) {
                    index.put(next, i);
                    break;
                }
            }
        }
        */

/*
        System.out.println("DEBUG Score Graph Change of:");
        System.out.println("Node: " + y);
        System.out.println("Parents1: " + parents1);
        System.out.println("Parents2: " + parents2);
        System.out.println("HashIndices: " + index);
        System.out.println("-------------------------------");
*/

        HashMap<Node, Integer> index = problem.getHashIndices();
/*
        if(index == null){
            System.out.println("REBUILDING HASHINDECES IN GRAPHSCORESCHANGE");
            problem.buildIndexing(graph);
            index = problem.getHashIndices();
        }
*/
        // Getting indexes
        int yIndex = index.get(y);
        int[] parentIndices1 = new int[parents1.size()];

        int count = 0;
        for (Node aParents1 : parents1) {
            parentIndices1[count++] = (index.get(aParents1));
        }

        int[] parentIndices2 = new int[parents2.size()];

        int count2 = 0;
        for (Node aParents2 : parents2) {
            parentIndices2[count2++] = (index.get(aParents2));
        }

        // Calculating the scores of both possibilities and returning the difference
        double score1 = localBdeuScore(yIndex, parentIndices1, parents1, problem);
        double score2 = localBdeuScore(yIndex, parentIndices2, parents2, problem);
        return score1 - score2;
    }



    /**
     * Bdeu Score function for a {@link Node node} and a set of parent nodes.
     *
     * @param nNode    index of the child node
     * @param nParents index of the parents of the node being considered as child.
     * @param setParents set with the parent of the node being considered as child.
     * @return The Bdeu score of the combination.
     */

    public static double localBdeuScore(int nNode, int[] nParents, Set<Node> setParents, Problem problem) {
        numTotalCalls++;

        LocalScoreCacheConcurrent localScoreCache = problem.getLocalScoreCache();

        double oldScore = localScoreCache.get(nNode, setParents);
        if (!Double.isNaN(oldScore)) {
            return oldScore;
        }
        numNonCachedCalls++;

        double fLogScore = problem.getBDeu().localScore(nNode, nParents);
        
        /*  int[] nValues = problem.getnValues();
            int[][] cases = problem.getCases();


            int numValues=nValues[nNode];
            int numParents=nParents.length;

            double ess= problem.getSamplePrior();
            double essPenalty = 1.0;
            double kappa= problem.getStructurePrior();

            int[] numValuesParents=new int[nParents.length];
            int cardinality=1;
            for(int i=0;i<numValuesParents.length;i++) {

                numValuesParents[i]=nValues[nParents[i]];
                cardinality*=numValuesParents[i];
            }

            int[][] Ni_jk = new int[cardinality][numValues];
    
            double Np_ijk = (essPenalty * ess) / (numValues*cardinality);
            double Np_ij = (essPenalty *ess) / cardinality;

            // initialize
            for (int j = 0; j < cardinality;j++)
                for(int k= 0; k<numValues; k++)
                    Ni_jk[j][k] = 0;

            for (int[] aCase : cases) {
                int iCPT = 0;
                for (int iParent = 0; iParent < numParents; iParent++) {
                    iCPT = iCPT * numValuesParents[iParent] + aCase[nParents[iParent]];
                }
                Ni_jk[iCPT][aCase[nNode]]++;
            }

            for (int iParent = 0; iParent < cardinality; iParent++) {
                double N_ij = 0;
                double N_ijk;

                for (int iSymbol = 0; iSymbol < numValues; iSymbol++) {
                    if (Ni_jk[iParent][iSymbol] != 0) {
                        N_ijk = Ni_jk[iParent][iSymbol];
                        fLogScore += ProbUtils.lngamma(N_ijk + Np_ijk);
                        fLogScore -= ProbUtils.lngamma(Np_ijk);
                        N_ij += N_ijk;
                    }
                }
                if (Np_ij != 0)
                    fLogScore += ProbUtils.lngamma(Np_ij);
                if (Np_ij + N_ij != 0)
                    fLogScore -= ProbUtils.lngamma(Np_ij + N_ij);
            }
            fLogScore += Math.log(kappa) * cardinality * (numValues - 1);
        */

        localScoreCache.add(nNode, setParents, fLogScore);
        return fLogScore;
    }
    
    

    //==========================SETTERS AND GETTERS=========================//

    /**
     * Sets the initial graph with the graph passed as argument.
     * @param currentGraph The Graph we want to set.
     */
    public void setInitialGraph(Graph currentGraph){
        this.initialDag = currentGraph;
    }

    /**
     * Gets the initial Dag_n of the Thread
     * @return the initial Dag_n
     */
    public Graph getInitialGraph(){
        return this.initialDag;
    }

    /**
     * Gets the variables of the DataSet
     * @return list of Nodes of the problem's DataSet
     */
    public List<Node> getVariables() {
        return problem.getVariables();
    }

    /**
     * Gets the structurePrior
     * @return the structurePrior
     */
    public double getStructurePrior() {
        return problem.getStructurePrior();
    }

    /**
     * Gets the samplePrior
     * @return the samplePrior
     */
    public double getSamplePrior() {
        return problem.getSamplePrior();
    }

    /**
     * Sets the structurePrior
     * @param structurePrior the structurePrior to be set
     */
    public void setStructurePrior(double structurePrior) {
        this.problem.setStructurePrior(structurePrior);
    }

    /**
     * Sets the samplePrior
     * @param samplePrior the samplePrior of the thread
     */
    public void setSamplePrior(double samplePrior) {
        this.problem.setSamplePrior(samplePrior);
    }

    /**
     * Gets the elapsed time the thread has been executed.
     * @return the elapsed time.
     */
    public long getElapsedTime() {
        return elapsedTime;
    }


    /**
     * Gets the maximum number of edges allowed
     * @return the maximum number of edges
     */
    public int getMaxNumEdges() {
        return maxNumEdges;
    }

    /**
     * Sets the maximum number of edges
     * @param maxNumEdges maximum number of edges to be set.
     */
    public void setMaxNumEdges(int maxNumEdges) {
        if (maxNumEdges < -1)
            throw new IllegalArgumentException();

        this.maxNumEdges = maxNumEdges;
    }



    /**
     * Sets the maximum iterations the thread can do. This is used in {@link FESThread ThFES} threads.
     * @param maxIt the maximum number of iterations
     */
    public void setMaxIt(int maxIt) {
        this.maxIt = maxIt;
    }

    /**
     * Sets the subset that is searched by the thread.
     *
     * @param subset Subset of {@link Edge Edges}.
     */
    public void setSubSetSearch(Set<Edge> subset) {
        this.S = subset;

    }

    /**
     * Checking if the thread is aggressively preventing cycles
     * @return true if it prevents cycles, false otherwise.
     */
    public boolean isAggressivelyPreventCycles() {
        return this.aggressivelyPreventCycles;
    }

    /**
     * Sets the aggressivelyPreventCycles boolean value
     * @param aggressivelyPreventCycles true if it prevents cycles, false otherwise.
     */
    public void setAggressivelyPreventCycles(boolean aggressivelyPreventCycles) {
        this.aggressivelyPreventCycles = aggressivelyPreventCycles;
    }

    /**
     * Gets the BDeu score of the current Graph.
     * @return the BDeu score of the current Graph.
     */
    public double getScoreBDeu() {
        return this.modelBDeu;
    }

    //-----------------------------THREAD METHODS--------------------------------//

    /**
     * Gets the flag of the thread that indicates if the thread has added or deleted an edge in it's algorithm stage.
     * @return the flag of the thread
     * @throws InterruptedException Exception caused by external interruption
     */
    public synchronized boolean getFlag() throws InterruptedException{
        // While searching is not complete, wait
        while (currentGraph == null)
            wait();
        return this.flag;
    }

    /**
     * Resets the flag of the thread. The flag value is changed to false.
     */
    public synchronized void resetFlag(){
        this.flag = false;
    }

    /**
     * Synchronized getter for the currentGraph. Since the thread could still be working when called, a synchronization
     * needs to be established.
     * @return the currentGraph done by the thread
     * @throws InterruptedException Exception caused by external interruptions.
     */
    public synchronized Graph getCurrentGraph() throws InterruptedException{
        // This is a thread method that needs to be synchronized.
        // While searching is not complete, wait
        while (currentGraph == null)
            wait();
        return this.currentGraph;
    }



    /**
     * Get all nodes that are connected to Y by an undirected edge and not
     * adjacent to X.
     * @param x {@link Node Node} X, where the resulting nodes are neighbors of the {@link Node Node} Y, but not of X.
     * @param y {@link Node Node} Y, where the resulting nodes are neighbors of the {@link Node Node} Y, but not of X.
     * @param graph {@link Graph Graph} of the current constructed graph.
     * @return {@link List List} of {@link Node Nodes} that are neighbors of the {@link Node Node} Y, but not of {@link Node Node} X.
     */
    public static List<Node> getSubsetOfNeighbors(Node x, Node y, Graph graph) {
        List<Node> tNeighbors = new LinkedList<>(graph.getAdjacentNodes(y));
        tNeighbors.removeAll(graph.getAdjacentNodes(x));

        for (int i = tNeighbors.size() - 1; i >= 0; i--) {
            Node z = tNeighbors.get(i);
            Edge edge = graph.getEdge(y, z);
            
            if (edge != null && !Edges.isUndirectedEdge(edge)) {
                tNeighbors.remove(z);
            }
        }

        return tNeighbors;
    }

    public int getIterations(){
        return this.iterations;
    }


    public int getId(){
        return this.id;
    }

    public LocalScoreCacheConcurrent getLocalScoreCache() {
        return problem.getLocalScoreCache();
    }

    protected boolean isTimeout(){
        //return ((timeout != -1) &&
        //        ((System.currentTimeMillis() - startTime) > timeout));
        long time = (System.currentTimeMillis() - startTime);
        double zScore = 0;
        //System.out.println("Forward Meantime: " + ForwardStage.meanTimeTotal);
        //System.out.println("Backward Meantime: " + BackwardStage.meanTimeTotal);

        if(isForwards) {
            if (ForwardStage.meanTimeTotal != 0)
                zScore = (time - ForwardStage.meanTimeTotal) / Math.sqrt(ForwardStage.varianceTimeTotal);
        }
        else{
            if (BackwardStage.meanTimeTotal != 0)
                zScore = (time - BackwardStage.meanTimeTotal) / Math.sqrt(BackwardStage.varianceTimeTotal);

        }
        if(zScore > 3)
            System.out.println("Timeout! Finishing Thread");
        return zScore > 3;

    }
}
