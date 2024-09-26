package org.albacete.simd.mctsbn;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.search.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import edu.cmu.tetrad.search.score.IndTestScore;
import edu.cmu.tetrad.search.test.IndTestChiSquare;
import org.albacete.simd.algorithms.bnbuilders.GES_BNBuilder;
import org.albacete.simd.algorithms.bnbuilders.PGESwithStages;
import org.albacete.simd.clustering.Clustering;
import org.albacete.simd.clustering.HierarchicalClustering;
import org.albacete.simd.framework.BNBuilder;
import org.albacete.simd.utils.Utils;
import org.albacete.simd.utils.Problem;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class MCTSBN {

    /**
     * Time limit in miliseconds
     */
    private int TIME_LIMIT;
    /**
     * Iteration limit of the search
     */
    private int ITERATION_LIMIT;

    /**
     * Exploration constant c for the UCT equation: UCT_j = X_j + c * sqrt(ln(N) / n_j)
     */
    public double EXPLORATION_CONSTANT = 1 * Math.sqrt(2); //1.0 / Math.sqrt(2);
    
    public double EXPLOITATION_CONSTANT = 50;
    
    public double PROBABILITY_SWAP = 0.25;

    public double NUMBER_SWAPS = 0.25;

    public final int NUM_ROLLOUTS = 1;

    public final int NUM_SELECTION = 1;

    public final int NUM_EXPAND = 1;

    /**
     * Problem of the search
     */
    private final Problem problem;
    private final ArrayList<Integer> allVars;
    public final HillClimbingEvaluator hc;

    private TreeNode root;

    private double bestScore = Double.NEGATIVE_INFINITY;
    private List<Integer> bestOrder = new ArrayList<>();

    private Graph bestDag = null;

    private boolean convergence = false;

    private final ConcurrentHashMap<String,Double> cache;

    private final HashSet<TreeNode> selectionSet = new HashSet<>();

    private final Random random = new Random();
    
    private double mean;
    private double standardDeviation;

    private Dag initializeDag;
    
    private final ArrayList<ArrayList> orderSet = new ArrayList<>();

    public double PGESTime;

    public String initializeAlgorithm;

    // Write results in each round
    File file;
    BufferedWriter csvWriter;
    String firstPart;


    public MCTSBN(Problem problem, int iterationLimit){
        this.problem = problem;
        this.cache = problem.getConcurrentHashMap();
        this.ITERATION_LIMIT = iterationLimit;
        this.hc = new HillClimbingEvaluator(problem, cache);
        this.allVars = hc.nodeToIntegerList(problem.getVariables());
    }

    public MCTSBN(Problem problem, int iterationLimit, double exploitConstant, double numberSwaps, double probabilitySwap, String initializeAlgorithm){
        this.problem = problem;
        this.cache = problem.getConcurrentHashMap();
        this.ITERATION_LIMIT = iterationLimit;
        this.hc = new HillClimbingEvaluator(problem, cache);
        this.allVars = hc.nodeToIntegerList(problem.getVariables());

        this.initializeAlgorithm = initializeAlgorithm;

        this.EXPLOITATION_CONSTANT = exploitConstant;
        this.NUMBER_SWAPS = numberSwaps;
        this.PROBABILITY_SWAP = probabilitySwap;
     }

    public Dag search(){
        // 1. Initialize the writer
        initializeWriter();

        if (root == null) {
            root = new TreeNode(problem, this);
            long startTime = System.currentTimeMillis();
            expandFirstRound();

            long endTime = System.currentTimeMillis();
            // Calculating time of the iteration
            double totalTimeRound = 1.0 * (endTime - startTime) / 1000;
            saveRound(String.valueOf(1), totalTimeRound);
        }

        //System.out.println("\n\nSTARTING MCTSBN\n------------------------------------------------------");

        double lastScore = this.bestScore;
        // Search loop
        for (int i = 2; i < ITERATION_LIMIT; i++) {
            // Executing round
            long startTime = System.currentTimeMillis();
            executeRound();
            long endTime = System.currentTimeMillis();
            // Calculating time of the iteration
            double totalTimeRound = 1.0 * (endTime - startTime) / 1000;

            if (this.bestScore > lastScore) {
                lastScore = this.bestScore;
                saveRound(String.valueOf(i), totalTimeRound);
            }

            if(convergence){
                System.out.println("Convergence has been found. Ending search");
                break;
            }
        }

        try {
            csvWriter.flush();
            csvWriter.close();
        } catch (IOException ex) { ex.printStackTrace(); }

        return new Dag(bestDag);
    }

    public void expandFirstRound() {
        // 2. Add PGES order
        switch (this.initializeAlgorithm) {
            default:
            case "HC":
                initializeWithHC();
                break;
            case "GES":
                initializeWithGES();
                break;
            case "pGES":
                initializeWithPGES(4);
                break;
            case "fGES":
                initializeWithfGES();
                break;
            case "PC":
                initializeWithPC();
                break;
            case "CPC":
                initializeWithCPC();
                break;
        }

        // 3. Create a node for each variable (totally expand root). Implicit warmup
        ArrayList<TreeNode> selection = new ArrayList<>();
        selection.add(this.root);

        double random_const = PROBABILITY_SWAP;
        PROBABILITY_SWAP = 0;

        double[] rewards = new double[allVars.size()];
        for (int i = 0; i < allVars.size(); i++) {
            // Expand selected node
            TreeNode expandedNode = expand(selection).get(0);

            // Rollout and Backpropagation
            double reward = rollout(expandedNode);
            rewards[expandedNode.node] = reward;
            backPropagate(expandedNode, reward);

            String i_str = String.format("0.%0" + String.valueOf(allVars.size()).length() + "d", i);
            saveRound(i_str, 0);
        }
        this.root.setFullyExpanded(true);

        PROBABILITY_SWAP = random_const;

        // 4. Train the normalizer with the mean and sd of the scores of all vars
        normalize_fit(rewards);

        // Convert the BDeus obtained
        root.setTotalReward(normalize_predict(root.getTotalReward()));
        for (TreeNode tn : root.getChildren().keySet()) {
            double reward = normalize_predict(tn.getTotalReward());
            tn.setTotalReward(reward);
        }
    }

    /**
     * Executes one round of the selection, expansion, rollout and backpropagation iterations.
     */
    private void executeRound(){
        //1. Selection and Expansions
        List<TreeNode> selectedNodes = selectNode();
        if(selectedNodes.isEmpty()) {
            convergence = true;
            return;
        }
        // 2. Expand selected node
        List<TreeNode> expandedNodes = expand(selectedNodes);

        //3. Rollout and Backpropagation
        expandedNodes.parallelStream().forEach(expandedNode -> {
            double reward = rollout(expandedNode);
            backPropagate(expandedNode, normalize_predict(reward));
        });
    }

    /**
     * Selects the best nodes to expand. The nodes are selected with the UCT equation.
     * @return List of selected nodes to be expanded
     */
    private List<TreeNode> selectNode(){
        // Creating the arraylist of the selected nodes.
        List<TreeNode> selection = new ArrayList<>();
        for (int i = 0; i < NUM_SELECTION; i++) {

            // Getting the best node
            if (this.selectionSet.isEmpty()) break;
            TreeNode selectNode = Collections.max(this.selectionSet);

            // Checking if the parent of the best node has already been expanded at least once
            if (selectNode.getParent() == null || selectNode.getParent().isExpanded()) {
                this.selectionSet.remove(selectNode);
                selection.add(selectNode);
            }
            else {
                // Parent has not been fully expanded, adding it for expansion
                this.selectionSet.remove(selectNode.getParent());
                selection.add(selectNode.getParent());
            }
        }
        return selection;
    }

    /**
     * Expands the nodes in the list of selected nodes. The nodes are expanded by creating n child for each selected node.
     * @param selection List of selected nodes
     * @return List of expanded nodes.
     */
    public List<TreeNode> expand(List<TreeNode> selection){
        List<TreeNode> expansion = new ArrayList<>();

        for (TreeNode node : selection) {
            int nExpansion = 0;

            //1. Get all possible actions
            List<Integer> actions = node.getPossibleChilds(getRandomOrder());
            //2. Get actions already taken for this node
            Set<Integer> childrenActions = node.getChildrenIDs();
            for (Integer action: actions) {
                // Checking if the number of expansion for this node is greater than the limit
                if(nExpansion >= NUM_EXPAND)
                    break;

                //3. Check if the actions has already been taken
                if (!childrenActions.contains(action)){
                    // 4. Expand the tree by creating a new node and connecting it to the tree.
                    TreeNode newNode = new TreeNode(action, node);

                    // 5. Check if there are more actions to be expanded in this node, and if not, change the isFullyExpanded value
                    if(node.getChildrenIDs().size() == actions.size())
                        node.setFullyExpanded(true);

                    // 7. Adding the expanded node to the list and queue
                    expansion.add(newNode);
                    selectionSet.add(newNode);
                    nExpansion++;
                }
            }
        }
        return expansion;
    }

    synchronized private void checkBestScore(double score, List<Integer> order) {
        if(score > bestScore){
            bestScore = score;
            bestOrder = order;
            bestDag = hc.getGraph();
        }
    }

    /**
     * Random policy rollout. Given a state with a partial order, we generate a random order that starts off with the initial order
     * @param node TreeNode that provides the partial order for a final randomly generated order.
     * @return Score obtained from the rollout
     */
    public double rollout(TreeNode node){
        // Generating candidates and shuffling
        double scoreSum = 0;

        for (int i = 0; i < NUM_ROLLOUTS; i++) {
            // Pseudorandom order by PGES
            List<Integer> candidates = getRandomOrder();

            // Creating order for HC
            LinkedHashSet<Integer> filteredOrder = new LinkedHashSet<>(node.getOrder());
            filteredOrder.addAll(candidates);
            ArrayList<Integer> finalOrder = new ArrayList<>(filteredOrder);

            for (int j = 0; j < NUMBER_SWAPS * Math.sqrt(finalOrder.size()); j++) {
                if (PROBABILITY_SWAP > 0 && random.nextDouble() <= PROBABILITY_SWAP) {
                    // Randomly swap two elements in the order
                    int index1 = random.nextInt(finalOrder.size());
                    int index2 = random.nextInt(finalOrder.size());

                    // Swapping
                    Collections.swap(finalOrder, index1, index2);
                }
            }

            hc.setOrder(finalOrder);
            
            double score = hc.search();
            scoreSum+= score;
            
            // Updating best score, order and graph
            checkBestScore(score, finalOrder);
        }
        scoreSum = scoreSum / NUM_ROLLOUTS;

        return scoreSum;
    }

    /**
     * Backpropagates the reward and visits of the nodes where a rollout has been done.
     * @param node Node that has been expanded
     * @param reward Reward obtained from the rollout
     */
    synchronized public void backPropagate(TreeNode node, double reward){
        // Add one visit and total reward to the currentNode
        node.backPropagate(reward);

        TreeNode currentNode = node;
        while (currentNode != null){
            if(!currentNode.isFullyExpanded()) {
                selectionSet.add(currentNode);
            }

            // Update currentNode to its parent.
            currentNode = currentNode.getParent();
        }
    }
    
    private ArrayList<Integer> getRandomOrder() {
        return new ArrayList(orderSet.get(random.nextInt(orderSet.size())));
    }

    private void initializeWithHC() {
        double init = System.currentTimeMillis();
        HillClimbingEvaluator hc = new HillClimbingEvaluator(problem, cache);
        Dag dag = new Dag(hc.searchUnrestricted());
        initialize(dag, init);
    }

    private void initializeWithGES() {
        double init = System.currentTimeMillis();
        GES_BNBuilder alg = new GES_BNBuilder(problem.getData(), true);
        Dag dag = new Dag(alg.search());
        initialize(dag, init);
    }
    
    private void initializeWithPGES(int nThreads) {        
        // Execute PGES to obtain a good order
        double init = System.currentTimeMillis();
        Clustering hierarchicalClustering = new HierarchicalClustering();
        BNBuilder algorithm = new PGESwithStages(problem, hierarchicalClustering, nThreads, Integer.MAX_VALUE, Integer.MAX_VALUE, false);
        algorithm.search();

        // Create the set with some orders to use in rollout
        Dag currentDag = algorithm.getCurrentDag();

        initialize(currentDag, init);
    }

    private void initializeWithfGES() {
        double init = System.currentTimeMillis();
        Fges alg = new Fges(problem.getBDeu());
        Dag dag = new Dag(Utils.removeInconsistencies(alg.search()));
        initialize(dag, init);
    }

    // TODO: Revisar tests estadísticos
    private void initializeWithPC() {
        double init = System.currentTimeMillis();
        IndependenceTest bdeu_test = new IndTestChiSquare(problem.getData(),0.05);
        Pc alg = new Pc(bdeu_test);
        Dag dag = new Dag(Utils.removeInconsistencies(alg.search()));
        initialize(dag, init);
    }

    // TODO: Revisar tests estadísticos
    private void initializeWithCPC() {
        double init = System.currentTimeMillis();
        IndependenceTest bdeu_test = new IndTestChiSquare(problem.getData(),0.05);
        Cpc alg = new Cpc(bdeu_test);
        Dag dag = new Dag(Utils.removeInconsistencies(alg.search()));
        initialize(dag, init);
    }

    private void initialize(Dag currentDag, double init) {
        for (int i = 0; i < 10000; i++) {
            orderSet.add(hc.nodeToIntegerList(Utils.getTopologicalOrder(currentDag)));
        }

        System.out.println("\nFINISHED " + this.initializeAlgorithm + " (" + ((System.currentTimeMillis() - init)/1000.0) + " s)");
        // Score with fGES
        //System.out.println("BDeu: " + calculateBDeu(new BN_DataSet(problem.getData(),"data"), currentDag));

        this.PGESTime = (System.currentTimeMillis() - init)/1000.0;
        this.initializeDag = currentDag;
    }
    
    /**
     * Normalize (standardize) the sample, so it is has a mean of 0 and a standard deviation of 1.
     *
     * @param sample Sample to normalize.
     * @since 2.2
     */
    private void normalize_fit(double[] sample) {
        DescriptiveStatistics stats = new DescriptiveStatistics();

        // Add the data from the series to stats
        for (double v : sample) {
            stats.addValue(v / this.problem.getData().getNumRows());
        }

        // Compute mean and standard deviation
        mean = stats.getMean();
        standardDeviation = stats.getStandardDeviation();
    }
    
    private double normalize_predict(double sample) {
        sample = sample/this.problem.getData().getNumRows();
        return (sample- mean) / standardDeviation;
    }

    public void setInitialTree(TreeNode root) {
        this.root = root;
    }
    
    private void saveRound(String iteration, double totalTimeRound) {
        try {
            String result = (firstPart
                    + bestScore + ","
                    + iteration + ","
                    + totalTimeRound + "\n");
            csvWriter.append(result);
            csvWriter.flush();
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void initializeWriter() {
        String PATH = ExperimentMCTSLauncher.PATH;
        // Creating the folder if not exists
        File directory = new File(PATH + "results-it");
        if (!directory.exists()){
            directory.mkdir();
        }

        String savePath = PATH + "results-it/experiment_mcts-" + initializeAlgorithm + "_" +
                this.problem.getData().getName() + "_it" + this.ITERATION_LIMIT + "_ex" + this.EXPLOITATION_CONSTANT
                + "_ps" + this.NUMBER_SWAPS + "_ns" + this.PROBABILITY_SWAP + ".csv";
        file = new File(savePath);
        try {
            csvWriter = new BufferedWriter(new FileWriter(savePath, true));
            if (file.length() == 0) {
                String header = "algorithm,bbdd,itLimit,exploitConst,numSwaps,probSwap,bdeuMCTS,iteration,time\n";
                csvWriter.append(header);
            }
            csvWriter.flush();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        this.firstPart = "mcts-" + initializeAlgorithm + "," + this.problem.getData().getName() + "," +
                this.ITERATION_LIMIT  + "," + this.EXPLOITATION_CONSTANT + "," + this.NUMBER_SWAPS + "," + this.PROBABILITY_SWAP + ",";
    }

    public List<Integer> getBestOrder(){
        return bestOrder;
    }


    public Graph getBestDag() {
        return bestDag;
    }

    public void setBestDag(Graph dag, double score) {
        this.bestDag = dag;
        this.bestScore = score;
    }

    public Dag getInitializeDag() {
        return initializeDag;
    }

    @Override
    public String toString() {
        return root.toString();
    }

    public TreeNode getTreeRoot() {
        return root;
    }

    public void setIterationLimit(int limit) {
        this.ITERATION_LIMIT = limit;
    }
}
