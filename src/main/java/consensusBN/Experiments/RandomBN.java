package consensusBN.Experiments;

import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.bayes.*;
import edu.cmu.tetrad.data.*;

public class RandomBN {
	int seed;
	Random generator;
	int numNodes;
	int maxInDegree;
	int maxOutDegree;
	int maxDegree;
	int maxEdges;
	int numBNs;
	int numIterations;
	int[][] parentMatrix;
	int[][] childMatrix;
	int[][] initialParentMatrix;
	int[][] initialChildMatrix;
	public ArrayList<Dag> setOfRandomDags;
	ArrayList<Node> nodesDags;
	/**
     * Parent of random edge. 0 is the default parent node.
     */
    private int randomParent = 0;

    /**
     * Child of random edge. 0 is the default child node.
     */
    private int randomChild = 1;
	public ArrayList<BayesIm> setOfRandomBNs;
	DataSet dataSamples;
	private boolean simulate = false;
	private boolean initialDag = false;
	private int sampleSize = 100;


	/** Uses the parameters of paper "Efficient and accurate structural fusion of Bayesian networks"*/
	public RandomBN(int seed, int numNodes, int numBNs){
		this.seed = seed;
		this.generator = new Random(seed);
		this.numNodes = numNodes;

		// Maximum number of parents of each node
		this.maxInDegree = 3;
		// Maximum number of children of each node
		this.maxOutDegree = 4;

		this.maxDegree = numNodes - 1;
		this.maxEdges = (int) (numNodes * 2.5);
		this.numBNs = numBNs;
		this.numIterations = (int) (numNodes * 0.75);

		this.parentMatrix = null;
		this.childMatrix = null;
		this.initialParentMatrix = null;
		this.initialChildMatrix = null;

		this.setOfRandomDags = new ArrayList<Dag>();
		this.setOfRandomBNs = new ArrayList<BayesIm>();

		this.nodesDags = new ArrayList<Node>();
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(0);

		int numDigits = (int) Math.ceil(Math.log(this.numNodes) / Math.log(10.0));
		nf.setMinimumIntegerDigits(numDigits);
		nf.setGroupingUsed(false);

		for (int i = 1; i <= this.numNodes; i++) {
			GraphNode node = new GraphNode("X" + nf.format(i));
			this.nodesDags.add(node);
		}
	}

	/** Uses a real initial DAG */
	public RandomBN(BayesIm bayesIm, int seed, int numBNs){
		Graph dag = bayesIm.getDag();
		this.initialDag = true;

		this.seed = seed;
		this.generator = new Random(seed);
		this.numNodes = dag.getNumNodes();

		int maxInDegree = 0;
		int maxOutDegree = 0;
		for (Node node : dag.getNodes()) {
			if (dag.getIndegree(node) > maxInDegree) {
				maxInDegree = dag.getIndegree(node);
			}
			if (dag.getOutdegree(node) > maxOutDegree) {
				maxOutDegree = dag.getOutdegree(node);
			}
		}

		// Maximum number of parents of each node
		this.maxInDegree = maxInDegree + 1;
		// Maximum number of children of each node
		this.maxOutDegree = maxOutDegree + 1;

		this.maxDegree = numNodes - 1;
		this.maxEdges = (int) (dag.getNumEdges() * 1.1);
		this.numBNs = numBNs;
		this.numIterations = (int) (numNodes * 0.75);

		this.parentMatrix = null;
		this.childMatrix = null;
		this.initialParentMatrix = null;
		this.initialChildMatrix = null;

		this.setOfRandomDags = new ArrayList<Dag>();
		this.setOfRandomBNs = new ArrayList<BayesIm>();

		this.nodesDags = new ArrayList<>(dag.getNodes());
	}

	public RandomBN(int seed, int numNodes, int numEdges, int numBNs, int numIterations){
		this.seed = seed;
		this.generator = new Random(seed);
		this.numNodes = numNodes;
		this.maxInDegree = numNodes-1;
		this.maxOutDegree = numNodes-1;
		this.maxDegree = numNodes-1;
		this.maxEdges = numEdges;
		this.numBNs = numBNs;
		this.numIterations = numIterations;
		
		this.parentMatrix = null;
		this.childMatrix = null;
		this.initialParentMatrix = null;
		this.initialChildMatrix = null;
		
		this.setOfRandomDags = new ArrayList<Dag>();
		this.setOfRandomBNs = new ArrayList<BayesIm>();
		
		this.nodesDags = new ArrayList<Node>();
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(0);

		int numDigits = (int) Math.ceil(Math.log(this.numNodes) / Math.log(10.0));
		nf.setMinimumIntegerDigits(numDigits);
		nf.setGroupingUsed(false);
		
		for (int i = 1; i <= this.numNodes; i++) {
			GraphNode node = new GraphNode("X" + nf.format(i));
			this.nodesDags.add(node);
		}
	}
	
	public RandomBN(int seed, int numNodes, int numEdges, int numBNs, int numIterations, int sampleSize){
		this.seed = seed;
		this.generator = new Random(seed);
		this.numNodes = numNodes;
		this.maxInDegree = numNodes-1;
		this.maxOutDegree = numNodes-1;
		this.maxDegree = numNodes-1;
		this.maxEdges = numEdges;
		this.numBNs = numBNs;
		this.numIterations = numIterations;
		this.simulate = true;
	
		this.parentMatrix = null;
		this.childMatrix = null;
		this.initialParentMatrix = null;
		this.initialChildMatrix = null;
		
		this.setOfRandomDags = new ArrayList<Dag>();
		this.setOfRandomBNs = new ArrayList<BayesIm>();
		this.dataSamples = null;
		this.sampleSize = sampleSize;
		
		
		this.nodesDags = new ArrayList<Node>();
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(0);

		int numDigits = (int) Math.ceil(Math.log(this.numNodes) / Math.log(10.0));
		nf.setMinimumIntegerDigits(numDigits);
		nf.setGroupingUsed(false);

		for (int i = 1; i <= this.numNodes; i++) {
			GraphNode node = new GraphNode("X" + nf.format(i));
			this.nodesDags.add(node);
		}
	}

	public int getNumNodes() {
		return numNodes;
	}

	/**
	 * Sets the number of nodes and resets all the other parameters to
	 * default values accordingly.
	 *
	 * @param numNodes Must be an integer >= 4.
	 */
	public void setNumNodes(int numNodes) {
		if (numNodes < 1) {
			throw new IllegalArgumentException("Number of nodes must be >= 1.");
		}

		this.numNodes = numNodes;
		this.maxDegree = numNodes - 1;
		this.maxInDegree = numNodes - 1;
		this.maxOutDegree = numNodes - 1;
		this.maxEdges = numNodes - 1;
		this.numIterations = 6 * numNodes * numNodes;

		if (this.numIterations > 300000000) {
			this.numIterations = 300000000;
		}

		this.parentMatrix = null;
		this.childMatrix = null;
	}

	public int getNumIterations() {
		return numIterations;
	}

	public void setNumIterations(int numIterations) {
		this.numIterations = numIterations;
	}

	public void generate(){
		if(!this.initialDag) generateInitialDag();
		generateDags();
		generateProps();
		if(this.simulate) sampling();
	}

	private void sampling(){
		this.dataSamples = this.setOfRandomBNs.get(0).simulateData(this.sampleSize, true);
	}

	private void generateProps() {
		BayesPm bayesPm = new BayesPm(this.setOfRandomDags.get(0), 2, 2);
		MlBayesIm bn = new MlBayesIm(bayesPm,MlBayesIm.RANDOM);
		this.setOfRandomBNs.add(bn);
		for(int i = 1; i < this.numBNs; i++){
			bayesPm = new BayesPm(this.setOfRandomDags.get(i), 2, 2);
			this.setOfRandomBNs.add(new MlBayesIm(bayesPm));
		}
	}

	public int getMaxDegree() {
		return maxDegree;
	}

	/**
	 * Sets the maximum degree of any nodes in the graph.
	 *
	 * @param maxDegree An integer between 3 and numNodes - 1, inclusively.
	 */
	public void setMaxDegree(int maxDegree) {
		if (maxDegree < 3) {
			throw new IllegalArgumentException("Degree of nodes must be >= 3.");
		}
		this.maxDegree = maxDegree;
	}

	public int getMaxInDegree() {
		return maxInDegree;
	}

	public void setMaxInDegree(int maxInDegree) {
		 if (maxInDegree < 2) {
			throw new IllegalArgumentException("Max indegree must be >= 2 " +
					"when generating DAGs under the assumption of " +
					"connectedness.");
		}
		this.maxInDegree = maxInDegree;
	}

	public int getMaxOutDegree() {
		return maxOutDegree;
	}

	public void setMaxOutDegree(int maxOutDegree) {
		if (getMaxInDegree() < 2) {
			throw new IllegalArgumentException("Max indegree must be >= 2 " +
					"when generating DAGs under the assumption of " +
					"connectedness.");
		}
		this.maxOutDegree = maxOutDegree;
	}

	private int getMaxEdges() {
		return maxEdges;
	}

	public int getMaxPossibleEdges() {
		return getNumNodes() * (getMaxDegree()-1) / 2;
	}

	public void setMaxEdges(int maxEdges) {
		if (maxEdges < 0) {
			throw new IllegalArgumentException("Max edges must be >= 0.");
		}

		if (maxEdges > getMaxPossibleEdges()) {
			maxEdges = getMaxPossibleEdges();
		}

		this.maxEdges = maxEdges;
	}

	public Dag getDag() {
		return getDag(this.nodesDags);
	}
	
	public Dag getDag(List<Node> nodes) {
		if (nodes.size() != this.numNodes) {
			throw new IllegalArgumentException("Only " + nodes.size() + " nodes were provided, but the " +
					"simulated graph has " + this.numNodes + ".");
		}

		Dag dag = new Dag();
		for (Node node : nodes) {
			dag.addNode(node);
		}

		for (int i = 0; i < this.numNodes; i++) {
			Node child = nodes.get(i);

			if (parentMatrix[i][0] != 1) {
				for (int j = 1; j < parentMatrix[i][0]; j++) {
					Node parent = nodes.get(parentMatrix[i][j]);
					dag.addDirectedEdge(parent, child);
				}
			}
		}

		GraphUtils.circleLayout(dag, 200, 200, 150);
		return dag;
	}

	//================================PRIVATE METHODS======================//

	/**
	 * This is the algorithm in Melancon and Philippe, "Generating connected
	 * acyclic digraphs uniformly at random" (draft of March 25, 2004). In
	 * addition to acyclicity, some other conditions have been added in.
	 */
	private void generateInitialDag(){
		initialDag();
		if (getNumNodes() <= 1) {
			return;
		}

		int totalEdges = sampleEdgesWhileDisconnecting();

		for (int i = 0; i < getNumNodes()*getNumIterations(); i++) {
			totalEdges = tryAddEdgeSampled(totalEdges);
		}

		this.initialParentMatrix = new int[getNumNodes()][getMaxDegree() + 2];
		this.initialChildMatrix = new int[getNumNodes()][getMaxDegree() + 2];

		for(int j=0; j< getNumNodes(); j++)
			for(int i = 0; i< (getMaxDegree()+2); i++){
				initialParentMatrix[j][i]=parentMatrix[j][i];
			}

		for(int j=0; j< getNumNodes(); j++)
			for(int i = 0; i< (getMaxDegree()+2); i++){
				initialChildMatrix[j][i]=childMatrix[j][i];
			}

		this.setOfRandomDags.add(getDag());
	}

	private void generateDags() {
		for (int i = 0 ; i< this.numBNs-1; i++){
			initialize();

			if (getNumNodes() <= 1) {
				return;
			}

			int totalEdges = sampleEdgesWhileDisconnecting();

			for (int i1 = 0; i1 < getNumIterations(); i1++) {
				totalEdges = tryAddEdgeSampled(totalEdges);
			}
			this.setOfRandomDags.add(getDag());
		}
	}

	private int sampleEdgesWhileDisconnecting() {
		int totalEdges = getNumNodes() - 1;
		while (isDisconnecting()) {
			sampleEdge();
			if (edgeExists()) {
				continue;
			}

			if (isAcyclic() && maxDegreeNotExceeded()) {
				addEdge();
				totalEdges++;
			}
		}
		return totalEdges;
	}

	private int tryAddEdgeSampled(int totalEdges) {
		sampleEdge();
		if (edgeExists()) {
			if (isDisconnecting()) {
				removeEdge();
				reverseDirection();

				if (totalEdges < getMaxEdges() && maxDegreeNotExceeded() &&
						maxIndegreeNotExceeded() &&
						maxOutdegreeNotExceeded() && isAcyclic()) {
					addEdge();
				}
				else {
					reverseDirection();
					addEdge();
				}
			}
			else {
				removeEdge();
				totalEdges--;
			}
		}
		else {
			if (totalEdges < getMaxEdges() && maxDegreeNotExceeded() &&
					maxIndegreeNotExceeded() && maxOutdegreeNotExceeded() &&
					isAcyclic()) {
				addEdge();
				totalEdges++;
			}
		}
		return totalEdges;
	}

	private void reverseDirection() {
		int temp = randomChild;
		randomChild = randomParent;
		randomParent = temp;
	}

	/**
	 * Returns true if the edge parent-->child exists in the graph.
	 */
	private boolean edgeExists() {
		for (int i = 1; i < parentMatrix[randomChild][0]; i++) {
			if (parentMatrix[randomChild][i] == randomParent) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Returns true if the degree of the current nodes randomParent and
	 * randomChild do not exceed maxDegree.
	 */
	private boolean maxDegreeNotExceeded() {
		int parentDegree = parentMatrix[randomParent][0] +
				childMatrix[randomParent][0] - 1;
		int childDegree =
				parentMatrix[randomChild][0] + childMatrix[randomChild][0] - 1;
		return parentDegree <= getMaxDegree() && childDegree <= getMaxDegree();
	}

	/**
	 * Returns true if the degrees of the current nodes randomParent and
	 * randomChild do not exceed maxIndegree.
	 */
	private boolean maxIndegreeNotExceeded() {
		return parentMatrix[randomChild][0] <= getMaxInDegree();
	}

	/**
	 * Returns true if the degrees of the current nodes randomParent and
	 * randomChild do not exceed maxOutdegree.
	 */
	private boolean maxOutdegreeNotExceeded() {
		return childMatrix[randomParent][0] <= getMaxOutDegree();
	}

	/**
	 * Returns true iff the random edge randomParent-->randomChild would be
	 * disconnecting were it to be removed.
	 */
	private boolean isDisconnecting() {
		boolean visited[] = new boolean[getNumNodes()];
		int list[] = new int[getNumNodes()];
		int index = 0;
		int lastIndex = 1;
		list[0] = 0;
		visited[0] = true;
		while (index < lastIndex) {
			int currentNode = list[index];

			// verify parents of current node
			for (int i = 1; i < parentMatrix[currentNode][0]; i++) {
				if (currentNode == randomChild &&
						parentMatrix[currentNode][i] == randomParent) {
					continue;
				}

				if (!visited[parentMatrix[currentNode][i]]) {
					list[lastIndex] = parentMatrix[currentNode][i];
					visited[parentMatrix[currentNode][i]] = true;
					lastIndex++;
				}
			}

			// verify children of current node
			for (int i = 1; i < childMatrix[currentNode][0]; i++) {
				if (currentNode == randomParent &&
						childMatrix[currentNode][i] == randomChild) {
					continue;
				}

				if (!visited[childMatrix[currentNode][i]]) {
					list[lastIndex] = childMatrix[currentNode][i];
					visited[childMatrix[currentNode][i]] = true;
					lastIndex++;
				}
			}

			index++;
		}

		// verify whether all nodes were visited
		for (boolean aVisited : visited) {
			if (!aVisited) {
				return true;
			}
		}

		return false;
	}


	/**
	 * Returns true if the graph is still acyclic after the last edge was added.
	 * This method only works before adding the random edge, not after removing
	 * an edge.
	 */
	private boolean isAcyclic() {
		boolean[] visited = new boolean[getNumNodes()];
		boolean noCycle = true;
		int[] list = new int[getNumNodes() + 1];
		int index = 0;
		int lastIndex = 1;
		list[0] = randomParent;
		visited[randomParent] = true;
		while (index < lastIndex && noCycle) {
			int currentNode = list[index];
			int i = 1;

			// verify parents of current node
			while ((i < parentMatrix[currentNode][0]) && noCycle) {
				if (!visited[parentMatrix[currentNode][i]]) {
					if (parentMatrix[currentNode][i] != randomChild) {
						list[lastIndex] = parentMatrix[currentNode][i];
						lastIndex++;
					}
					else {
						noCycle = false;
					}
					visited[parentMatrix[currentNode][i]] = true;
				}
				i++;
			}
			index++;
		}
		//System.out.println("\tnoCycle:"+noCycle);
		return noCycle;
	}

	/**
	 * Initializes the parentMatrix and childMatrix to the initial values.
	 */
	private void initialize() {
		parentMatrix = new int[getNumNodes()][getMaxDegree() + 2];
		childMatrix = new int[getNumNodes()][getMaxDegree() + 2];
		
		for(int j=0; j< getNumNodes(); j++)
			for(int i = 0; i < (getMaxDegree()+2); i++){
				parentMatrix[j][i] = initialParentMatrix[j][i];
			}
		
		for(int j=0; j< getNumNodes(); j++)
			for(int i = 0; i < (getMaxDegree()+2); i++){
				childMatrix[j][i] = initialChildMatrix[j][i];
			}
	}
	
	/**
	 * Initializes the graph as a simple ordered tree, 0-->1-->2-->...-->n.
	 */
	private void initialDag(){
		parentMatrix = new int[getNumNodes()][getMaxDegree() + 2];
		childMatrix = new int[getNumNodes()][getMaxDegree() + 2];

		for (int i = 0; i < getNumNodes(); i++) {
			for (int j = 1; j < getMaxDegree() + 1; j++) {
				parentMatrix[i][j] = -5; //set first node
				childMatrix[i][j] = -5;
			}
		}
		parentMatrix[0][0] = 1; //set first node
		childMatrix[0][0] = 2;    //set first node
		childMatrix[0][1] = 1;    //set first node
		parentMatrix[getNumNodes() - 1][0] = 2;  //set last node
		parentMatrix[getNumNodes() - 1][1] = getNumNodes() - 2;  //set last node
		childMatrix[getNumNodes() - 1][0] = 1;     //set last node
		for (int i = 1; i < (getNumNodes() - 1); i++) {  // set the other nodes
			parentMatrix[i][0] = 2;
			parentMatrix[i][1] = i - 1;
			childMatrix[i][0] = 2;
			childMatrix[i][1] = i + 1;
		}
	}

	/**
	 * Sets randomParent-->randomChild to a random edge, chosen uniformly.
	 */
	private void sampleEdge() {
			int rand = this.generator.nextInt(getNumNodes() * (getNumNodes() - 1));
			randomParent = rand / (getNumNodes() - 1);
			int rest = rand - randomParent * (getNumNodes() - 1);
			if (rest >= randomParent) {
				randomChild = rest + 1;
			}
			else{
				randomChild = rest;
			}
	}

	/**
	 * Adds the edge randomParent-->randomChild to the graph.
	 */
	private void addEdge() {
		childMatrix[randomParent][childMatrix[randomParent][0]] = randomChild;
		childMatrix[randomParent][0]++;
		parentMatrix[randomChild][parentMatrix[randomChild][0]] = randomParent;
		parentMatrix[randomChild][0]++;
	}

	/**
	 * Removes the edge randomParent-->randomChild from the graph.
	 */
	private void removeEdge() {
		boolean go = true;
		int lastNode;
		int proxNode;
		int atualNode;
		if ((parentMatrix[randomChild][0] != 1) &&
				(childMatrix[randomParent][0] != 1)) {
			lastNode =
					parentMatrix[randomChild][parentMatrix[randomChild][0] - 1];
			for (int i = (parentMatrix[randomChild][0] - 1); (i > 0 && go); i--) {
				// remove element from parentMatrix
				atualNode = parentMatrix[randomChild][i];
				if (atualNode != randomParent) {
					proxNode = atualNode;
					parentMatrix[randomChild][i] = lastNode;
					lastNode = proxNode;
				}
				else {
					parentMatrix[randomChild][i] = lastNode;
					go = false;
				}
			}
			if ((childMatrix[randomParent][0] != 1) &&
					(childMatrix[randomParent][0] != 1)) {
				lastNode = childMatrix[randomParent][childMatrix[randomParent][0] - 1];
				go = true;
				for (int i = (childMatrix[randomParent][0] - 1); (i > 0 &&
						go); i--) { // remove element from childMatrix
					atualNode = childMatrix[randomParent][i];
					if (atualNode != randomChild) {
						proxNode = atualNode;
						childMatrix[randomParent][i] = lastNode;
						lastNode = proxNode;
					}
					else {
						childMatrix[randomParent][i] = lastNode;
						go = false;
					}
				} // end of for
			}
			childMatrix[randomParent][(childMatrix[randomParent][0] - 1)] = -4;
			childMatrix[randomParent][0]--;
			parentMatrix[randomChild][(parentMatrix[randomChild][0] - 1)] = -4;
			parentMatrix[randomChild][0]--;
		}
	}

	public static void main(String[] args) {
		RandomBN randomBN = new RandomBN(1, 10, 5);
		randomBN.generate();

		System.out.println(randomBN.setOfRandomDags);
	}
}
