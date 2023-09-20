package org.albacete.simd.utils;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DiscreteVariable;
import edu.cmu.tetrad.graph.EdgeListGraph_n;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.BDeuScore;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class Problem {

    /**
     * Data of the problem
     */
    private DataSet data;

    /**
     * Array of variable names from the data set, in order.
     */
    private String[] varNames;

    /**
     * List of variables in the data set, in order.
     */
    private List<Node> variables;

    /**
     * For discrete data scoring, the structure prior.
     */
    protected double structurePrior;

    /**
     * For discrete data scoring, the sample prior.
     */
    protected double samplePrior;

    /**
     * Cases for each variable of the problem.
     */
    protected int[][] cases;

    /**
     * Number of values a variable can take.
     */
    protected int[] nValues;

    /**
     * Map from variables to their column indices in the data set.
     */
    protected  HashMap<Node, Integer> hashIndices = null;

    /**
     * Caches scores for discrete search.
     */
    protected LocalScoreCacheConcurrent localScoreCache = new LocalScoreCacheConcurrent();

    /**
     * BDeu Score.
     */
    protected BDeuScore bdeu;


    public Problem(DataSet dataSet){

        //Setting dataset
        List<String> _varNames = dataSet.getVariableNames();

        this.data = dataSet;
        this.varNames = _varNames.toArray(new String[0]);
        this.variables = dataSet.getVariables();
        // Starting cases
        cases=new int[dataSet.getNumRows()][dataSet.getNumColumns()];
        for(int i=0;i<dataSet.getNumRows();i++) {
            for (int j = 0; j < dataSet.getNumColumns(); j++) {
                cases[i][j] = dataSet.getInt(i, j);
            }
        }
        // Initializing nValues
        nValues=new int[dataSet.getNumColumns()];
        for(int i=0;i<dataSet.getNumColumns();i++)
            nValues[i]=((DiscreteVariable)dataSet.getVariable(i)).getNumCategories();

        //Initializing SamplePrior
        structurePrior = 0.001;
        samplePrior = 10.0;

        //building index
        Graph graph = new EdgeListGraph_n(new LinkedList<Node>(this.variables));
        buildIndexing(graph);
        
        bdeu = new BDeuScore(data);
    }


    public Problem(String file){
        this(Utils.readData(file));
    }


    /**
     * Builds the indexing structure for the Graph passed as an argument.
     * @param g Graph being indexed.
     */
    private void buildIndexing(Graph g) {
        Graph graph = new EdgeListGraph_n(g);
        this.hashIndices = new HashMap<>();
        for (Node next : graph.getNodes()) {
            for (int i = 0; i < varNames.length; i++) {
                if (varNames[i].equals(next.getName())) {
                    hashIndices.put(next, i);
                    break;
                }
            }
        }
    }

    @Override
    public boolean equals(Object obj) {
        if(this == obj){
            return true;
        }
        if (obj == null){
            return false;
        }
        if (obj instanceof Problem){
            Problem p = (Problem) obj;
            return data.equals(p.getData());
        }
        return false;

    }

    public DataSet getData() {
        return data;
    }

    public double getSamplePrior() {
        return samplePrior;
    }

    public void setSamplePrior(double samplePrior){
        this.samplePrior = samplePrior;
    }


    public double getStructurePrior() {
        return structurePrior;
    }

    public void setStructurePrior(double structurePrior) {
        this.structurePrior = structurePrior;
    }

    public int[] getnValues() {
        return nValues;
    }

    public int[][] getCases() {
        return cases;
    }

    public List<Node> getVariables() {
        return variables;
    }

    public String[] getVarNames() {
        return varNames;
    }

    public HashMap<Node, Integer> getHashIndices() {
        return hashIndices;
    }

    public LocalScoreCacheConcurrent getLocalScoreCache() {
        return localScoreCache;
    }
    
    public BDeuScore getBDeu() {
        return bdeu;
    }

}
