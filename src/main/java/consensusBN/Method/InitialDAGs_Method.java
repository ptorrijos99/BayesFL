package consensusBN.Method;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static consensusBN.ConsensusUnion.applyUnion;

public class InitialDAGs_Method implements Population {

    private Random random;

    private List<Edge> edgesOriginal;
    private int nDags;
    private int[] nEdges;
    private List<Node> alpha;
    private int maxTreewidth;

    private Dag greedyDag;
    private double executionTimeGreedy;

    @Override
    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random) {
        this.alpha = alpha;
        this.random = random;
        this.maxTreewidth = maxTreewidth;

        nDags = dags.size();
        nEdges = new int[nDags];
        this.edgesOriginal = new ArrayList<>();
        for (int i = 0; i < nDags; i++) {
            nEdges[i] = dags.get(i).getNumEdges();
            edgesOriginal.addAll(dags.get(i).getEdges());
        }
    }

    /**
     * Initialize the population. The initial population is composed of the edges of the original DAGs.
     */
    @Override
    public boolean[][] initializePopulation(int populationSize) {
        int totalEdges = edgesOriginal.size();
        boolean[][] population = new boolean[populationSize][totalEdges];

        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                population[i][j] = random.nextBoolean();
            }
        }

        return population;
    }

    @Override
    public Dag getUnionFromChromosome(boolean[] chromosome) {
        // Create the DAG that corresponds to each individual
        ArrayList<Dag> candidates = fromChromosomeToDags(chromosome);
        Dag union = applyUnion(alpha, candidates);

        // Check cycles
        if (!union.findCycle().isEmpty()) {
            return null;
        }

        return union;
    }

    private ArrayList<Dag> fromChromosomeToDags(boolean[] individual) {
        ArrayList<Dag> dags = new ArrayList<>();
        int cumulativeEdges = 0;
        for (int i = 0; i < nDags; i++) {
            Dag dag = new Dag(alpha);
            for (int j = 0; j < nEdges[i]; j++) {
                if (individual[cumulativeEdges]) {
                    dag.addEdge(edgesOriginal.get(cumulativeEdges));
                }
                cumulativeEdges++;
            }
            dags.add(dag);
        }

        return dags;
    }

    @Override
    public Dag getGreedyDag() {
        return null;
    }

    @Override
    public double getExecutionTimeGreedy() {
        return executionTimeGreedy;
    }
}
