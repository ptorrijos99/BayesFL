package consensusBN.Method;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;

import java.util.*;

import static consensusBN.BetaToAlpha.transformToAlpha;
import static consensusBN.ConsensusUnion.*;

public class InitialDAGsWoRepeat_Method implements Population {

    private Random random;
    private int maxTreewidth;

    private List<Node> alpha;
    private ArrayList<Dag> originalDags;
    private ArrayList<Edge> edges;
    private int totalEdges;
    private HashMap<Edge, Integer> edgeFrequency;
    private ArrayList<Integer> edgeFrequencyArray;

    private Dag greedyDag;
    private List<Dag> greedyDags;
    private double executionTimeGreedy;

    @Override
    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random) {
        this.random = random;
        this.alpha = alpha;
        this.maxTreewidth = maxTreewidth;
        this.originalDags = new ArrayList<>(dags);

        // Order the edges of all the DAGs by the number of times they appear
        edgeFrequency = new HashMap<>();
        edgeFrequencyArray = new ArrayList<>();
        for (Dag d : dags) {
            for (Edge edge : d.getEdges()) {
                // Add the edge to the map. If it already exists, increase its frequency
                Integer added = edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
                // If the edge is new, add it to the list of edges and the map with its position
                if (added == null) {
                    edgeFrequencyArray.add(1);
                }
            }
        }
        edges = new ArrayList<>(edgeFrequency.keySet());

        double startTime = System.currentTimeMillis();
        greedyDags = originalDAGsGreedyTreewidthBeforeWoRepeat(dags, alpha, ""+maxTreewidth);
        greedyDag = fusionUnion(greedyDags);
        executionTimeGreedy = (System.currentTimeMillis() - startTime) / 1000;

        totalEdges = edgeFrequency.size();
    }

    /**
     * Initialize the population. The initial population is composed of the edges of the original DAGs.
     * The edges of the DAGs are not repeated in the population.
     */
    @Override
    public boolean[][] initializePopulation(int populationSize) {
        int maxFreq = Collections.max(edgeFrequency.values());
        int minFreq = Collections.min(edgeFrequency.values()) - 1;

        // Initialize the population
        boolean[][] population = new boolean[populationSize][totalEdges];
        boolean uniform = minFreq == maxFreq;

        // Add the greedy solution to the population
        for (int i = 0; i < totalEdges; i++) {
            for (Dag greedy : greedyDags) {
                population[0][i] = greedy.containsEdge(edges.get(i)) || population[0][i];
            }
        }

        // Add the greedy solutions with maxTreewidth-1 to the population
        List<Dag> greedy = originalDAGsGreedyTreewidthBeforeWoRepeat(originalDags, alpha, ""+(maxTreewidth-1));
        for (int i = 0; i < totalEdges; i++) {
            for (Dag g : greedy) {
                population[1][i] = g.containsEdge(edges.get(i)) || population[1][i];
            }
        }

        // Initialize the rest of the population with random individuals based on the frequency of the edges
        for (int i = 2; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                if (uniform) {
                    population[i][j] = random.nextBoolean();
                } else {
                    double normalized = (double) (edgeFrequencyArray.get(j) - minFreq) / (maxFreq - minFreq);
                    population[i][j] = random.nextDouble() < 1/(1-Math.log(normalized));
                }
            }
        }

        return population;
    }

    @Override
    public Dag getUnionFromChromosome(boolean[] chromosome) {
        // Create the DAG that corresponds to each individual
        ArrayList<Dag> candidates = fromChromosomeToDags(chromosome);

        ArrayList<Dag> outputDags = new ArrayList<>();
        for (Dag dag : candidates) {
            outputDags.add(transformToAlpha(dag, alpha));
        }

        return applyUnion(alpha, outputDags);
    }

    private ArrayList<Dag> fromChromosomeToDags(boolean[] chromosome) {
        ArrayList<Dag> dags = new ArrayList<>();
        int i = 0;
        for (Dag originalDag : originalDags) {
            Dag dag = new Dag(alpha);
            for (int j = 0; j < totalEdges; j++) {
                if (chromosome[j] && originalDag.containsEdge(edges.get(j))) {
                    dag.addEdge(edges.get(j));
                }
            }
            dags.add(dag);
        }

        return dags;
    }

    @Override
    public Dag getGreedyDag() {
        return greedyDag;
    }

    @Override
    public double getExecutionTimeGreedy() {
        return executionTimeGreedy;
    }
}
