package consensusBN.Method;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;

import java.util.*;

import static consensusBN.ConsensusUnion.applyGreedyMaxTreewidth;

public class Fusion_Method implements Population {

    private Random random;
    private int maxTreewidth;

    private List<Node> alpha;
    private ArrayList<Edge> edgesAlpha;
    private int totalEdges;
    private HashMap<Edge, Integer> edgeFrequency;
    private ArrayList<Integer> edgeFrequencyArray;

    private Dag greedyDag;
    private double executionTimeGreedy;

    public void initialize(List<Dag> dags, List<Node> alpha, List<Dag> alphaDags, int maxTreewidth, Random random) {
        this.random = random;
        this.alpha = alpha;
        this.maxTreewidth = maxTreewidth;

        // Order the edges of all the DAGs by the number of times they appear
        edgeFrequency = new HashMap<>();
        edgesAlpha = new ArrayList<>();
        edgeFrequencyArray = new ArrayList<>();
        for (Dag d : alphaDags) {
            for (Edge edge : d.getEdges()) {
                // Add the edge to the map. If it already exists, increase its frequency
                Integer added = edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
                // If the edge is new, add it to the list of edges and the map with its position
                if (added == null) {
                    edgesAlpha.add(edge);
                    edgeFrequencyArray.add(1);
                }
            }
        }

        double startTime = System.currentTimeMillis();
        greedyDag = applyGreedyMaxTreewidth(alpha, edgesAlpha, ""+maxTreewidth);
        executionTimeGreedy = (System.currentTimeMillis() - startTime) / 1000;

        totalEdges = edgeFrequency.size();
    }

    /**
     * Initialize the population. The initial population is composed of the edges of the DAGs in the alpha order.
     * The first individual is the greedy solution, the second is the greedy solution with maxTreewidth-1, and the rest
     * are random individuals based on the frequency of the edges.
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
            population[0][i] = greedyDag.containsEdge(edgesAlpha.get(i));
        }

        // Add the greedy solutions with maxTreewidth-1 to the population
        Dag greedy = applyGreedyMaxTreewidth(alpha, edgesAlpha, ""+(maxTreewidth-1));
        for (int i = 0; i < totalEdges; i++) {
            population[1][i] = greedy.containsEdge(edgesAlpha.get(i));
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
        // Create the DAG that corresponds to the individual
        Dag union = new Dag(alpha);
        for (int i = 0; i < totalEdges; i++) {
            if (chromosome[i]) {
                union.addEdge(edgesAlpha.get(i));
            }
        }
        return union;
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
