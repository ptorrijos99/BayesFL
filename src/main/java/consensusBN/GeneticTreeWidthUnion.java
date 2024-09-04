package consensusBN;

import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import org.albacete.simd.utils.Utils;

import java.util.*;
import java.util.stream.IntStream;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;
import static consensusBN.ConsensusUnion.*;

public class GeneticTreeWidthUnion {

    // Parameters
    public int numIterations = 1000;
    public int populationSize = 20;
    public int maxTreewidth = 5;
    public String method = "Gamez";

    // Best values and stats
    private boolean[] bestIndividual;
    public Dag bestDag;
    public double bestFitness = Double.MAX_VALUE;
    public double executionTime;
    public double executionTimeUnion;
    public double executionTimeGreedy;

    // "Final" variables
    private final Random random;
    private int totalEdges;
    private ArrayList<Node> alpha;
    private ArrayList<Dag> alphaDags;
    private ArrayList<Edge> edges;
    private ArrayList<Integer> edgeFrequencyArray;
    private HashMap<Edge, Integer> edgePosition;
    private HashMap<Edge, Integer> edgeFrequency;
    private int maxFreq;
    private int minFreq;
    public Dag greedyDag;
    public Dag fusionUnion;

    // Variables
    private boolean[][] population;
    private double[] fitness;
    private int[] treeWidths;

    public GeneticTreeWidthUnion() {
        this.random = new Random();
    }

    public GeneticTreeWidthUnion(int seed) {
        this.random = new Random(seed);
    }

    public GeneticTreeWidthUnion(int seed, int maxTreewidth) {
        this.maxTreewidth = maxTreewidth;
        this.random = new Random(seed);
    }

    /**
     * Complete union of the DAGs limiting the tree-width of the fused DAG.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public Dag fusionUnion(ArrayList<Dag> dags) {
        double startTime = System.currentTimeMillis();
        initializeVars(dags);

        if (method.equals("Gamez")) {
            fusionUnionGamez();
        } else {
            fusionUnionPuerta(alphaDags);
        }

        executionTime = (System.currentTimeMillis() - startTime) / 1000;

        System.out.println("Best fitness: " + bestFitness + " | Edges: " + bestDag.getNumEdges());
        System.out.println("Greedy SMHD: " + Utils.SMHD(fusionUnion,greedyDag) + " | Greedy edges: " + greedyDag.getNumEdges());
        
        return bestDag;
    }

    public void initializeVars(ArrayList<Dag> dags) {
        // Transform the DAGs to the same alpha order
        alpha = alphaOrder(dags);
        alphaDags = new ArrayList<>();
        for (Dag dag : dags) {
            alphaDags.add(transformToAlpha(dag, alpha));
        }
        System.out.println("Alpha order: " + alpha);
        for (Dag dag : alphaDags) {
            System.out.println(dag);
        }

        double startTime = System.currentTimeMillis();
        fusionUnion = applyUnion(alpha, alphaDags);
        executionTimeUnion = (System.currentTimeMillis() - startTime) / 1000;

        // Order the edges of all the DAGs by the number of times they appear
        int i = 0;
        edgeFrequency = new HashMap<>();
        edgePosition = new HashMap<>();
        edges = new ArrayList<>();
        edgeFrequencyArray = new ArrayList<>();
        for (Dag d : alphaDags) {
            for (Edge edge : d.getEdges()) {
                // Add the edge to the map. If it already exists, increase its frequency
                Integer added = edgeFrequency.put(edge, edgeFrequency.getOrDefault(edge, 0) + 1);
                // If the edge is new, add it to the list of edges and the map with its position
                if (added == null) {
                    edges.add(edge);
                    edgePosition.put(edge, i);
                    edgeFrequencyArray.add(1);
                    i++;
                } else {
                    edgeFrequencyArray.set(edgePosition.get(edge), edgeFrequency.get(edge));
                }
            }
        }
        totalEdges = edgeFrequency.size();
        maxFreq = Collections.max(edgeFrequency.values());
        minFreq = Collections.min(edgeFrequency.values()) - 1;

        treeWidths = new int[populationSize];
        population = new boolean[populationSize][totalEdges];
    }

    private void fusionUnionGamez() {
        // Genetic algorithm
        //System.out.println("Initializing");
        initialize();
        for (int j = 0; j < numIterations; j++) {
            //System.out.println("Iteration " + j);
            evaluate();
            crossover();
            evaluate();
            mutate();
        }
    }

    private void fusionUnionPuerta(ArrayList<Dag> dags) {
        // Genetic algorithm
        initialize();
        for (int j = 0; j < numIterations; j++) {
            evaluate();
            crossover();
            mutate();
        }
    }

    private void initialize() {
        boolean uniform = minFreq == maxFreq;

        // Add the greedy solution to the population
        double startTime = System.currentTimeMillis();
        greedyDag = applyGreedyMaxTreewidth(alpha, edges, String.valueOf(maxTreewidth));
        for (int i = 0; i < totalEdges; i++) {
            population[0][i] = greedyDag.containsEdge(edges.get(i));
        }
        executionTimeGreedy = (System.currentTimeMillis() - startTime) / 1000;

        // Add the greedy solutions with maxTreewidth-1 to the population
        Dag greedy = applyGreedyMaxTreewidth(alpha, edges, String.valueOf(maxTreewidth-1));
        for (int i = 0; i < totalEdges; i++) {
            population[1][i] = greedy.containsEdge(edges.get(i));
        }

        for (int i = 2; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                if (uniform) {
                    population[i][j] = random.nextBoolean();
                } else {
                    double normalized = (double) (edgeFrequencyArray.get(j) - minFreq) / (maxFreq-minFreq);
                    population[i][j] = random.nextDouble() < 1/(1-Math.log(normalized));
                }
            }
        }
    }

    private void evaluate() {
        fitness = new double[populationSize];

        IntStream.range(0, populationSize)
                .parallel()
                .forEach(i -> fitness[i] = calculateFitness(i, 0));

        /*System.out.println("Fitness: ");
        for (int i = 0; i < populationSize; i++) {
            System.out.printf("%.2f ", fitness[i]);
        }
        System.out.println();*/
    }

    private void crossover() {
        boolean[][] newPopulation = new boolean[populationSize][totalEdges];
        // Add the best global individual to the new population
        //System.out.println("Best individual: " + bestFitness + " | Edges: " + bestDag.getNumEdges());
        newPopulation[0] = bestIndividual.clone();

        // Add the best individual of the last iteration to the new population
        int bestIndex = 1;
        double bestFitness = fitness[1];
        for (int i = 2; i < populationSize; i++) {
            if (fitness[i] < bestFitness) {
                bestFitness = fitness[i];
                bestIndex = i;
            }
        }
        //System.out.println("Best individual of the last iteration: " + bestFitness);
        newPopulation[1] = population[bestIndex];

        tournamentCrossover(newPopulation);
        //uniformCrossover(newPopulation);

        for (int i = 0; i < populationSize; i++) {
            double x = (double) treeWidths[i] / maxTreewidth;
            // Number of trues
            int numTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    numTrues++;
                }
            }
            //System.out.println(i + " | FIT " + String.format("%.1f", fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | Edges: " + numTrues);
        }

    }

    /** Uniform crossover with roulette wheel selection */
    private void uniformCrossover(boolean[][] newPopulation) {
        double[] cumulativeProbs = new double[populationSize];
        for (int i = 0; i < populationSize; i++) {
            cumulativeProbs[i] = 1/fitness[i];
        }
        double sum = 0;
        for (int i = 0; i < populationSize; i++) {
            sum += cumulativeProbs[i];
            cumulativeProbs[i] = sum;
        }

        for (int i = 2; i < populationSize; i++) {
            int ind1 = getIndividualByCumulative(cumulativeProbs);
            int ind2 = getIndividualByCumulative(cumulativeProbs);
            for (int j = 0; j < totalEdges; j++) {
                if (random.nextBoolean()) {
                    newPopulation[i][j] = population[ind1][j];
                } else {
                    newPopulation[i][j] = population[ind2][j];
                }
            }
        }
        population = newPopulation;
    }

    private int getIndividualByCumulative(double[] cumulativeProbs) {
        double randomValue = random.nextDouble();
        for (int k = 0; k < populationSize; k++) {
            if (randomValue <= cumulativeProbs[k]) {
                return k;
            }
        }
        return cumulativeProbs.length-1;
    }

    /** Tournament selection and crossover */
    private void tournamentCrossover(boolean[][] newPopulation) {
        for (int i = 2; i < populationSize; i++) {
            int index1 = random.nextInt(populationSize);
            int index2 = random.nextInt(populationSize);
            int index3 = random.nextInt(populationSize);
            int index4 = random.nextInt(populationSize);
            int crossoverPoint = random.nextInt(totalEdges);

            if (fitness[index1] < fitness[index2]) {
                System.arraycopy(population[index1], 0, newPopulation[i], 0, crossoverPoint);
            } else {
                System.arraycopy(population[index2], 0, newPopulation[i], 0, crossoverPoint);
            }
            if (fitness[index3] < fitness[index4]) {
                System.arraycopy(population[index3], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            } else {
                System.arraycopy(population[index4], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            }
        }
        population = newPopulation;
    }

    private void mutate() {
        for (int i = 0; i < populationSize; i++) {
            double x = (double) treeWidths[i] / maxTreewidth;

            // Number of trues
            int numTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    numTrues++;
                }
            }

            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    double probRemove;
                    if (x < 1) {
                        probRemove = Math.pow(x/10, 2);
                    } else {
                        probRemove = (Math.log10(x)/3) + 0.01;
                    }

                    if (random.nextDouble() < probRemove) {
                        population[i][j] = false;
                    }
                } else {
                    double probAdd;
                    if (x < 1) {
                        probAdd = Math.pow((x-1)/2, 2) + 0.01;
                    } else {
                        probAdd = 0.01/(x+0.01);
                    }

                    if (random.nextDouble() < probAdd) {
                        population[i][j] = true;
                    }
                }
            }

            // final number of trues
            int finalNumTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    finalNumTrues++;
                }
            }
            //System.out.println(i + " | FIT " + String.format("%.1f",fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | Edges: " + numTrues + ", " + finalNumTrues);
        }
    }

    private double calculateFitness(int index, int recursive) {
        // Create the DAG that corresponds to the individual
        Dag union = new Dag(alpha);
        for (int i = 0; i < totalEdges; i++) {
            if (population[index][i]) {
                union.addEdge(edges.get(i));
            }
        }

        // Get the cliques
        treeWidths[index] = Utils.getTreeWidth(union);

        double fitness;
        if (treeWidths[index] > maxTreewidth) {
            //population[index] = fixIndividual(population[index], cliques, recursive);
            //return calculateFitness(index, recursive+1);
            fitness = Utils.SMHD(union, fusionUnion) * ((double)  treeWidths[index] / maxTreewidth);
        } else {
            fitness = Utils.SMHD(union, fusionUnion);
        }

        if (fitness < bestFitness && treeWidths[index] <= maxTreewidth) {
            bestIndividual = population[index].clone();
            bestDag = union;
            bestFitness = fitness;
        }

        return fitness;
    }

    private boolean[] fixIndividual(boolean[] individual, Map<Node, Set<Node>> cliques, int recursive) {
        boolean[] fixed = individual.clone();
        for (Node node : cliques.keySet()) {
            Set<Node> clique = cliques.get(node);
            if (clique.size() > maxTreewidth - recursive) {
                int numToRemove = clique.size() - maxTreewidth + recursive;
                for (Node neighbor : clique) {
                    if (numToRemove == 0) {
                        break;
                    }
                    if (neighbor.equals(node)) {
                        continue;
                    }
                    boolean removed = false;

                    Integer position1 = edgePosition.get(new Edge(node, neighbor, Endpoint.TAIL, Endpoint.ARROW));
                    if (position1!=null && fixed[position1]) {
                        fixed[position1] = false;
                        removed = true;
                    }
                    Integer position2 = edgePosition.get(new Edge(neighbor, node, Endpoint.TAIL, Endpoint.ARROW));
                    if (position2!=null && fixed[position2]) {
                        fixed[position2] = false;
                        removed = true;
                    }
                    if (removed) {
                        numToRemove--;
                    }
                    Set<Node> neighborClique = cliques.get(neighbor);
                    if (neighborClique != null) {
                        neighborClique.remove(node);
                    }
                }
            }
        }
        return fixed;
    }
}
