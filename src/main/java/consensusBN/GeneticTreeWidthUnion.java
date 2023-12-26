package consensusBN;

import edu.cmu.tetrad.bayes.GraphTools;
import edu.cmu.tetrad.graph.*;

import java.util.*;

import static consensusBN.AlphaOrder.alphaOrder;
import static consensusBN.BetaToAlpha.transformToAlpha;
import static consensusBN.ConsensusUnion.applyGreedyMaxTreewidth;
import static edu.cmu.tetrad.bayes.GraphTools.moralize;

public class GeneticTreeWidthUnion {

    // Parameters
    public int numIterations = 2000;
    public int populationSize = 50;
    public int maxTreewidth;
    public String method = "Gamez";

    // Variables
    private boolean[] bestIndividual;
    private Dag bestDag;
    private double bestFitness;


    private boolean[][] population;
    private int totalEdges;
    private ArrayList<Node> alpha;
    private ArrayList<Edge> edges;
    private ArrayList<Integer> edgeFrequencyArray;
    private HashMap<Edge, Integer> edgePosition;
    private HashMap<Edge, Integer> edgeFrequency;
    private double[] fitness;
    private int[] treeWidths = new int[populationSize];
    int maxFreq;
    int minFreq;

    Random random;

    public GeneticTreeWidthUnion(int maxTreewidth) {
        this.maxTreewidth = maxTreewidth;
        random = new Random();
    }

    public GeneticTreeWidthUnion(int maxTreewidth, int seed) {
        this.maxTreewidth = maxTreewidth;
        random = new Random(seed);
    }

    /**
     * Complete union of the DAGs limiting the tree-width of the fused DAG.
     * @param dags The DAGs to be fused.
     * @return The union of the DAGs.
     */
    public Dag fusionUnion(ArrayList<Dag> dags) {
        // Transform the DAGs to the same alpha order
        alpha = alphaOrder(dags);
        ArrayList<Dag> alphaDags = new ArrayList<>();
        for (Dag dag : dags) {
            alphaDags.add(transformToAlpha(dag, alpha));
        }

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
        minFreq = Collections.min(edgeFrequency.values());

        if (method.equals("Gamez")) {
            fusionUnionGamez(alphaDags);
        } else {
            fusionUnionPuerta(alphaDags);
        }

        System.out.println("Best fitness: " + bestFitness + " | Edges: " + bestDag.getNumEdges());

        return bestDag;
    }


    private void fusionUnionGamez(ArrayList<Dag> dags) {
        // Genetic algorithm
        System.out.println("Initializing");
        initialize();
        for (int j = 0; j < numIterations; j++) {
            System.out.println("Iteration " + j);
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
        population = new boolean[populationSize][totalEdges];
        boolean uniform = minFreq == maxFreq;

        // Add the greedy solution to the population
        Dag greedy = applyGreedyMaxTreewidth(alpha, edges, String.valueOf(maxTreewidth));
        for (int i = 0; i < totalEdges; i++) {
            population[0][i] = greedy.containsEdge(edges.get(i));
        }

        for (int i = 1; i < populationSize; i++) {
            for (int j = 0; j < totalEdges; j++) {
                if (uniform) {
                    population[i][j] = random.nextBoolean();
                } else {
                    double normalized = (double) (edgeFrequencyArray.get(j) - minFreq) /(maxFreq-minFreq);
                    population[i][j] = random.nextDouble() < 1/(1-Math.log(normalized));
                }
            }
        }
    }

    private void evaluate() {
        fitness = new double[populationSize];
        for (int i = 0; i < populationSize; i++) {
            fitness[i] = calculateFitness(i, 0);
        }

        System.out.println("Fitness: ");
        for (int i = 0; i < populationSize; i++) {
            System.out.printf("%.2f ", fitness[i]);
        }
        System.out.println();
    }

    private void crossover() {
        boolean[][] newPopulation = new boolean[populationSize][totalEdges];
        // Add the best global individual to the new population
        System.out.println("Best individual: " + bestFitness);
        newPopulation[0] = bestIndividual.clone();

        // Add the best individual of the last iteration to the new population
        int bestIndex = 0;
        double bestFitness = 0;
        for (int i = 1; i < populationSize; i++) {
            if (fitness[i] > bestFitness) {
                bestFitness = fitness[i];
                bestIndex = i;
            }
        }
        System.out.println("Best individual of the last iteration: " + bestFitness);
        newPopulation[1] = population[bestIndex];

        // Tournament selection and crossover
        for (int i = 2; i < populationSize; i++) {
            int index1 = random.nextInt(populationSize);
            int index2 = random.nextInt(populationSize);
            int index3 = random.nextInt(populationSize);
            int index4 = random.nextInt(populationSize);
            int crossoverPoint = random.nextInt(totalEdges);

            if (fitness[index1] > fitness[index2]) {
                System.arraycopy(population[index1], 0, newPopulation[i], 0, crossoverPoint);
            } else {
                System.arraycopy(population[index2], 0, newPopulation[i], 0, crossoverPoint);
            }
            if (fitness[index3] > fitness[index4]) {
                System.arraycopy(population[index3], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            } else {
                System.arraycopy(population[index4], crossoverPoint, newPopulation[i], crossoverPoint, totalEdges-crossoverPoint);
            }
        }

        population = newPopulation;

        for (int i = 0; i < populationSize; i++) {
            double x = (double) treeWidths[i] / maxTreewidth;

            // Number of trues
            int numTrues = 0;
            for (int j = 0; j < totalEdges; j++) {
                if (population[i][j]) {
                    numTrues++;
                }
            }
            System.out.println(i + " | FIT " + String.format("%.1f", fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | TRUES: " + numTrues);
        }

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
            System.out.println(i + " | FIT " + String.format("%.1f",fitness[i]) + " | TW " + treeWidths[i] + " | x " + x + " | TRUES: " + numTrues + ", " + finalNumTrues);
        }
    }

    private double calculateFitness(int index, int recursive) {
        // 1. Create the DAG that corresponds to the individual
        Dag union = new Dag(alpha);
        for (int i = 0; i < totalEdges; i++) {
            if (population[index][i]) {
                union.addEdge(edges.get(i));
            }
        }

        // 2. Moralize the graph
        Graph undirectedGraph = moralize(union);

        // 3. Triangulate the graph
        // tetrad-lib/src/main/java/edu/cmu/tetrad/bayes/JunctionTreeAlgorithm.java#L106
        Node[] maximumCardinalityOrdering = GraphTools.getMaximumCardinalityOrdering(undirectedGraph);
        GraphTools.fillIn(undirectedGraph, maximumCardinalityOrdering);

        // 4. Find the maximum clique size
        maximumCardinalityOrdering = GraphTools.getMaximumCardinalityOrdering(undirectedGraph);
        Map<Node, Set<Node>> cliques = GraphTools.getCliques(maximumCardinalityOrdering, undirectedGraph);
        int maxCliqueSize = 0;
        for (Set<Node> clique : cliques.values()) {
            if (clique.size() > maxCliqueSize) {
                maxCliqueSize = clique.size();
            }
        }

        treeWidths[index] = maxCliqueSize;

        double fitness;
        if (maxCliqueSize > maxTreewidth) {
            population[index] = fixIndividual(population[index], cliques, recursive);
            return calculateFitness(index, recursive+1);
            //fitness = union.getNumEdges() * ((double) maxTreewidth / maxCliqueSize);
        } else {
            fitness = union.getNumEdges() * ((double) maxCliqueSize / maxTreewidth);
        }

        if (fitness > bestFitness && maxCliqueSize <= maxTreewidth) {
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
