package consensusBN.circularFusion;

import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Graph;
import org.albacete.simd.framework.FESFusion;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.threads.FESThread;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Problem;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;
import static org.albacete.simd.utils.Utils.pdagToDag;

public class CircularDag {
    public Dag_n dag;
    public final int id;
    public boolean convergence = false;
    
    private double bdeu = Double.NEGATIVE_INFINITY;
    private double lastBdeu;
    private final Problem problem;
    private final Set<Edge> subsetEdges;
    private final int nItInterleaving;
    
    private Dag_n inputDag;
    
    public static final String EXPERIMENTS_FOLDER = "./experiments/";

    public CircularDag(Problem problem, Set<Edge> subsetEdges, int nItInterleaving, int id) {
        this.id = id;
        this.problem = problem;
        this.subsetEdges = subsetEdges;
        this.nItInterleaving = nItInterleaving;
        this.dag = new Dag_n(problem.getVariables());
    }

    public void fusionGES() throws InterruptedException {

        // Setup
        // 1. Update bdeu and convergence variables
        setup();
        
        // 2. Check if the input dag is empty
        if (!inputDag.getEdges().isEmpty()) {
            // 3. Merge dags into an arraylist
            ArrayList<Dag_n> dags = mergeBothDags(inputDag);

            // 4. FES Fusion (Consensus Fusion + FES)
            applyFESFusion(dags);
        }
        
        // 5. GES Stage
        applyGES();

        // 6. Update bdeu value
        updateResults();

        // 7. Convergence
        checkConvergence();
    }
    
    private void setup() {
        lastBdeu = bdeu;
        convergence = false;
    }
    
    private ArrayList<Dag_n> mergeBothDags(Dag_n dag2) {
        ArrayList<Dag_n> dags = new ArrayList<>();
        dags.add(dag);
        dags.add(dag2);
        return dags;
    }
    
    private void applyFESFusion(ArrayList<Dag_n> dags) throws InterruptedException {
        FESFusion fusion = new FESFusion(problem, dag, dags, true);
        dag = fusion.fusion();
        printResults(id, "Fusion", GESThread.scoreGraph(dag, problem));
        
        // Do the BESThread to complete the GES of the fusion
        BESThread bes = new BESThread(problem, dag, dag.getEdges());
        bes.run();

        dag = transformPDAGtoDAG(bes.getCurrentGraph());
    }

    private void printResults(int id, String stage, double BDeu) {
        String savePath = EXPERIMENTS_FOLDER + "temp_" + id + ".csv";
        File file = new File(savePath);
        FileWriter csvWriter = null;
        try {
            csvWriter = new FileWriter(file, true);
            csvWriter.append(id + "," + stage + "," + BDeu + "\n");
            csvWriter.flush();
        } catch (IOException ex) {
        }
    }

    private void applyGES() throws InterruptedException {
        // Do the FESThread
        FESThread fes = new FESThread(problem, dag, subsetEdges, this.nItInterleaving, false, true, true);
        fes.run();
        Graph fesGraph = fes.getCurrentGraph();
        fesGraph = transformPDAGtoDAG(fesGraph);
        printResults(id, "FES", GESThread.scoreGraph(fesGraph, problem));
    
        // Do the BESThread to complete the GES of the fusion
        BESThread bes = new BESThread(problem, fesGraph, subsetEdges);
        bes.run();

        dag = transformPDAGtoDAG(bes.getCurrentGraph());
    }

    private Dag_n transformPDAGtoDAG(Graph besGraph) {
        pdagToDag(besGraph);
        return new Dag_n(besGraph);
    }
    
    private void updateResults() {
        bdeu = GESThread.scoreGraph(dag, problem);
        printResults(id, "BES", bdeu);
    }

    private void checkConvergence() {
        if (bdeu <= lastBdeu) convergence = true;
    }
    
    public double getBDeu() {
        return bdeu;
    }

    public void setBDeu(double bdeu) {
        this.bdeu = bdeu;
    }

    public void setInputDag(Dag_n inputDag) {
        this.inputDag = inputDag;
    }
}
