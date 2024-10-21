package consensusBN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.EdgeListGraph;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.Node;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.threads.GESThread;

import static org.albacete.simd.utils.Utils.pdagToDag;

public class HeuristicConsensusBES {
	
	ArrayList<Node> alpha;
	Dag outputDag = null;
	ArrayList<Dag> setOfdags;
	ArrayList<Dag> setOfOutDags;
	Dag union;
	Map<String, Double> localScore = new HashMap<>();
	double percentage;  // 1.0
	int maxSize;  // 10

	
	public HeuristicConsensusBES(ArrayList<Dag> dags, double percentage, int maxSize){
		this.setOfdags = dags;
		this.alpha = AlphaOrder.alphaOrder(dags);
		this.setOfOutDags = new ArrayList<>();
		for (Dag d : dags) {
			this.setOfOutDags.add(BetaToAlpha.transformToAlpha(d, this.alpha));
		}
		this.percentage = percentage;
		this.maxSize = maxSize;
		this.union = ConsensusUnion.fusionUnion(this.setOfOutDags);
	}

	public Dag fusion(){
		double score = 0;
		double bestScore = score;

		Graph graph = new EdgeListGraph(this.union);
		Node x, y;
		HashSet<Node> bestSubSet = new HashSet<>();
		do {
			x = y = null;
			Set<Edge> edges1 = graph.getEdges();
			List<Edge> edges = new ArrayList<>();

			for (Edge edge : edges1) {
				Node _x = edge.getNode1();
				Node _y = edge.getNode2();

				if (Edges.isUndirectedEdge(edge)) {
					edges.add(Edges.directedEdge(_x, _y));
					edges.add(Edges.directedEdge(_y, _x));
				} else {
					edges.add(edge);
				}
			}
			for (Edge edge : edges) {
				Node _x = edge.getNode1();
				Node _y = edge.getNode2();

				double evalScore;
				List<Node> hNeighbors = getHNeighbors(_x, _y, graph);
				List<HashSet<Node>> hSubsets = BESThread.generatePowerSet(hNeighbors);

				for (HashSet<Node> hSubset : hSubsets) {
					if(hSubset.size() > maxSize) break;
					double deleteEval = deleteEval(_x, _y, hSubset, graph);
					if (!(deleteEval >= this.percentage)) deleteEval = 0.0;
					evalScore = score + deleteEval;

             		// System.out.println("Attempt removing " + _x + "-->" + _y + "(" +evalScore + ") "+ hSubset.toString());

					if (!(evalScore > bestScore)) {
						continue;
					}

					// INICIO TEST 1
					List<Node> naYXH = findNaYX(_x, _y, graph);
					naYXH.removeAll(hSubset);
					if (!isClique(naYXH, graph)) {
		                bestScore = evalScore;
						bestSubSet = hSubset;
						x = _x;
						y = _y;
						continue;
					}
					// FIN TEST 1
					break;
				}
			}
			if (x != null) {
				//System.out.println("DELETE " + graph.getEdge(x, y) + t.toString() + " (" +bestScore + ")");
				GESThread.delete(x, y, bestSubSet, graph);
				score = bestScore;
			}
		} while (x != null);
//		System.out.println("Pdag: "+ graph.toString());
		pdagToDag(graph);
//		System.out.println("PdagToDag"+graph.toString());
		this.outputDag = new Dag();
		for (Node node : graph.getNodes()) this.outputDag.addNode(node);
		Node nodeT, nodeH;
		for (Edge e : graph.getEdges()){
			if(!e.isDirected()) continue;
			Endpoint endpoint1 = e.getEndpoint1();
			if (endpoint1.equals(Endpoint.ARROW)){
				nodeT = e.getNode1(); 
				nodeH = e.getNode2();
			}else{
				nodeT = e.getNode2();
				nodeH = e.getNode1();
			}
			if(!this.outputDag.paths().existsDirectedPath(nodeT, nodeH)) this.outputDag.addEdge(e);
		}
//		System.out.println("DAG: "+this.outputDag.toString());
		return this.outputDag;
	}

	private static boolean isClique(List<Node> set, Graph graph) {
		List<Node> setv = new LinkedList<>(set);
		for (int i = 0; i < setv.size() - 1; i++) {
			for (int j = i + 1; j < setv.size(); j++) {
				if (!graph.isAdjacentTo(setv.get(i), setv.get(j))) {
					return false;
				}
			}
		}
		return true;
	}

	private static List<Node> getHNeighbors(Node x, Node y, Graph graph) {
		List<Node> hNeighbors = new LinkedList<>(graph.getAdjacentNodes(y));
		hNeighbors.retainAll(graph.getAdjacentNodes(x));

		for (int i = hNeighbors.size() - 1; i >= 0; i--) {
			Node z = hNeighbors.get(i);
			Edge edge = graph.getEdge(y, z);
			if (!Edges.isUndirectedEdge(edge)) {
				hNeighbors.remove(z);
			}
		}

		return hNeighbors;
	}
	
	
	double deleteEval(Node x, Node y, Set<Node> h, Graph graph){
		 Set<Node> set1 = new HashSet<>(findNaYX(x, y, graph));
		 set1.removeAll(h);
		 set1.addAll(graph.getParents(y));
		 set1.remove(x);
		 return scoreGraphChangeDelete(y, x, set1); // calcular si y esta d-separado de x dado el set1 en cada grafo.
	}
	
	double scoreGraphChangeDelete(Node y, Node x, Set<Node> set){
		String key = y.getName()+x.getName()+set.toString();
		Double val = this.localScore.get(key);
		if(val == null) {
			val = 0.0;
			LinkedList<Node> conditioning = new LinkedList<>(set);
			for (Dag g : this.setOfdags) {
				if (dSeparated(g, y, x, conditioning)) ++val;
			}
			val = val / (double) this.setOfdags.size();
			this.localScore.put(key, val);
		}
		return val;
	}

	boolean dSeparated(Dag g, Node x, Node y, LinkedList<Node> cond){
		LinkedList<Node> open = new LinkedList<>();
		HashMap<String,Node> close = new HashMap<>();
		open.add(x);
		open.add(y);
		open.addAll(cond);
		while (!open.isEmpty()){
			Node a = open.getFirst();
			open.remove(a);
			close.put(a.toString(),a);
			List<Node> pa =g.getParents(a);
			for(Node p : pa){
				if(close.get(p.toString()) == null){
					if(!open.contains(p)) open.addLast(p);
				}
			}
		}

		Graph aux = new EdgeListGraph();

		for (Node node : g.getNodes()) aux.addNode(node);
		Node nodeT, nodeH;
		for (Edge e : g.getEdges()){
			if(!e.isDirected()) continue;
			nodeT = e.getNode1();
			nodeH = e.getNode2();
			if((close.get(nodeH.toString())!=null)&&(close.get(nodeT.toString())!=null)){
				Edge newEdge = new Edge(e.getNode1(),e.getNode2(),e.getEndpoint1(),e.getEndpoint2());
				aux.addEdge(newEdge);
			}
		}

		close = new HashMap<>();
		for(Edge e: aux.getEdges()){
			if(e.isDirected()){
				Node h;
				if(e.getEndpoint1()==Endpoint.ARROW){
					h = e.getNode1();
				}else h = e.getNode2();
				if(close.get(h.toString())==null){
					close.put(h.toString(),h);
					List<Node> pa = aux.getParents(h);
					if(pa.size()>1){
						for(int i = 0 ; i< pa.size() - 1; i++) {
							for (int j = i + 1; j < pa.size(); j++) {
								Node p1 = pa.get(i);
								Node p2 = pa.get(j);
								boolean found = false;
								for (Edge edge : aux.getEdges()) {
									if (edge.getNode1().equals(p1) && (edge.getNode2().equals(p2))) {
										found = true;
										break;
									}
									if (edge.getNode2().equals(p1) && (edge.getNode1().equals(p2))) {
										found = true;
										break;
									}
								}
								if (!found) aux.addUndirectedEdge(p1, p2);
							}
						}
					}
				}
			}
		}

		for(Edge e: aux.getEdges()){
			if(e.isDirected()){
				e.setEndpoint1(Endpoint.TAIL);
				e.setEndpoint2(Endpoint.TAIL);
			}
		}

		aux.removeNodes(cond);

		open = new LinkedList<>();
		close = new HashMap<>();
		open.add(x);
		while (!open.isEmpty()){
			Node a = open.getFirst();
			if(a.equals(y)) return false;
			open.remove(a);
			close.put(a.toString(),a);
			List<Node> pa =aux.getAdjacentNodes(a);
			for(Node p : pa){
				if(close.get(p.toString()) == null){
					if(!open.contains(p)) open.addLast(p);
				}
			}
		}

		return true;
	}

    private static List<Node> findNaYX(Node x, Node y, Graph graph) {
        List<Node> naYX = new LinkedList<>(graph.getAdjacentNodes(y));
        naYX.retainAll(graph.getAdjacentNodes(x));

        for (int i = naYX.size()-1; i >= 0; i--) {
            Node z = naYX.get(i);
            Edge edge = graph.getEdge(y, z);

            if (!Edges.isUndirectedEdge(edge)) {
                naYX.remove(z);
            }
        }

        return naYX;
    }
    
    public Dag getFusion(){
    	return this.outputDag;
    }

}
