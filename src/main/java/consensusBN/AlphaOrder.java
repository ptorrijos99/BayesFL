package consensusBN;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Edges;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.graph.Node;
import java.util.HashMap;

public class AlphaOrder {
	ArrayList<Dag_n> setOfDags = null;
	ArrayList<Node> alpha = null;
	ArrayList<Dag_n> setOfauxG = null;
//	ArrayList<int[][]> dpaths = null;
	
	public AlphaOrder(ArrayList<Dag_n> dags){
		
		this.setOfDags = dags;
		this.alpha = new ArrayList<Node>();
		this.setOfauxG = new ArrayList<Dag_n>();
//		this.dpaths = new ArrayList<int[][]>();
		
		for (Dag_n i : setOfDags)	{
			Dag_n aux_G = new Dag_n(i);
			setOfauxG.add(aux_G);
//			dpaths.add(computeDirectedPathFromTo(aux_G));
		}
		
	}
	
	public int[][] computeDirectedPathFromTo(Dag_n graph) {
		
		LinkedList<Edge> dpathNewEdges = new LinkedList<Edge>();
		dpathNewEdges.clear();
		dpathNewEdges.addAll(graph.getEdges());
		List<Node> dpathNodes = null;
		dpathNodes = graph.getNodes();
		
		int numNodes = dpathNodes.size();
		int [][] dpath = new int[numNodes][numNodes];
		
		while (!dpathNewEdges.isEmpty()) {
			Edge edge = dpathNewEdges.removeFirst();
			Node _nodeT = Edges.getDirectedEdgeTail(edge);
			Node _nodeH = Edges.getDirectedEdgeHead(edge);
			int _indexT = dpathNodes.indexOf(_nodeT);
			int _indexH = dpathNodes.indexOf(_nodeH);
			dpath[_indexT][_indexH] = 1;
			int dPathT = 0;
			int dPathH = 0;
			int mindPath = 0;
			for (int i = 0; i < dpathNodes.size(); i++) {
				dPathT = dpath[i][_indexT];
				if (dpath[i][_indexT] >= 1) {
					dPathH = dpath[i][_indexH];
					if(dPathH == 0) dpath[i][_indexH] = dPathT+1;
					else{
						mindPath = Math.min(dPathH, dPathT+1);
						dpath[i][_indexH]=mindPath;
					}
				}
				dPathH = dpath[_indexH][i];
				if(dpath[_indexH][i] >= 1){
					dPathT = dpath[_indexT][i];
					if(dPathT ==0) dpath[_indexT][i] = dPathH+1;
					else{
						mindPath = Math.min(dPathT, dPathH+1);
						dpath[_indexT][i] = mindPath;
					}
					
				}
			}
		}
		
		return dpath;
    }
	
	public List<Node> getNodes(){
		return(setOfDags.get(0).getNodes());
	}
	
	// heursitica para orden de conceso basada en el numero de caminos dirigidos. (Es muy mala no se utiliza)
	
	public void computeAlphaH1(){
		
		List<Node> nodes = setOfDags.get(0).getNodes();
		LinkedList<Node> alpha = new LinkedList<Node>();
		
		while(nodes.size()>0){
			int index_alpha = computeNextH1(nodes);
			Node node_alpha = nodes.get(index_alpha);
			alpha.addFirst(node_alpha);
			for(Dag_n g: this.setOfauxG){
				removeNode(g,node_alpha);
				int[][] newDpaths = computeDirectedPathFromTo(g);
//				this.dpaths.set(this.setOfauxG.indexOf(g), newDpaths);
			}
			nodes.remove(node_alpha);
		}
		this.alpha = new ArrayList<Node>(alpha);
	}
	
	// heuistica para encontrar un orden de conceso. Se basa en los enlaces que generaria seguir una secuencia creada desde los nodos sumideros hacia arriba.
	
public void computeAlphaH2(){
		
		List<Node> nodes = setOfDags.get(0).getNodes();
		LinkedList<Node> alpha = new LinkedList<Node>();
		
		while(nodes.size()>0){
			int index_alpha = computeNextH2(nodes);
			Node node_alpha = nodes.get(index_alpha);
			alpha.addFirst(node_alpha);
			for(Dag_n g: this.setOfauxG){
				removeNode(g,node_alpha);
			}
			nodes.remove(node_alpha);
		}
		this.alpha = new ArrayList<Node>(alpha);
	}
	
	
	
	int computeNextH2(List<Node> nodes){
		
		int changes = 0;
		int inversion = 0;
		int addition = 0;
		int indexNode = 0;
		int min = Integer.MAX_VALUE;
		
		for(int i=0; i<nodes.size(); i++){
			Node nodei = nodes.get(i);
			for(Dag_n g: this.setOfauxG){
				ArrayList<Edge> inserted = new ArrayList<Edge>();
				List<Node> children = g.getChildren(nodei);
				inversion += (children.size()-1);
				List<Node> paX = g.getParents(nodei);
				for(Node child: children){
					List<Node> paY = g.getParents(child);
					for(Node nodep: paX){
							if(g.getEdge(nodep, child)==null){
								addition++;
							}
					}
					for(Node nodec: paY){
						if(!nodec.equals(nodei)){
							if((g.getEdge(nodec,nodei)==null) && (g.getEdge(nodei,nodec)==null)){
								Edge toBeInserted = new Edge(nodec,nodei,Endpoint.CIRCLE,Endpoint.CIRCLE);
								boolean contains = false;
								for(Edge e: inserted){
									if((e.getNode1().equals(nodec) && (e.getNode2().equals(nodei))) || 
									  ((e.getNode1().equals(nodei) && (e.getNode2().equals(nodec))))){
										contains = true;
										break;
									}
								}
								if(!contains){
									addition++;
									inserted.add(toBeInserted);
								}
							}
						}
					}
				}
			}
			changes = inversion + addition;
			if(changes < min){
				min = changes;
				indexNode = i;
			}
			changes = 0;
			inversion = 0;
			addition = 0;
		}
		return indexNode;
	}
	
	void removeNode(Dag_n g, Node node_alpha){
		
		List<Node> children = g.getChildren(node_alpha);
		
		while(!children.isEmpty()){
			int i=0;
			Node child;
			boolean seguir = false;
			do{
				child = children.get(i++);
				g.removeEdge(node_alpha, child);
				seguir=false;
				if(g.existsDirectedPathFromTo(node_alpha,child)){
					seguir=true;
					g.addEdge(new Edge(node_alpha,child,Endpoint.TAIL, Endpoint.ARROW));
				}
			}while(seguir);

			List<Node> paX = g.getParents(node_alpha);
			List<Node> paY = g.getParents(child);
			paY.remove(node_alpha);
			g.addEdge(new Edge(child,node_alpha,Endpoint.TAIL, Endpoint.ARROW));
			for(Node nodep: paX){
				Edge pay = g.getEdge(nodep, child);
				if(pay == null)
					g.addEdge(new Edge(nodep,child,Endpoint.TAIL,Endpoint.ARROW));

			}
			for(Node nodep : paY){
				Edge paz = g.getEdge(nodep,node_alpha);
				if(paz == null) 
					g.addEdge(new Edge(nodep,node_alpha,Endpoint.TAIL,Endpoint.ARROW));
			}

			children.remove(child);
		}
		g.removeNode(node_alpha);
	}


	int computeNextH1(List<Node> nodes){
		
		int min = Integer.MAX_VALUE;
		int minIndex = 0;
		
		for(int i=0 ; i< nodes.size(); i++){
			int weightNodei = 0;
			for(Dag_n dag : this.setOfauxG){
	//			int[][] dpath = this.dpaths.get(this.setOfauxG.indexOf(dag));
	//			for(int j=0 ; j<nodes.size(); j++) weightNodei+= dpath[i][j];
			}
			if(weightNodei < min){
				min = weightNodei;
				minIndex = i;
			}
			
		}
		
		return minIndex;
		
	}
	
	ArrayList<Node> getOrder(){
		
		return this.alpha;
	}
	
	
	public static void main(String args[]) {
		
//		ArrayList<Dag_n> dags = new ArrayList<Dag_n>();
//		ArrayList<Node> alfa = new ArrayList<Node>();
//
//
//		System.out.println("Grafos de Partida:   ");
//		System.out.println("---------------------");
////		Graph graph = GraphConverter.convert("X1-->X5,X2-->X3,X3-->X4,X4-->X1,X4-->X5");
////		Dag_n dag = new Dag_n(graph);
//
//		Dag_n dag = new Dag_n();
//	//	dag = GraphUtils.randomDag(Integer.parseInt(args[0]), Integer.parseInt(args[1]), true);
//		dags.add(dag);
//		System.out.println("DAG: ---------------");
//		System.out.println(dag.toString());
//		for (int i=0 ; i < Integer.parseInt(args[2])-1 ; i++){
//		//	Dag_n newDag = GraphUtils.randomDag(dag.getNodes(),Integer.parseInt(args[1]) ,true);
//			dags.add(newDag);
//			System.out.println("DAG: ---------------");
//			System.out.println(newDag.toString());
//		}
//
//		AlphaOrder order = new AlphaOrder(dags);
//		order.computeAlphaH2();
//		alfa = order.getOrder();
//
//		System.out.println("Orden de Consenso: " + alfa.toString());
		
		
	}
	
	
}
