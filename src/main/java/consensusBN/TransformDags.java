package consensusBN;

import java.util.ArrayList;


import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Dag_n;

public class TransformDags {

	ArrayList<Dag_n> setOfDags = null;
	ArrayList<Dag_n> setOfOutputDags = null;
	ArrayList<Node> alfa = null;
	ArrayList<BetaToAlpha> metAs= null;
	int numberOfInsertedEdges = 0;
//	int weight[][][] = null;
	
	public TransformDags(ArrayList<Dag_n> dags, ArrayList<Node> alfa){
		
		this.setOfDags = dags;
		this.setOfOutputDags = new ArrayList<Dag_n>();
		this.metAs = new ArrayList<BetaToAlpha>();
		this.alfa = alfa;
		
		for (Dag_n i : setOfDags)	{
			Dag_n out = new Dag_n(i);
			this.metAs.add(new BetaToAlpha(out,alfa));
		}
		
	}
	
	
	ArrayList<Dag_n> transform (){
		
		this.numberOfInsertedEdges = 0;
		
		for(BetaToAlpha transformDagi: this.metAs){
			transformDagi.transform();
			this.numberOfInsertedEdges += transformDagi.getNumberOfInsertedEdges();
			this.setOfOutputDags.add(transformDagi.G);
		}
		
		
		return this.setOfOutputDags;
		
	}
	
	public int getNumberOfInsertedEdges(){
		return this.numberOfInsertedEdges;
	}
	
	
	
	
//	void computeWeight(){
//		
//		this.weight = new int[this.setOfDags.size()][this.alfa.size()][this.alfa.size()];
//		
//		for(Dag_n g: this.setOfOutputDags){
//			for(Node nodei : g.getNodes()){
//				List<Node> pa = g.getParents(nodei);
//				if(pa.isEmpty()) continue;
//				List<Node> anc = new ArrayList<Node>();
//				anc.add(nodei);
//				anc = g.getAncestors(anc);
//				Dag_n gAnc =  new Dag_n(g.subgraph(anc));
//				// me quedo con el grafo ancestral del node_i
//				for(Node pai: pa){ // Calculo el numero de  caminos desde los ancestros que se "activan" borrando cada padre.
//					int npaths = 0;
//					Dag_n gAncNopai = new Dag_n(gAnc);
//					for(Node rm : pa) if(!rm.equals(pai)) gAncNopai.removeNode(rm); // borro todos los padres menos el pa_i en el grafo ancestral.
//					for(Node nodeAc: anc){ // para cada ancestro voy mirando si hay un camino dirigido.
//						if((gAncNopai.getNodes().contains(nodeAc))&&(!nodeAc.equals(nodei)))
//							if(GraphUtils.existsDirectedPathFromTo(gAncNopai,nodeAc, nodei)) npaths++;
//					}
//					// npaths tiene el numero de caminos diridos que se han activado quitando el padre pa_i
//					this.weight[this.setOfOutputDags.indexOf(g)][g.getNodes().indexOf(nodei)][g.getNodes().indexOf(pai)] = npaths;
//				}
//				
//			}
//			
//		}
//
//	}
	
//	public Dag_n computeWeightDag(boolean w){
//	
//		Dag_n wDag = new Dag_n(this.alfa);
//		for(Node nodei : this.alfa){
//			for(Node nodej : this.alfa){
//				if(nodei.equals(nodej)) continue;
//				int wij = 0;
//				for(Dag_n g: this.setOfOutputDags){
//					int wg = this.weight[this.setOfOutputDags.indexOf(g)][g.getNodes().indexOf(nodei)][g.getNodes().indexOf(nodej)];
//					if(wg > 0 && !w) wg = 1;
//					else if (wg == 0) wg =-1;
//					wij+=wg;
//				}
//				if(wij > 0) wDag.addEdge(new Edge(nodej,nodei,Endpoint.TAIL,Endpoint.ARROW));
//			}
//		}
//		return wDag;
//	}
//	
//	public static void main(String args[]) {
//		
//		ArrayList<Dag_n> dags = new ArrayList<Dag_n>();
//		ArrayList<Node> alfa = new ArrayList<Node>();
//		Random aleatorio = new Random(222);
//		
//		
//		System.out.println("Grafos de Partida:   ");
//		System.out.println("---------------------");
////		Graph graph = GraphConverter.convert("X1-->X5,X2-->X3,X3-->X4,X4-->X1,X4-->X5");
////		Dag_n dag = new Dag_n(graph);
//		
//		Dag_n dag = new Dag_n();
//		
//		dag = GraphUtils.randomDag(Integer.parseInt(args[0]), Integer.parseInt(args[1]), true);
//		BetaToAlpha mt = new BetaToAlpha(dag);
//		alfa = mt.randomAlfa(aleatorio);
//		dags.add(dag);
//		System.out.println("DAG: ---------------");
//		System.out.println(dag.toString());
//		for (int i=0 ; i < Integer.parseInt(args[2])-1 ; i++){
//			Dag_n newDag = GraphUtils.randomDag(dag.getNodes(),Integer.parseInt(args[1]) ,true);
//			dags.add(newDag);
//			System.out.println("DAG: ---------------");
//			System.out.println(newDag.toString());
//		}
//		
//		
//		
//		System.out.println("Orden de Consenso: " + alfa.toString());
//		
//		TransformDags setOfDags = new TransformDags(dags,alfa);
//		setOfDags.transform();
//		
//		
//		
//		
//		for(Dag_n d : setOfDags.setOfOutputDags){
//			System.out.println("DAG trasformado: ---------------");
//			System.out.println(d.toString());
//		}
//		
//		
//		
//		Dag_n union = new Dag_n(alfa);
//		
//		for(Node nodei: alfa){
//			for(Dag_n d : setOfDags.setOfOutputDags){
//				List<Node>parent = d.getParents(nodei);
//				for(Node pa: parent){
//					if(!union.isParentOf(pa, nodei)) union.addEdge(new Edge(pa,nodei,Endpoint.TAIL,Endpoint.ARROW));
//				}
//			}
//			
//		}
//		
//		
//		System.out.println("Grafo UNION: "+union.toString());
//		setOfDags.computeWeight();
//		Dag_n wDag = setOfDags.computeWeightDag(true);
//		System.out.println("Grafo Consenso: "+ wDag.toString());
//		Dag_n wDag2 = setOfDags.computeWeightDag(false);
//		System.out.println("Grafo Consenso sin pesos: "+ wDag2.toString());
//		
//		
//
//		
//		
//		
//		
////		Node nod = dag.getNodes().get(aleatorio.nextInt(alfa.size()));
////		ArrayList<Node> a = new ArrayList<Node>();
////		a.add(nod);
////		List<Node> anc = dag.getAncestors(a);
////	
////		System.out.println("Ancenstros de " + nod.toString()+ " "+anc.toString());
////		
////		System.out.println("Subgraph: "+ dag.subgraph(anc));
////	
////		List<Node> pa = dag.getParents(nod);
////		System.out.println("padres de: "+nod.toString()+ " :  "+pa.toString());
////		Node pai = pa.get(aleatorio.nextInt(pa.size()));
////		System.out.println("Padre elegido a borrar: "+pai.toString());
////		pa.remove(pai);
////		
////		
////		
////		for(Node rm: pa) anc.remove(rm);
////		
////		Graph pp =  dag.subgraph(anc);
////		int npath = 0;
////		for(Node ancestor: pp.getNodes()){
////			if(!ancestor.equals(nod)){
////				List<List<Node>> paths = GraphUtils.allPathsFromTo(pp, ancestor, nod);
////				if(dag.existsDirectedPathFromTo(ancestor, nod)) npath++;
////				System.out.println("Caminos desde: "+ancestor.toString()+"  a:  "+nod.toString()+"  : "+paths.toString());
////			}
////		}
////		System.out.println(" El numero de caminos desde hacia: "+ nod+" es de: "+npath);
//	}
//	
}
