package consensusBN;



import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;


public class BetaToAlpha {

	Dag_n G = null;
	ArrayList<Node> beta = new ArrayList<Node>();
	ArrayList<Node> alfa = new ArrayList<Node>();
	HashMap<Node,Integer> alfaHash= new HashMap<Node,Integer>();
	Dag_n G_aux = null;
	int numberOfInsertedEdges = 0;
	
	public BetaToAlpha(Dag_n G, ArrayList<Node> alfa){

		this.alfa = alfa;
		this.G = G;
		this.beta = null;
		for(int i= 0; i< alfa.size(); i++){
			Node n = alfa.get(i);
			alfaHash.put(n, new Integer(i));
		}
		
	}
	
	public BetaToAlpha(Dag_n G){

		this.alfa = null;
		this.G = G;
		this.beta = null;
		
	}

	void computeAlfaHash(){
		
		if(this.alfa !=null){
			if(alfaHash.isEmpty()){
				for(int i= 0; i< alfa.size(); i++){
					Node n = alfa.get(i);
					alfaHash.put(n, new Integer(i));
				}
			}
		}
		
	}
	
	// Only to test the methods, to build a random order.
	
	
	public ArrayList<Node> randomAlfa (Random aleatorio){
		
		List<Node> nodes = this.G.getNodes();
		this.alfa = new ArrayList<Node>();

		int[] index = new int[nodes.size()];

		for(int i = 0; i< nodes.size() ; i++){
			index[i]=i;
		}

		for (int j = 0; j < nodes.size(); j++){

			int indi = aleatorio.nextInt(nodes.size());
			int indj = aleatorio.nextInt(nodes.size());
			int sw = index[indi];
			index[indi] = index[indj];
			index[indj] = sw;
		}

		for (int i = 0; i< nodes.size(); i++){
			this.alfa.add(i, nodes.get(index[i]));
		}
		this.computeAlfaHash();
		return this.alfa;
	}
	
	
	public void transform(){
		
		this.G_aux = new Dag_n(this.G);
		this.beta = new ArrayList<Node>();
		ArrayList<Node> sinkNodes = getSinkNodes(this.G_aux);
		this.beta.add(sinkNodes.get(0)); 
		List<Node> pa = G_aux.getParents(sinkNodes.get(0));
		this.G_aux.removeNode(sinkNodes.get(0));
		sinkNodes.remove(0); 
		// Compute the new sink nodes
		for(Node nodep: pa){
			List<Node> chld = G_aux.getChildren(nodep);
			if (chld.size() == 0) sinkNodes.add(nodep);
		}

		// Construct beta order as closer as possible to alfa.
		
		while (this.G_aux.getNumNodes()>0){
			//	sinkNodes = getSinkNodes(this.G_aux);
			Node sink = sinkNodes.get(0);
			pa = G_aux.getParents(sink);
			this.G_aux.removeNode(sink);
			sinkNodes.remove(0);
			// Compute the new sink nodes
			for(Node nodep: pa){
				List<Node> chld = G_aux.getChildren(nodep);
				if (chld.size() == 0) sinkNodes.add(nodep);
			}

			int index_alfa_sink =  this.alfaHash.get(sink);    //this.alfa.indexOf(sink);
			boolean ok = true;
			int i = 0;
			
			while(ok){
				
				Node nodej = this.beta.get(i);
				int index_alfa_nodej =  this.alfaHash.get(nodej); //this.alfa.indexOf(nodej);
			
				if (index_alfa_nodej > index_alfa_sink){ ok = false; break;}
				if (this.G.getParents(nodej).contains(sink)){ ok = false; break;}
				if (i == this.beta.size()-1){ ok = false; break;}
				i++;
			}
			
			this.beta.add(i,sink);
		}
		
		// transform graph G into an I-map minimal with alpha order
		
		ArrayList<Node> aux_beta = new ArrayList<Node>();
		aux_beta.add(this.beta.get(0));
		this.beta.remove(0);
		
		while(this.beta.size()>0){ // check each variable from the sink nodes.
			
			aux_beta.add(this.beta.get(0));
			this.beta.remove(0);
			int i = aux_beta.size();
			boolean ok = true;
			
			while (ok){
				
				if(i==1) break;
				ok = false;
				Node nodeY = aux_beta.get(i-1);
				Node nodeZ = aux_beta.get(i-2);
			
//				if ((nodeZ != null) && (this.alfa.indexOf(nodeZ) > this.alfa.indexOf(nodeY))){
				if ((nodeZ != null) && (this.alfaHash.get(nodeZ) > this.alfaHash.get(nodeY))){
					if(this.G.getEdge(nodeZ, nodeY) != null){
						List<Node> paZ = this.G.getParents(nodeZ);
						List<Node> paY = this.G.getParents(nodeY);
						paY.remove(nodeZ);
						this.G.removeEdge(nodeZ, nodeY);
						this.G.addEdge(new Edge(nodeY,nodeZ,Endpoint.TAIL, Endpoint.ARROW));
						for(Node nodep: paZ){
							Edge pay = this.G.getEdge(nodep, nodeY);
							if(pay == null){
								this.G.addEdge(new Edge(nodep,nodeY,Endpoint.TAIL,Endpoint.ARROW));
								this.numberOfInsertedEdges++;
							}
						}
						for(Node nodep : paY){
							Edge paz = this.G.getEdge(nodep,nodeZ);
							if(paz == null){
								this.G.addEdge(new Edge(nodep,nodeZ,Endpoint.TAIL,Endpoint.ARROW));
								this.numberOfInsertedEdges++;
							}
						}
					}
					ok = true;
					aux_beta.remove(nodeY);
					aux_beta.add(i-2,nodeY);
					i--;	
				}
			}
		}
		
		this.beta = aux_beta;
		
	}
	
	public int getNumberOfInsertedEdges(){
		
		return this.numberOfInsertedEdges;
	}
	
	ArrayList<Node> getSinkNodes(Dag_n g){
		
		ArrayList<Node> sourcesNodes = new ArrayList<Node>();
		List<Node> nodes = g.getNodes();
		
		for (Node nodei : nodes){
			if(g.getChildren(nodei).isEmpty()) sourcesNodes.add(nodei);
		}
		return sourcesNodes;
		
	}
	
	
	
//	public static void main(String args[]) {
//
//		 //Graph graph = GraphConverter.convert("X1-->X2,X1-->X3,X2-->X4,X3-->X4");
//		 Graph graph = GraphConverter.convert("X2-->X1,X3-->X1,X1-->X4,X5-->X4,X4-->X6");
//		 Dag_n dag = new Dag_n(graph);
//		
//	     Dag_n dag2 = GraphUtils.randomDag(dag.getNodes(), 7, true);
////	     BayesPm bayesPm = new BayesPm(dag, 3, 3);
////	     MlBayesIm bayesIm = new MlBayesIm(bayesPm);
////	     
////	     Element element = BayesXmlRenderer.getElement(bayesIm);
////	     System.out.println("Started with this bayesIm: " + bayesIm);
////	     System.out.println("\nGot this XML for it:");
////	     Document xmldoc = new Document(element);
////	     Serializer serializer = new Serializer(System.out);
////	     serializer.setLineSeparator("\n");
////	     serializer.setIndent(2);
////	     try {
////	    	 serializer.write(xmldoc);  
////	     }
////	     catch (IOException e) {
////	    	 throw new RuntimeException(e);
////	     }
//	     
//	     
//	     System.out.println(GraphUtils.graphToDot(dag));
//	     
//	     
////	     System.out.println("Dag_n Inicial: "+ dag.toString());
//	     
//	     Random aleatorio = new Random(150);
//	     BetaToAlpha mt = new BetaToAlpha(dag);  
//	     mt.randomAlfa (aleatorio);
//	     mt.transform();
////	     System.out.println(mt.G.toString()+" Alfa: "+mt.alfa.toString()+" Beta:  "+ mt.beta.toString() );
//	     
//	     System.out.println(GraphUtils.graphToDot(mt.G));
//	     
//	     
//	     
////	     System.out.println("Dag_n Inicial: "+ dag2.toString());
//	     
//	     System.out.println(GraphUtils.graphToDot(dag2));
//	     
//	     BetaToAlpha mt2 = new BetaToAlpha(dag2);
//	     Random aleat2 = new Random(150);
//	     mt2.randomAlfa(aleat2);
//	     mt2.transform();
//	     
////	     System.out.println(mt2.G.toString()+" Alfa: "+mt2.alfa.toString()+" Beta:  "+ mt2.beta.toString() ); 
//	     
//	     System.out.println(GraphUtils.graphToDot(mt2.G));
//	     
//	     
//
//	}
	   

		
	}

		
