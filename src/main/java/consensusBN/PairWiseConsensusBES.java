package consensusBN;

import java.util.ArrayList;


import edu.cmu.tetrad.graph.Dag_n;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;



public class PairWiseConsensusBES implements Runnable{
	
	private Dag_n b1 = null;
	private Dag_n b2 = null;
	private Dag_n conDAG = null;
	private ConsensusBES conBES= null;
	private int numberOfInsertedEdges = 0;
	private int numberOfUnionEdges = 0;
	
	public PairWiseConsensusBES(Dag_n b1, Dag_n b2) {
		super();
		this.b1 = b1;
		this.b2 = b2;
	}
	
	public void getFusion(){
		ArrayList<Dag_n> setOfDags = new ArrayList<Dag_n>();
		setOfDags.add(this.b1);
		setOfDags.add(this.b2);
		conBES = new ConsensusBES(setOfDags);
		conBES.fusion();
		this.numberOfInsertedEdges = conBES.getNumberOfInsertedEdges();
		this.numberOfUnionEdges  = conBES.union.getNumEdges();
		this.conDAG = conBES.getFusion();
	}
	
	public int getNumberOfInsertedEdges(){
		return this.numberOfInsertedEdges;
	}
	
	public int getNumberOfUnionEdges(){
		return this.numberOfUnionEdges;
	}
	
	public int getHammingDistance(){
		if(this.conDAG==null) this.getFusion();
		int distance = 0;
		for(Edge ed: this.conDAG.getEdges()){
			Node tail = ed.getNode1();
			Node head = ed.getNode2();
			for(Dag_n g: conBES.setOfOutDags){	
				Edge edge1 = g.getEdge(tail, head);
				Edge edge2 = g.getEdge(head, tail);
				if(edge1 == null && edge2==null) distance++;
			}
		}
		return distance+this.getNumberOfInsertedEdges();
	}
	
	public Dag_n getDagFusion(){
		return this.conDAG;
		
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		this.getFusion();
	}

}
