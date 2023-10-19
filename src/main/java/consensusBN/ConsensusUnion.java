package consensusBN;

import java.util.ArrayList;
import java.util.List;

import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Endpoint;
import edu.cmu.tetrad.graph.Node;

public class ConsensusUnion {
	
	ArrayList<Node> alpha = null;
	AlphaOrder heuristic = null;
	TransformDags imaps2alpha = null;
	ArrayList<Dag> setOfdags = null;
	Dag union = null;
	int numberOfInsertedEdges = 0;
	
	
	public ConsensusUnion(ArrayList<Dag> dags, ArrayList<Node> order){
		this.setOfdags = dags;
		this.alpha = order;
		
	}	
	
	
	
	public ConsensusUnion(ArrayList<Dag> dags){
		this.setOfdags = dags;
		this.heuristic = new AlphaOrder(this.setOfdags);
		
	}	
	
	public ConsensusUnion(){
		this.setOfdags = null;
	}
	
	public int getNumberOfInsertedEdges(){
		return this.numberOfInsertedEdges;
	}
	
	public Dag union(){
            
		if(this.alpha == null){
			
			this.heuristic.computeAlphaH2();
			this.alpha = this.heuristic.alpha;
		}
		
		this.imaps2alpha = new TransformDags(this.setOfdags,this.alpha);
		this.imaps2alpha.transform();
		this.numberOfInsertedEdges = this.imaps2alpha.getNumberOfInsertedEdges();
	
		this.union = new Dag(this.alpha);
		for(Node nodei: this.alpha){
			for(Dag d : this.imaps2alpha.setOfOutputDags){
				List<Node>parent = d.getParents(nodei);
				for(Node pa: parent){
					if(!this.union.isParentOf(pa, nodei)) this.union.addEdge(new Edge(pa,nodei,Endpoint.TAIL,Endpoint.ARROW));
				}
			}
			
		}
		return this.union;
		
	}
	
	public Dag getUnion(){
	
		return this.union;
		
	}
	
	void setDags(ArrayList<Dag> dags){
		this.setOfdags = dags;
		this.heuristic = new AlphaOrder(this.setOfdags);
		this.heuristic.computeAlphaH2();
		this.alpha = this.heuristic.alpha;
		this.imaps2alpha = new TransformDags(this.setOfdags,this.alpha);
		this.imaps2alpha.transform();
	}

}
