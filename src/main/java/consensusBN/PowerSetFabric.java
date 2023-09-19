package consensusBN;

import java.util.List;

import edu.cmu.tetrad.graph.Node;

public class PowerSetFabric {
	
	public static final int MODE_FES=0;
	public static final int MODE_BES=1;
	
	private static int mode=MODE_FES;
	private static boolean usePowerSetsCache=false;
//	
//	private static HashMap<Node, HashMap<Node, PowerSet>> hashMap=new HashMap<Node, HashMap<Node, PowerSet>>();
//	
//	private PowerSetFabric() {
//	}
	
	public static PowerSet getPowerSet(List<Node> nodes, int k){
		return new PowerSet(nodes,k);
		
	}
	
	public static PowerSet getPowerSet(Node x, Node y, List<Node> nodes) {
		return new PowerSet(nodes);
//		if(!usePowerSetsCache)
//			return new PowerSet(nodes);
//		PowerSet pSet=get(x,y);
//		if(pSet==null || pSet.nodes.size()!=nodes.size()) { // if(pSet==null || !pSet.t.equals(nodes)) {
//			pSet=new PowerSet(nodes);
//			put(x,y,pSet);
//		}
//		else {
//			pSet.reset(mode==MODE_FES);
//		}
//		return pSet;
	}
//	
//	private static void put(Node x, Node y, PowerSet pSet) {
//		hashMap.get(x).put(y, pSet);
//		
//	}
//	
//	private static PowerSet get(Node x, Node y) {
//		HashMap<Node, PowerSet> aux=hashMap.get(x);
//		if(aux==null) {
//			aux=new HashMap<Node,PowerSet>();
//			hashMap.put(x, aux);
//		}
//		return aux.get(y);
//	}
//	
//	public static int getMode() {
//		return mode;
//	}
//	
	public static void setMode(int mode) {
		PowerSetFabric.mode=mode;
	}

	public static boolean isUsePowerSetsCache() {
		return usePowerSetsCache;
	}

	public static void setUsePowerSetsCache(boolean usePowerSetsCache) {
		PowerSetFabric.usePowerSetsCache = usePowerSetsCache;
	}
}
