package consensusBN;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;


import edu.cmu.tetrad.graph.Node;

public class PowerSet implements Enumeration<SubSet> {
	List<Node> nodes;
	private List<SubSet> subSets;
	private int index;
	private int[] lista;
	private HashMap<Integer,SubSet> hashMap;
	
	
	PowerSet(List<Node> nodes,int k) {
		if(nodes.size()<=k)
			k=nodes.size();
		this.nodes=nodes;
		subSets = new ArrayList<SubSet>();
		index=0;
		hashMap=new HashMap<Integer, SubSet>();
		lista=ListFabric.getList(nodes.size());
		for (int i : lista) {
			SubSet newSubSet = new SubSet();
			String selection = Integer.toBinaryString(i);
			for (int j = selection.length() - 1; j >= 0; j--) {
				if (selection.charAt(j) == '1') {
					newSubSet.add(nodes.get(selection.length() - j - 1));
				}
			}
			if(newSubSet.size()<=k){
				subSets.add(newSubSet);
				hashMap.put(new Integer(i), newSubSet);
			}
		}
	}

	
	PowerSet(List<Node> nodes) {
		if(nodes.size()>maxPow)
			maxPow=nodes.size();
		this.nodes=nodes;
		subSets = new ArrayList<SubSet>();
		index=0;
		hashMap=new HashMap<Integer, SubSet>();
		lista=ListFabric.getList(nodes.size());
		for (int i : lista) {
			SubSet newSubSet = new SubSet();
			String selection = Integer.toBinaryString(i);
			for (int j = selection.length() - 1; j >= 0; j--) {
				if (selection.charAt(j) == '1') {
					newSubSet.add(nodes.get(selection.length() - j - 1));
				}
			}
			subSets.add(newSubSet);
			hashMap.put(new Integer(i), newSubSet);
		}
	}
    
	public boolean hasMoreElements() {
		return index<subSets.size();
	}

	public SubSet nextElement() {
		return subSets.get(index++);
	}
	
	public void resetIndex(){
		this.index = 0;
	}
	
	private static int maxPow = 0;
	public static long maxPowerSetSize() {
		return (long) Math.pow(2,maxPow);
	}
	
//	public void firstTest(boolean result) {
//		int numInicial=lista[index-1];
//		for(int i=0;i<lista.length;i++) {
//			if((lista[i] & numInicial)==numInicial) {
//				if(result)
//					hashMap.get(lista[i]).firstTest=SubSet.TEST_TRUE;
//				else
//					hashMap.get(lista[i]).firstTest=SubSet.TEST_FALSE;
//			}
//		}
//	}
//	
//	public void secondTest(boolean result) {
//		int numInicial=lista[index-1];
//		for(int i=0;i<lista.length;i++) {
//			if((lista[i] & numInicial)==numInicial) {
//				if(result)
//					hashMap.get(lista[i]).secondTest=SubSet.TEST_TRUE;
//				else
//					hashMap.get(lista[i]).secondTest=SubSet.TEST_FALSE;
//			}
//		}
//	}
//	
//	public void reset(boolean isFordwardSearch) {
//		index=0;
//		for(int i=0;i<subSets.size();i++) {
//			SubSet aux=subSets.get(i);
//			if(isFordwardSearch)
//				aux.secondTest=SubSet.TEST_NOT_EVALUATED;
//			else
//				aux.firstTest=SubSet.TEST_NOT_EVALUATED;
//		}
//	}
}
