package consensusBN;

public class ListFabric {
	
	private static int maxSize=Integer.MAX_VALUE; // Número máximo de unos que tienen los elementos de la lista.
//	private static HashMap<Integer, int[]> hashMap=new HashMap<Integer, int[]>();
//
//	public static int[] getList(int size) {
//		Integer key=new Integer(size);
//		int[] lista=hashMap.get(key);
//		if(lista==null) {
//			lista=generateList(size);
//			hashMap.put(key, lista);
//		}
//		return lista;
//	}
	
	public static int[] getList(int size) {
		return generateList(size);
	}
	
	private static int[] generateList(int size) {
		int[] lista;
		if(size==0) {
			return new int[1];
		}
		int[] pow2=new int[size];
		pow2[0]=1;
		for(int i=1;i<size;i++) {
			pow2[i]=2*pow2[i-1];
		}
		int tam=(int)Math.pow(2, size);
		int[][] aux=new int[2][tam];
		aux[0][0]=0; // aux[0] almacena los numeros
		aux[1][0]=0; // aux[1] el nº de unos de aux[0]
		int counter=1;
		int index=0;
		while(aux[1][index]<size && aux[1][index]<maxSize) {
			for(int i=0;i<pow2.length;i++) {
				if(pow2[i]>aux[0][index]) {
					aux[0][counter]=aux[0][index]+pow2[i];
					aux[1][counter]=aux[1][index]+1;
					counter++;
				}
			}
			index++;
		}
		lista=new int[counter];
		System.arraycopy(aux[0], 0, lista, 0, counter);
		return lista;
	}

	public static void setMaxSize(int maxParents) {
		ListFabric.maxSize = maxParents;
	}

	public static int getMaxSize() {
		return ListFabric.maxSize;
	}
}
