/*
 * This file is licensed to You under the "Simplified BSD License".
 * You may not use this software except in compliance with the License. 
 * You may obtain a copy of the License at
 *
 * http://www.opensource.org/licenses/bsd-license.php
 * 
 * See the COPYRIGHT file distributed with this work for information
 * regarding copyright ownership.
 */
package consensusBN;

import java.util.ArrayList;

import edu.cmu.tetrad.graph.Dag_n;


/**
 * The HierarchicalAgglomerativeClusterer creates a hierarchical agglomerative clustering.
 * 
 *
 * 
 * @author Matthias.Hauswirth@usi.ch
 */

public final class HierarchicalAgglomerativeClustererBNs {

    ArrayList<Dag_n> setOfBNs = null;
    Dag_n[][] clustersBN = null;
    boolean [][][] clustersIndexes = null;
    double averageNEdges = 0.00;
    double maxComplexityCluster = Double.MAX_VALUE;
    int maxSize = 0;
    int maxLevel = 0;
    PairWiseConsensusBES[][] initialpairwisedistance = null;
    int numberOfInsertedEdges = 0;
    
    int nDags = 0;
    boolean[] indexUsed = null;
    int[] clusterCardinalities = null;
    double[] clusterComplexity = null;
    
    PairWiseConsensusBES[][] dissimilarityMatrix = null;
	private int[][] clustersInsertedEdges;
   
    
    public HierarchicalAgglomerativeClustererBNs(ArrayList<Dag_n> setOfBNs, int maxSize) {
        this.setOfBNs = setOfBNs;
        this.clustersBN = new Dag_n[setOfBNs.size()][setOfBNs.size()];
        this.clustersIndexes = new boolean[setOfBNs.size()][setOfBNs.size()][setOfBNs.size()];
        for(int i=  0 ; i< setOfBNs.size(); i++){
        	this.averageNEdges+=(double) setOfBNs.get(i).getNumEdges();
        }
        this.maxSize = maxSize;
        this.maxLevel = this.getSetOfBNs().size()-1;
        this.nDags = this.getSetOfBNs().size();
        this.indexUsed = new boolean[nDags];
        this.clusterCardinalities = new int[nDags];
        this.clusterComplexity = new double[nDags];
        this.clustersInsertedEdges = new int[setOfBNs.size()][setOfBNs.size()];
    }
    
    public HierarchicalAgglomerativeClustererBNs(ArrayList<Dag_n> setOfBNs, double maxComplexity) {
        this.setOfBNs = setOfBNs;
        this.clustersBN = new Dag_n[setOfBNs.size()][setOfBNs.size()];
        this.clustersIndexes = new boolean[setOfBNs.size()][setOfBNs.size()][setOfBNs.size()];
        for(int i=  0 ; i< setOfBNs.size(); i++){
        	this.averageNEdges+=(double) setOfBNs.get(i).getNumEdges();
        }
        this.maxLevel = this.getSetOfBNs().size()-1;
        this.averageNEdges/=setOfBNs.size();
        this.maxComplexityCluster = (1.0+maxComplexity);
        this.nDags = this.getSetOfBNs().size();
        this.indexUsed = new boolean[nDags];
        this.clusterCardinalities = new int[nDags];
        this.clusterComplexity = new double[nDags];
        this.clustersInsertedEdges = new int[setOfBNs.size()][setOfBNs.size()];
    }
    
//    public HierarchicalAgglomerativeClustererBNs(ArrayList<BayesIm> setOfBNs, int sampleSize) {
//        this.setOfBNs = setOfBNs;
//        this.clustersBN = new BayesIm[setOfBNs.size()][setOfBNs.size()];
//        this.clustersIndexes = new boolean[setOfBNs.size()][setOfBNs.size()][setOfBNs.size()];
//        this.sampleSize = sampleSize;
//        this.method = 1;
//    }
    
    public void setDags(ArrayList<Dag_n> setOfBNs) {
  
    	this.setOfBNs = setOfBNs;
    }
    
    public ArrayList<Dag_n> getSetOfBNs() {
        return this.setOfBNs;
    }
    
    public int cluster() {
    	 
       // Perform nObservations-1 agglomerations
        
        for (int i = 0; i<nDags; i++) {
            indexUsed[i] = true;
            clusterCardinalities[i] = 1;
            clusterComplexity[i] = setOfBNs.get(i).getNumEdges()/this.averageNEdges;
            clustersBN[i][0] = setOfBNs.get(i);
            clustersIndexes[i][i][0] = true;
        }
        
        dissimilarityMatrix = computeDissimilarityMatrix();
       
        for (int a = 1; a<nDags; a++) {
            // Determine the two most similar clusters, i and j (such that i<j)
            final Pair pair = findMostSimilarClusters();
            final int i = pair.getSmaller();
            final int j = pair.getLarger();
            if(i==j){
            	this.maxLevel = a-1;
            	return a-1;
            }
            
            PairWiseConsensusBES d = dissimilarityMatrix[i][j];
           
//            System.out.println("Agglomeration #"+a+
//                    ": merging clusters "+i+
//                    " (cardinality "+(clusterCardinalities[i])+") and "+j+
//                    " (cardinality "+(clusterCardinalities[j])+") with dissimilarity "+d.getHammingDistance());
//            
            // update clustering
            merge(i, j, d , a);
            clusterCardinalities[i] = clusterCardinalities[i]+clusterCardinalities[j];
            clusterCardinalities[j] = 0;
            clusterComplexity[i] = clustersBN[i][a].getNumEdges()/averageNEdges;
            clusterComplexity[j] = 0.0;
//            System.out.println(" Indices de los Cluster a nivel: "+a);
//            for(int c = 0 ; c< setOfBNs.size();c++){
//            	System.out.print(" Indices cluster: "+c+ " ( ");
//            	for(int g = 0; g<setOfBNs.size(); g++){
//            		if(this.clustersIndexes[c][g][a]) System.out.print(g+", ");
//            	}
//            	System.out.println(") ");
//            }
            
            // erase cluster j
            indexUsed[j] = false;
            for (int k = 0; k<nDags; k++) {
                dissimilarityMatrix[j][k] = null;
                dissimilarityMatrix[k][j] = null;
            }
            
            // cluster i becomes new cluster
            // (by agglomerating former clusters i and j)
            // update dissimilarityMatrix[i][*] and dissimilarityMatrix[*][i]
            // if(clusterCardinalities[i] > maxSize) return a;
            if((clusterComplexity[i]<= maxComplexityCluster&&maxSize==0)||(maxSize>0 && clusterCardinalities[i]<=maxSize)){
            	for (int k = 0; k<nDags; k++) {
            		if ((k!=i)&&(k!=j)&&indexUsed[k]) {
            			PairWiseConsensusBES dissimilarity = computeDissimilarity(i,k,a);
            			dissimilarityMatrix[i][k] = dissimilarity;
            			dissimilarityMatrix[k][i] = dissimilarity;
            		}
            	}
            }else indexUsed[i] = false;
        }
        return nDags;
    }
    
    public Dag_n computeConsensusDag(int level){

    	if (level <= this.maxLevel && level > 0 && this.initialpairwisedistance!=null){
    		int[] distance = new int[this.setOfBNs.size()];
    		int[] index = new int[this.setOfBNs.size()];
    		for(int i = 0; i< this.setOfBNs.size(); i++){
    			distance[i] = 0;
    			index[i] = -1;
    		}

    		for(int cluster = 0 ; cluster < this.setOfBNs.size();cluster++ ){
    			int bestDistance = Integer.MAX_VALUE;
    			for(int j = 0; j< this.setOfBNs.size(); j++){
    				for(int k = 0; k< this.setOfBNs.size(); k++){
    					if(this.clustersIndexes[cluster][j][level]&&this.clustersIndexes[cluster][k][level]&&(j!=k)){
        					distance[j]+=this.initialpairwisedistance[j][k].getHammingDistance();//getNumberOfInsertedEdges();
    					}
    				}
    				if(clustersIndexes[cluster][j][level]&&distance[j]<bestDistance){
    					bestDistance = distance[j];
    					index[cluster] = j;
    				}
    			}
    		}
    		ArrayList<Dag_n> setOfDags = new ArrayList<Dag_n>();
    		for (int cluster = 0; cluster < this.setOfBNs.size(); cluster++){
    			if(index[cluster]!=-1) setOfDags.add(this.setOfBNs.get(index[cluster]));
    		}
    		ConsensusBES fus = new ConsensusBES(setOfDags);
    		fus.fusion();
    		this.numberOfInsertedEdges = fus.getNumberOfInsertedEdges();
    		return fus.getFusion();
    	}else{
    		if(level == this.maxLevel+1){
    			for(int cluster =0 ; cluster < this.setOfBNs.size(); cluster++){
    				for(int index =0; index < this.setOfBNs.size(); index++){
    					if(this.clustersIndexes[cluster][index][level-1]){
    						return this.clustersBN[cluster][level-1];
    					}
    				}
    			}
    		}
    		else return null;
    	}
    	return null;
    }
    
    
    public int getInsertedEdges(int level){
    	int insertedEdges = 0;
    	for(int cluster=0; cluster< this.setOfBNs.size(); cluster++){
    		if(clustersBN[cluster][level]!= null){
    			insertedEdges+= clustersInsertedEdges[cluster][level];
    		}
    	}
    	return insertedEdges;
    }
    
    
    public ArrayList<Dag_n> getClustersOutput(int level){
    	
    	ArrayList<Dag_n> clusters = new ArrayList<Dag_n>();
    	for(int cluster = 0; cluster< this.setOfBNs.size(); cluster++){
    		if(this.clustersBN[cluster][level]!= null)
    			clusters.add(this.clustersBN[cluster][level]);
    	}
    	
    	return clusters;
    	
    }
    
    public int getNumberOfInsertedEdges(){
		return this.numberOfInsertedEdges;
	}
    
    private void merge(int i, int j, PairWiseConsensusBES d, int level) {
		
    	for(int u = 0; u < this.getSetOfBNs().size(); u++){
    		for(int v = 0; v < this.getSetOfBNs().size(); v++){
    			this.clustersIndexes[u][v][level] = this.clustersIndexes[u][v][level-1];
    		}
    	}
    	
    	for(int ant = 0 ; ant < this.getSetOfBNs().size(); ant++){
    		if(this.clustersIndexes[j][ant][level]){
    			this.clustersIndexes[i][ant][level] = true;
    		}
    	}
    	
    	for(int ant = 0 ; ant <this.getSetOfBNs().size(); ant++){
//    		this.clustersIndexes[ant][j][level] = false;
    		this.clustersIndexes[j][ant][level] = false;
    	}
  
    	
    	
    	
    	for (int c = 0; c < this.getSetOfBNs().size(); c++)
    		if(c == i){
    			this.clustersBN[c][level] = d.getDagFusion();
    			this.clustersInsertedEdges[c][level] = d.getNumberOfInsertedEdges();
    		}else{
    			this.clustersBN[c][level] = this.clustersBN[c][level-1];
    			this.clustersInsertedEdges[c][level] = this.clustersInsertedEdges[c][level-1];
    		}
    	this.clustersBN[j][level]= null;
	}

	
	private PairWiseConsensusBES[][] computeDissimilarityMatrix() {
        final PairWiseConsensusBES[][] dissimilarityMatrix = new PairWiseConsensusBES[this.getSetOfBNs().size()][this.getSetOfBNs().size()];
        this.initialpairwisedistance = new PairWiseConsensusBES[this.getSetOfBNs().size()][this.getSetOfBNs().size()];
        // fill diagonal
        for (int o = 0; o<dissimilarityMatrix.length; o++) {
            dissimilarityMatrix[o][o] = null;
            this.initialpairwisedistance[o][o] = null;
        }
        // fill rest (only compute half, then mirror accross diagonal, assuming
        // a symmetric dissimilarity measure)
        for (int o1 = 0; o1<dissimilarityMatrix.length; o1++) {
            for (int o2 = 0; o2<o1; o2++) {
            	PairWiseConsensusBES dissimilarity = computeDissimilarity(o1, o2, 0);
                dissimilarityMatrix[o1][o2] = dissimilarity;
                dissimilarityMatrix[o2][o1] = dissimilarity;
                this.initialpairwisedistance[o1][o2] = dissimilarity;
                this.initialpairwisedistance[o2][o1] = dissimilarity;
            }
        }
        return dissimilarityMatrix;
    }

    private PairWiseConsensusBES computeDissimilarity(int o1, int o2, int level) {
    	
    	
    	if (this.maxSize > 0 && (this.maxSize >= (this.clusterCardinalities[o1] + this.clusterCardinalities[o2]))|| level == 0){
    		PairWiseConsensusBES pairBNs = new PairWiseConsensusBES(this.clustersBN[o1][level],this.clustersBN[o2][level]);
    		PairWiseConsensusBES pairDag= (PairWiseConsensusBES) pairBNs;
    		pairDag.getFusion();
    		return pairBNs;
    	}else if(this.maxSize == 0){
    			PairWiseConsensusBES pairBNs = new PairWiseConsensusBES(this.clustersBN[o1][level],this.clustersBN[o2][level]);
    			PairWiseConsensusBES pairDag= (PairWiseConsensusBES) pairBNs;
    			pairDag.getFusion();
    			if((pairDag.getDagFusion().getNumEdges())/this.averageNEdges <= this.maxComplexityCluster|| level == 0)
    				return pairBNs;	
    			else return null;
    	}else return null;
	}

	private  Pair findMostSimilarClusters() {
		
        final Pair mostSimilarPair = new Pair();
        double smallestDissimilarity = Double.POSITIVE_INFINITY;
        double smallnEdgesUnion = Double.POSITIVE_INFINITY;
        
        for (int cluster = 0; cluster<dissimilarityMatrix.length; cluster++) {
            if (indexUsed[cluster]) {
                for (int neighbor = 0; neighbor<dissimilarityMatrix.length; neighbor++) {
                	if((cluster!=neighbor)&&(indexUsed[neighbor])){
                		PairWiseConsensusBES inCluster = dissimilarityMatrix[cluster][neighbor];
                		if(inCluster!= null){
                			double complexity = 0.0;
                			complexity = (float) inCluster.getHammingDistance();//getNumberOfInsertedEdges();
                			if (indexUsed[neighbor]&&complexity<smallestDissimilarity&&cluster!=neighbor) {
                				smallestDissimilarity = complexity;
                				smallnEdgesUnion = (float) inCluster.getNumberOfUnionEdges();
                				mostSimilarPair.set(cluster, neighbor);
                			}else if (indexUsed[neighbor]&&complexity==smallestDissimilarity&&cluster!=neighbor) {
                				if(smallnEdgesUnion > (float) inCluster.getNumberOfUnionEdges()){
                					smallnEdgesUnion = (float) inCluster.getNumberOfUnionEdges();
                					mostSimilarPair.set(cluster, neighbor);
                				}  
                			}
                		}
                	}
                }
            }
        }
        return mostSimilarPair;
    }
	
	


    private static final class Pair {

        private int cluster1;
        private int cluster2;


        public final void set(final int cluster1, final int cluster2) {
            this.cluster1 = cluster1;
            this.cluster2 = cluster2;
        }

        public final int getLarger() {
            return Math.max(cluster1, cluster2);
        }

        public final int getSmaller() {
            return Math.min(cluster1, cluster2);
        }

    }

}
