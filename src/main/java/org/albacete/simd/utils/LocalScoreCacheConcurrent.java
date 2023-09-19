package org.albacete.simd.utils;

import edu.cmu.tetrad.graph.Node;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Objects;


public class LocalScoreCacheConcurrent {
    private ConcurrentHashMap<DualKey<Integer, Set<Node>>, Double> map = new ConcurrentHashMap<>();

    public LocalScoreCacheConcurrent() {
    }

    public void add(int variable, Set<Node> parents, double score) {
        DualKey<Integer, Set<Node>> key = new DualKey<>(variable, parents);

        this.map.put(key, score);
    }

    public double get(int variable, Set<Node> parents) {
        DualKey<Integer, Set<Node>> key = new DualKey<>(variable, parents);

        Double _score = this.map.get(key);
        return _score == null ? Double.NaN : _score;
    }

    public void clear() {
        this.map.clear();
    }

    @Override
    public String toString() {
        return "LocalScoreCacheConcurrent{" +
                "map=" + map +
                '}';
    }

    private class DualKey<K1,K2> {
        private final K1 key1;
        private final K2 key2;
        
        private int hash;
    
        public DualKey(K1 key1, K2 key2){
            this.key1 = key1;
            this.key2 = key2;
        }
    
        public K1 getKey1(){
            return key1;
        }
    
        public K2 getKey2(){
            return key2;
        }
    
        @Override
        public boolean equals(Object other){
            if (other instanceof DualKey<?,?>){
                DualKey<?,?> obj = (DualKey<?,?>)other;
                if ( obj.getKey1().equals(this.key1) && obj.getKey2().equals(this.key2)){
                    return true;
                }
            }
            return false;
        }
    
        @Override
        public String toString() {
            return "(" + key1.toString() + ", " + key2.toString() + ")";
        }
    
        @Override
        public int hashCode() {
            if (hash == 0){
                hash = Objects.hash(key1, key2);
            }
            //return super.hashCode();
            return hash;
        }
    }
}
