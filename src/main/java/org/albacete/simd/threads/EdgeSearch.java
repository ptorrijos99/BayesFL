package org.albacete.simd.threads;

import edu.cmu.tetrad.graph.Edge;
import edu.cmu.tetrad.graph.Node;

import java.util.HashSet;

public class EdgeSearch implements Comparable<EdgeSearch> {

    public double score;
    public HashSet<Node> hSubset;
    public Edge edge;

    public EdgeSearch(double score, HashSet<Node> hSubSet, Edge edge) {
        this.score = score;
        this.hSubset = hSubSet;
        this.edge = edge;
    }

    @Override
    public int compareTo(EdgeSearch o) {
        return Double.compare(this.score, (o).score);
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof EdgeSearch obj) {
            return obj.edge.equals(this.edge);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return this.edge.hashCode();
    }

    public double getScore() {
        return this.score;
    }

    public Edge getEdge() {
        return this.edge;
    }
    
    @Override
    public String toString() {
        return score + " =>   " + edge + "  " + hSubset;
    }
}
