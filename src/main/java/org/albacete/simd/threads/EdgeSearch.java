package org.albacete.simd.threads;

import consensusBN.SubSet;
import edu.cmu.tetrad.graph.Edge;
import java.util.Objects;

public class EdgeSearch implements Comparable<EdgeSearch> {

    public double score;
    public SubSet hSubset;
    public Edge edge;

    public EdgeSearch(double score, SubSet hSubSet, Edge edge) {
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
        if (other instanceof EdgeSearch) {
            EdgeSearch obj = (EdgeSearch) other;
            if (obj.edge.equals(this.edge)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.edge);
    }

    public double getScore() {
        return this.score;
    }

    public SubSet gethSubset() {
        return this.hSubset;
    }

    public Edge getEdge() {
        return this.edge;
    }
}
