package edu.cmu.tetrad.graph;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 *
 * @author Pablo Torrijos Arenas
 */
public class EdgeListGraph_n extends EdgeListGraph {
    
    private HashMap<Node,Set<Node>> neighboursMap;
    private HashSet<Node> nodesHash;
        
    public EdgeListGraph_n(){
        super();
        this.neighboursMap = new HashMap();
        this.nodesHash = new HashSet();
    }
    
    public EdgeListGraph_n(Graph graph){
        this();

        if (graph == null) {
            throw new NullPointerException("Graph must not be null.");
        }
        
        this.neighboursMap = new HashMap(graph.getNumNodes());
        this.nodesHash = new HashSet(graph.getNumNodes());
        this.namesHash = new HashMap<>(graph.getNumNodes());

        transferNodesAndEdges(graph);

        // Keep attributes from the original graph
        transferAttributes(graph);

        this.ambiguousTriples = graph.getAmbiguousTriples();
        this.underLineTriples = graph.getUnderLines();
        this.dottedUnderLineTriples = graph.getDottedUnderlines();

        for (Edge edge : graph.getEdges()) {
            if (graph.isHighlighted(edge)) {
                setHighlighted(edge, true);
            }
        }

        for (Node node : this.nodesHash) {
            this.namesHash.put(node.getName(), node);
        }

        this.setPag(graph.isPag());
        this.setCPDAG(graph.isCPDAG());
    }
    
    public EdgeListGraph_n(List<Node> nodes){
        this();

        if (nodes == null) {
            throw new NullPointerException();
        }
        
        this.neighboursMap = new HashMap(nodes.size());
        this.nodesHash = new HashSet(nodes.size());

        for (Node variable : nodes) {
            if (!addNode(variable)) {
                throw new IllegalArgumentException();
            }
        }
    }
    
    /**
     * Determines whether some edge or other exists between two nodes.
     * @param node1
     * @param node2
     * @return 
     */
    @Override
    public boolean isAdjacentTo(Node node1, Node node2) {
        if (node1 == null || node2 == null) {
            return false;
        }

        return neighboursMap.get(node1).contains(node2) && neighboursMap.get(node2).contains(node1);
    }
    
    /**
     * @return the set of nodes adjacent to the given node. If there are
     * multiple edges between X and Y, Y will show up twice in the list of
     * adjacencies for X, for optimality; simply create a list an and array from
     * these to eliminate the duplication.
     */
    @Override
    public List<Node> getAdjacentNodes(Node node) {
        return new ArrayList(neighboursMap.get(node));
    }
    
    /**
     * @return the set of nodes adjacent to the given node. If there are
     * multiple edges between X and Y, Y will show up twice in the list of
     * adjacencies for X, for optimality; simply create a list an and array from
     * these to eliminate the duplication.
     */
    public Set<Node> getAdjacentNodesSet(Node node) {
        return neighboursMap.get(node);
    }
    
    public Set<Node> getCommonAdjacents(Node x, Node y) {
        Set<Node> adj = this.getAdjacentNodesSet(x);
        adj.retainAll(this.getAdjacentNodes(y));
        return adj;
    }

    /**
     * Adds an edge to the graph.
     *
     * @param edge the edge to be added
     * @return true if the edge was added, false if not.
     */
    @Override
    public boolean addEdge(Edge edge) {
        synchronized (this.edgeLists) {
            if (edge == null) {
                throw new NullPointerException();
            }

            this.edgeLists.get(edge.getNode1()).add(edge);
            this.edgeLists.get(edge.getNode2()).add(edge);

            this.edgesSet.add(edge);
            
            // Ahora ambos nodos son vecinos
            this.neighboursMap.get(edge.getNode1()).add(edge.getNode2());
            this.neighboursMap.get(edge.getNode2()).add(edge.getNode1());

            if (Edges.isDirectedEdge(edge)) {
                Node node = Edges.getDirectedEdgeTail(edge);

                if (node.getNodeType() == NodeType.ERROR) {
                    getPcs().firePropertyChange("nodeAdded", null, node);
                }
            }

            this.ancestors = null;
            getPcs().firePropertyChange("edgeAdded", null, edge);
            return true;
        }
    }
    

    /**
     * Removes an edge from the graph. (Note: It is dangerous to make a
     * recursive call to this method (as it stands) from a method containing
     * certain types of iterators. The problem is that if one uses an iterator
     * that iterates over the edges of node A or node B, and tries in the
     * process to remove those edges using this method, a concurrent
     * modification exception will be thrown.)
     *
     * @param edge the edge to remove.
     * @return true if the edge was removed, false if not.
     */
    @Override
    public boolean removeEdge(Edge edge) {
        synchronized (this.edgeLists) {
            if (!this.edgesSet.contains(edge)) {
                return false;
            }

            Set<Edge> edgeList1 = this.edgeLists.get(edge.getNode1());
            Set<Edge> edgeList2 = this.edgeLists.get(edge.getNode2());

            edgeList1 = new HashSet<>(edgeList1);
            edgeList2 = new HashSet<>(edgeList2);
            
            // Si no existe el enlace inverso, dejan de ser vecinos
            if (!edgesSet.contains(edge.reverse())){
                this.neighboursMap.get(edge.getNode1()).remove(edge.getNode2());
                this.neighboursMap.get(edge.getNode2()).remove(edge.getNode1());
            }
            
            this.edgesSet.remove(edge);
            edgeList1.remove(edge);
            edgeList2.remove(edge);

            this.edgeLists.put(edge.getNode1(), edgeList1);
            this.edgeLists.put(edge.getNode2(), edgeList2);

            this.highlightedEdges.remove(edge);
            this.stuffRemovedSinceLastTripleAccess = true;

            this.ancestors = null;
            getPcs().firePropertyChange("edgeRemoved", edge, null);
            return true;
        }
    }
    
    /**
     * Removes the edge connecting the two given nodes.
     * @param node1
     * @param node2
     * @return 
     */
    @Override
    public boolean removeEdge(Node node1, Node node2) {
        List<Edge> edges = getEdges(node1, node2);

        if (edges.size() > 1) {
            throw new IllegalStateException(
                    "There is more than one edge between " + node1 + " and "
                            + node2);
        }

        return removeEdge(edges.get(0));
    }
    
    /**
     * Removes any relevant edge objects found in this collection. G
     *
     * @param edges the collection of edges to remove.
     * @return true if any edges in the collection were removed, false if not.
     */
    @Override
    public boolean removeEdges(Collection<Edge> edges) {
        boolean change = false;

        for (Edge edge : edges) {
            boolean _change = removeEdge(edge);
            change = change || _change;
        }

        return change;
    }
    
    /**
     * @param node1
     * @param node2
     * @return the edges connecting node1 and node2.
     */
    @Override
    public List<Edge> getEdges(Node node1, Node node2) {
        if (!isAdjacentTo(node1, node2)) {
            return new ArrayList<>();
        }
        
        Set<Edge> edges = this.edgeLists.get(node1);
        if (edges == null) {
            return new ArrayList<>();
        }

        List<Edge> _edges = new ArrayList<>();

        edges.stream().filter(edge -> (edge.getDistalNode(node1) == node2)).forEachOrdered(edge -> {
            _edges.add(edge);
        });

        return _edges;
    }
    
     /**
     * @param node
     * @return the list of edges connected to a particular node. No particular
     * ordering of the edges in the list is guaranteed.
     */
    @Override
    public List<Edge> getEdges(Node node) {
        Set<Edge> list = this.edgeLists.get(node);
        if (list == null) {
            return new ArrayList<>();
        }
        return new ArrayList<>(list);
    }
    
    /**
     * @return the list of edges in the graph. No particular ordering of the
     * edges in the list is guaranteed.
     */
    @Override
    public Set<Edge> getEdges() {
        return new HashSet<>(this.edgesSet);
    }
    
    /**
     * Adds a node to the graph. Precondition: The proposed name of the node
     * cannot already be used by any other node in the same graph.
     *
     * @param node the node to be added.
     * @return true if the the node was added, false if not.
     */
    @Override
    public boolean addNode(Node node) {
        if (node == null) {
            throw new NullPointerException();
        }
        
        if (!this.nodesHash.add(node)) {
            return false;
        }

        this.edgeLists.put(node, new HashSet<>());
        this.namesHash.put(node.getName(), node);
        
        this.neighboursMap.put(node, new HashSet<>());

        if (node.getNodeType() != NodeType.ERROR) {
            getPcs().firePropertyChange("nodeAdded", null, node);
        }

        return true;
    }
    
    /**
     * Removes a node from the graph.
     * @param node
     * @return 
     */
    @Override
    public boolean removeNode(Node node) {
        if (!this.nodesHash.remove(node)) {
            return false;
        }

        boolean changed = false;
        Set<Edge> edgeList1 = this.edgeLists.get(node);    //list of edges connected to that node
        edgesSet.removeAll(edgeList1);

        for (Iterator<Edge> i = edgeList1.iterator(); i.hasNext(); ) {
            Edge edge = (i.next());
            Node node2 = edge.getDistalNode(node);

            if (node2 != node) {
                Set<Edge> edgeList2 = this.edgeLists.get(node2);
                edgeList2.remove(edge);
                this.edgesSet.remove(edge);
                changed = true;
            }

            i.remove();
            getPcs().firePropertyChange("edgeRemoved", edge, null);
        }

        this.edgeLists.remove(node);
        this.namesHash.remove(node.getName());
        this.neighboursMap.remove(node);
        this.stuffRemovedSinceLastTripleAccess = true;

        getPcs().firePropertyChange("nodeRemoved", node, null);
        return changed;
    }
    
    /**
     * Determines whether the graph contains a particular node.
     * @param node
     * @return 
     */
    @Override
    public boolean containsNode(Node node) {
        return this.nodesHash.contains(node);
    }
    
    /**
     * @return a matrix of endpoints for the nodes in this graph, with nodes in
     * the same order as getNodes().
     */
    @Override
    public Endpoint[][] getEndpointMatrix() {
        Node[] arrNodes = (Node[])this.nodesHash.toArray();
        int size = arrNodes.length;
        Endpoint[][] endpoints = new Endpoint[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i == j) {
                    continue;
                }
                endpoints[i][j] = getEndpoint(arrNodes[i], arrNodes[j]);
            }
        }

        return endpoints;
    }
    
        
    /**
     * Resets the graph so that it is fully connects it using #-# edges, where #
     * is the given endpoint.
     * @param endpoint
     */
    @Override
    public void fullyConnect(Endpoint endpoint) {
        this.edgesSet.clear();
        this.edgeLists.clear();

        for (Node node : this.nodesHash) {
            this.edgeLists.put(node, new HashSet<>());
        }
        
        Node[] arrNodes = (Node[])this.nodesHash.toArray();
        int size = arrNodes.length;

        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                Edge edge = new Edge(arrNodes[i], arrNodes[j], endpoint, endpoint);
                addEdge(edge);
            }
        }
    }
    
    /**
     * @return the number of nodes in the graph.
     */
    @Override
    public int getNumNodes() {
        return this.nodesHash.size();
    }
    
    @Override
    public List<Node> getNodes() {
        return new ArrayList<>(this.nodesHash);
    }
    
    @Override
    public void setNodes(List<Node> nodes) {
        if (nodes.size() != this.nodesHash.size()) {
            throw new IllegalArgumentException("Sorry, there is a mismatch in the number of variables "
                    + "you are trying to set.");
        }

        this.nodesHash.clear();
        this.nodesHash.addAll(nodes);
    }
    
    public void setNodes(HashSet<Node> nodesHash) {
        if (nodesHash.size() != this.nodesHash.size()) {
            throw new IllegalArgumentException("Sorry, there is a mismatch in the number of variables "
                    + "you are trying to set.");
        }

        this.nodesHash.clear();
        this.nodesHash.addAll(nodes);
    }

    /**
     * Removes all nodes (and therefore all edges) from the graph.
     */
    @Override
    public void clear() {
        Iterator<Edge> it = getEdges().iterator();

        while (it.hasNext()) {
            Edge edge = it.next();
            it.remove();
            getPcs().firePropertyChange("edgeRemoved", edge, null);
        }

        Iterator<Node> it2 = this.nodesHash.iterator();

        while (it2.hasNext()) {
            Node node = it2.next();
            it2.remove();
            this.namesHash.remove(node.getName());
            getPcs().firePropertyChange("nodeRemoved", node, null);
        }

        this.edgeLists.clear();
    }

    
    @Override
    public int hashCode() {
        int hashCode = 0;

        for (Edge edge : getEdges()) {
            hashCode += edge.hashCode();
        }

        return (new HashSet<>(this.nodesHash)).hashCode() + hashCode;
    }

    /**
     * @return true iff the given object is a graph that is equal to this graph,
     * in the sense that it contains the same nodes and the edges are
     * isomorphic.
     */
    @Override
    public boolean equals(Object o) {
        if (o == null) {
            return false;
        }

        if (o instanceof EdgeListGraph_n) {
            EdgeListGraph_n _o = (EdgeListGraph_n) o;
            boolean nodesEqual = new HashSet<>(_o.nodesHash).equals(new HashSet<>(this.nodesHash));
            boolean edgesEqual = new HashSet<>(_o.edgesSet).equals(new HashSet<>(this.edgesSet));
            return (nodesEqual && edgesEqual);
        } else {
            Graph graph = (Graph) o;
            return new HashSet<>(graph.getNodeNames()).equals(new HashSet<>(getNodeNames()))
                    && new HashSet<>(graph.getEdges()).equals(new HashSet<>(getEdges()));

        }
    }
    
}
