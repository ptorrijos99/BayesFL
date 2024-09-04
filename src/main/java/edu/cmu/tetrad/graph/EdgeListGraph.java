///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015, 2022 by Peter Spirtes, Richard        //
// Scheines, Joseph Ramsey, and Clark Glymour.                               //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////
package edu.cmu.tetrad.graph;

import edu.cmu.tetrad.data.DiscreteVariable;

import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;

import static edu.cmu.tetrad.graph.Edges.directedEdge;

/**
 * <p>
 * Stores a graph a list of lists of edges adjacent to each node in the graph,
 * with an additional list storing all of the edges in the graph. The edges are
 * of the form N1 *-# N2. Multiple edges may be added per node pair to this
 * graph, with the caveat that all edges of the form N1 *-# N2 will be
 * considered equal. For example, if the edge X --&gt; Y is added to the graph,
 * another edge X --&gt; Y may not be added, although an edge Y --&gt; X may be added.
 * Edges from nodes to themselves may also be added.&gt; 0
 *
 * @author Joseph Ramsey
 * @author Erin Korber additions summer 2004
 * @see edu.cmu.tetrad.graph.Endpoint
 */
public class EdgeListGraph implements Graph {

    static final long serialVersionUID = 23L;

    /**
     * A list of the nodes in the graph, in the order in which they were added.
     *
     * @serial
     */
    protected LinkedHashSet<Node> nodes;
    
    private HashMap<Node,Set<Node>> neighboursMap;

    /**
     * The edges in the graph.
     *
     * @serial
     */
    final Map<Edge, Edge> edgesSet;

    /**
     * Map from each node to the List of edges connected to that node.
     *
     * @serial
     */
    final Map<Node, Set<Edge>> edgeLists;
    private final Map<String, Object> attributes = new HashMap<>();
    /**
     * Fires property change events.
     */
    protected transient PropertyChangeSupport pcs;
    /**
     * Set of ambiguous triples. Note the name can't be changed due to
     * serialization.
     */
    protected Set<Triple> ambiguousTriples = Collections.newSetFromMap(new HashMap<>());
    /**
     * Determines whether one node is an ancestor of another.
     */
    protected Map<Node, Set<Node>> ancestors;
    /**
     * @serial
     */
    Set<Triple> underLineTriples = Collections.newSetFromMap(new HashMap<>());
    /**
     * @serial
     */
    Set<Triple> dottedUnderLineTriples = Collections.newSetFromMap(new HashMap<>());
    /**
     * True iff nodes were removed since the last call to an accessor for
     * ambiguous, underline, or dotted underline triples. If there are triples
     * in the lists involving removed nodes, these need to be removed from the
     * lists first, so as not to cause confusion.
     */
    boolean stuffRemovedSinceLastTripleAccess;
    /**
     * The set of highlighted edges.
     */
    Set<Edge> highlightedEdges = new HashSet<>();

    /**
     * A hash from node names to nodes;
     */
    Map<String, Node> namesHash;
    private boolean cpdag;

    //==============================CONSTUCTORS===========================//
    private boolean pag;

    /**
     * Constructs a new (empty) EdgeListGraph_n.
     */
    public EdgeListGraph() {
        this.edgeLists = new HashMap<>();
        this.neighboursMap = new HashMap();
        this.nodes = new LinkedHashSet<>();
        this.edgesSet = new HashMap<>();
        this.namesHash = new HashMap<>();
    }

    /**
     * Constructs a EdgeListGraph_n using the nodes and edges of the given graph.
     * If this cannot be accomplished successfully, an exception is thrown. Note
     * that any graph constraints from the given graph are forgotten in the new
     * graph.
     *
     * @param graph the graph from which nodes and edges are is to be extracted.
     * @throws IllegalArgumentException if a duplicate edge is added.
     */
    public EdgeListGraph(Graph graph) throws IllegalArgumentException {
        this();

        if (graph == null) {
            throw new NullPointerException("Graph must not be null.");
        }

        this.neighboursMap = new HashMap(graph.getNumNodes());
        this.nodes = new LinkedHashSet(graph.getNumNodes());
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

        for (Node node : this.nodes) {
            this.namesHash.put(node.getName(), node);
        }

        this.pag = graph.isPag();
        this.cpdag = graph.isCPDAG();
    }

    /**
     * Constructs a new graph, with no edges, using the the given variable
     * names.
     */
    public EdgeListGraph(List<Node> nodes) {
        this();

        if (nodes == null) {
            throw new NullPointerException();
        }

        this.neighboursMap = new HashMap(nodes.size());
        this.nodes = new LinkedHashSet(nodes.size());
        
        for (Node variable : nodes) {
            if (!addNode(variable)) {
                throw new IllegalArgumentException();
            }
        }

        for (Node node : nodes) {
            this.namesHash.put(node.getName(), node);
        }
    }

    /**
     * Generates a simple exemplar of this class to test serialization.
     */
    public static EdgeListGraph serializableInstance() {
        return new EdgeListGraph();
    }

    private static boolean visibleEdgeHelper(Node A, Node B, Graph graph) {
        if (A.getNodeType() != NodeType.MEASURED) {
            return false;
        }
        if (B.getNodeType() != NodeType.MEASURED) {
            return false;
        }

        LinkedList<Node> path = new LinkedList<>();
        path.add(A);

        for (Node C : graph.getNodesInTo(A, Endpoint.ARROW)) {
            if (graph.isParentOf(C, A)) {
                return true;
            }

            if (EdgeListGraph.visibleEdgeHelperVisit(graph, C, A, B, path)) {
                return true;
            }
        }

        return false;
    }

    private static boolean visibleEdgeHelperVisit(Graph graph, Node c, Node a, Node b,
                                                  LinkedList<Node> path) {
        if (path.contains(a)) {
            return false;
        }

        path.addLast(a);

        if (a.equals(b)) {
            return true;
        }

        for (Node D : graph.getNodesInTo(a, Endpoint.ARROW)) {
            if (graph.isParentOf(D, c)) {
                return true;
            }

            if (a.getNodeType() == NodeType.MEASURED) {
                if (!graph.isDefCollider(D, c, a)) {
                    continue;
                }
            }

            if (graph.isDefCollider(D, c, a)) {
                if (!graph.isParentOf(c, b)) {
                    continue;
                }
            }

            if (EdgeListGraph.visibleEdgeHelperVisit(graph, D, c, b, path)) {
                return true;
            }
        }

        path.removeLast();
        return false;
    }

    /**
     * Adds a directed edge to the graph from node A to node B.
     *
     * @param node1 the "from" node.
     * @param node2 the "to" node.
     */
    @Override
    public boolean addDirectedEdge(Node node1, Node node2) {
        return addEdge(directedEdge(node1, node2));
    }

    /**
     * Adds an undirected edge to the graph from node A to node B.
     *
     * @param node1 the "from" node.
     * @param node2 the "to" node.
     */
    @Override
    public boolean addUndirectedEdge(Node node1, Node node2) {
        return addEdge(Edges.undirectedEdge(node1, node2));
    }

    /**
     * Adds a nondirected edge to the graph from node A to node B.
     *
     * @param node1 the "from" node.
     * @param node2 the "to" node.
     */
    @Override
    public boolean addNondirectedEdge(Node node1, Node node2) {
        return addEdge(Edges.nondirectedEdge(node1, node2));
    }

    /**
     * Adds a partially oriented edge to the graph from node A to node B.
     *
     * @param node1 the "from" node.
     * @param node2 the "to" node.
     */
    @Override
    public boolean addPartiallyOrientedEdge(Node node1, Node node2) {
        return addEdge(Edges.partiallyOrientedEdge(node1, node2));
    }

    /**
     * Adds a bidirected edge to the graph from node A to node B.
     *
     * @param node1 the "from" node.
     * @param node2 the "to" node.
     */
    @Override
    public boolean addBidirectedEdge(Node node1, Node node2) {
        return addEdge(Edges.bidirectedEdge(node1, node2));
    }

    @Override
    public boolean existsDirectedCycle() {
        for (Node node : getNodes()) {
            if (existsDirectedPathFromTo(node, node)) return true;
        }
        return false;
    }

    @Override
    public boolean isDirectedFromTo(Node node1, Node node2) {
        List<Edge> edges = getEdges(node1, node2);
        if (edges.size() != 1) {
            return false;
        }
        Edge edge = edges.get(0);
        return edge.pointsTowards(node2);
    }

    @Override
    public boolean isUndirectedFromTo(Node node1, Node node2) {
        Edge edge = getEdge(node1, node2);
        return edge != null && edge.getEndpoint1() == Endpoint.TAIL && edge.getEndpoint2() == Endpoint.TAIL;
    }

    /**
     * added by ekorber, 2004/06/11
     *
     * @return true if the given edge is definitely visible (Jiji, pg 25)
     * @throws IllegalArgumentException if the given edge is not a directed edge
     *                                  in the graph
     */
    @Override
    public boolean defVisible(Edge edge) {
        if (containsEdge(edge)) {

            Node A = Edges.getDirectedEdgeTail(edge);
            Node B = Edges.getDirectedEdgeHead(edge);

            for (Node C : getAdjacentNodes(A)) {
                if (!C.equals(B) && !isAdjacentTo(C, B)) {
                    Edge e = getEdge(C, A);

                    if (getProximalEndpoint(A,e) == Endpoint.ARROW) {
                        return true;
                    }
                }
            }

            return EdgeListGraph.visibleEdgeHelper(A, B, this);
        } else {
            throw new IllegalArgumentException(
                    "Given edge is not in the graph.");
        }
    }

    /**
     * IllegalArgument exception raised (by isDirectedFromTo(getEndpoint) or by
     * getEdge) if there are multiple edges between any of the node pairs.
     */
    @Override
    public boolean isDefNoncollider(Node node1, Node node2, Node node3) {
        List<Edge> edges = getEdges(node2);
        boolean circle12 = false;
        boolean circle32 = false;

        for (Edge edge : edges) {
            boolean _node1 = getDistalNode(node2,edge).equals(node1);
            boolean _node3 = getDistalNode(node2,edge).equals(node3);

            if (_node1 && edge.pointsTowards(node1)) {
                return true;
            }
            if (_node3 && edge.pointsTowards(node3)) {
                return true;
            }

            if (_node1 && getProximalEndpoint(node2,edge) == Endpoint.CIRCLE) {
                circle12 = true;
            }
            if (_node3 && getProximalEndpoint(node2,edge) == Endpoint.CIRCLE) {
                circle32 = true;
            }
            if (circle12 && circle32 && !isAdjacentTo(node1, node2)) {
                return true;
            }
        }

        return false;
    }

    @Override
    public boolean isDefCollider(Node node1, Node node2, Node node3) {
        Edge edge1 = getEdge(node1, node2);
        Edge edge2 = getEdge(node2, node3);

        if (edge1 == null || edge2 == null) return false;

        return getProximalEndpoint(node2,edge1) == Endpoint.ARROW && getProximalEndpoint(node2,edge2) == Endpoint.ARROW;

    }

    /**
     * @return true iff there is a directed path from node1 to node2. a
     */
    @Override
    public boolean existsDirectedPathFromTo(Node node1, Node node2) {
        Queue<Node> Q = new LinkedList<>();
        Set<Node> V = new HashSet<>();

        for (Node c : getChildren(node1)) {
            Q.add(c);
            V.add(c);
        }

        while (!Q.isEmpty()) {
            Node t = Q.remove();

            for (Node c : getChildren(t)) {
                if (c == node2) return true;

                if (!V.contains(c)) {
                    V.add(c);
                    Q.offer(c);
                }
            }
        }

        return false;
    }

    @Override
    public List<Node> findCycle() {
        for (Node a : getNodes()) {
            List<Node> path = findDirectedPath(a, a);
            if (!path.isEmpty()) {
                return path;
            }
        }

        return new LinkedList<>();
    }

    public List<Node> findDirectedPath(Node from, Node to) {
        LinkedList<Node> path = new LinkedList<>();

        for (Node d : getChildren(from)) {
            if (findDirectedPathVisit(from, d, to, path)) {
                path.addFirst(from);
                return path;
            }
        }

        return path;
    }

    private boolean findDirectedPathVisit(Node prev, Node next, Node to, LinkedList<Node> path) {
        if (path.contains(next)) return false;
        path.addLast(next);
        if (!getEdge(prev, next).pointsTowards(next)) throw new IllegalArgumentException();
        if (next == to) return true;

        for (Node d : getChildren(next)) {
            if (findDirectedPathVisit(next, d, to, path)) return true;
        }

        path.removeLast();
        return false;
    }

    @Override
    public boolean existsUndirectedPathFromTo(Node node1, Node node2) {
        return existsUndirectedPathVisit(node1, node2, new HashSet<>());
    }

    @Override
    public boolean existsSemiDirectedPathFromTo(Node node1, Set<Node> nodes) {
        return existsSemiDirectedPathVisit(node1, nodes,
                new LinkedList<>());
    }

    /**
     * Determines whether a trek exists between two nodes in the graph. A trek
     * exists if there is a directed path between the two nodes or else, for
     * some third node in the graph, there is a path to each of the two nodes in
     * question.
     */
    @Override
    public boolean existsTrek(Node node1, Node node2) {
        for (Node node : getNodes()) {
            if (isAncestorOf((node), node1) && isAncestorOf((node), node2)) {
                return true;
            }
        }

        return false;
    }

    /**
     * @return the list of children for a node.
     */
    @Override
    public List<Node> getChildren(Node node) {
        List<Node> children = new ArrayList<>();

        for (Edge edge : getEdges(node)) {
            if (Edges.isDirectedEdge(edge)) {
                Node sub = traverseDirected(node, edge);

                if (sub != null) {
                    children.add(sub);
                }
            }
        }

        return children;
    }

    @Override
    public int getConnectivity() {
        int connectivity = 0;

        List<Node> nodes = getNodes();

        for (Node node : nodes) {
            int n = getNumEdges(node);
            if (n > connectivity) {
                connectivity = n;
            }
        }

        return connectivity;
    }

    @Override
    public List<Node> getDescendants(List<Node> nodes) {
        Set<Node> descendants = new HashSet<>();

        for (Node node : nodes) {
            collectDescendantsVisit(node, descendants);
        }

        return new LinkedList<>(descendants);
    }

    /**
     * @return the edge connecting node1 and node2, provided a unique such edge
     * exists.
     */
    @Override
    public Edge getEdge(Node node1, Node node2) {
        Set<Edge> edges = this.edgeLists.get(node1);
        if (edges == null) {
            return null;
        }

        Edge edge = new Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW);
        if (edges.contains(edge)) {
            return edgesSet.get(edge);
        }

        edge = new Edge(node2, node1, Endpoint.TAIL, Endpoint.ARROW);
        if (edges.contains(edge)) {
            return edgesSet.get(edge);
        }

        edge = new Edge(node1, node2, Endpoint.TAIL, Endpoint.TAIL);
        if (edges.contains(edge)) {
            return edgesSet.get(edge);
        }

        edge = new Edge(node2, node1, Endpoint.TAIL, Endpoint.TAIL);
        if (edges.contains(edge)) {
            return edgesSet.get(edge);
        }

        return null;
    }

    @Override
    public Edge getDirectedEdge(Node node1, Node node2) {
        List<Edge> edges = getEdges(node1, node2);

        if (edges == null) {
            return null;
        }

        if (edges.isEmpty()) {
            return null;
        }

        for (Edge edge : edges) {
            if (Edges.isDirectedEdge(edge) && getProximalEndpoint(node2,edge) == Endpoint.ARROW) {
                return edge;
            }
        }

        return null;
    }

    /**
     * @return the list of parents for a node.
     */
    @Override
    public List<Node> getParents(Node node) {
        List<Node> parents = new ArrayList<>();
        Set<Edge> edges = this.edgeLists.get(node);
        
        if (edges != null) {
            for (Edge edge : edges) {
                if (edge == null) continue;

                Endpoint endpoint1 = getDistalEndpoint(node,edge);
                Endpoint endpoint2 = getProximalEndpoint(node,edge);

                if (endpoint1 == Endpoint.TAIL && endpoint2 == Endpoint.ARROW) {
                    parents.add(getDistalNode(node,edge));
                }
            }
        }

        return parents;
    }

    /**
     * @return the number of edges into the given node.
     */
    @Override
    public int getIndegree(Node node) {
        return getParents(node).size();
    }

    @Override
    public int getDegree(Node node) {
        return this.edgeLists.get(node).size();
    }

    /**
     * @return the number of edges out of the given node.
     */
    @Override
    public int getOutdegree(Node node) {
        return getChildren(node).size();
    }

    /**
     * Determines whether some edge or other exists between two nodes.
     */
    @Override
    public boolean isAdjacentTo(Node node1, Node node2) {
        if (node1 == null || node2 == null) {
            return false;
        }

        return neighboursMap.get(node1).contains(node2) && neighboursMap.get(node2).contains(node1);
    }

    /**
     * Determines whether one node is an ancestor of another.
     */
    @Override
    public boolean isAncestorOf(Node node1, Node node2) {
        return getAncestors(Collections.singletonList(node2)).contains(node1);
    }

    @Override
    public boolean possibleAncestor(Node node1, Node node2) {
        return existsSemiDirectedPathFromTo(node1,
                Collections.singleton(node2));
    }

    /**
     * @return true iff node1 is a possible ancestor of at least one member of
     * nodes2
     */
    protected boolean possibleAncestorSet(Node node1, List<Node> nodes2) {
        for (Node node2 : nodes2) {
            if (possibleAncestor(node1, node2)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public List<Node> getAncestors(List<Node> nodes) {
        Set<Node> ancestors = new HashSet<>();

        for (Node node : nodes) {
            collectAncestorsVisit(node, ancestors);
        }

        return new ArrayList<>(ancestors);
    }

    /**
     * Determines whether one node is a child of another.
     */
    @Override
    public boolean isChildOf(Node node1, Node node2) {
        for (Edge edge : getEdges(node2)) {
            Node sub = traverseDirected(node2, edge);

            if (sub.equals(node1)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Determines whether one node is a descendent of another.
     */
    @Override
    public boolean isDescendentOf(Node node1, Node node2) {
        return node1.equals(node2) || GraphUtils.existsDirectedPathFromTo(node2, node1, this);
    }

    /**
     * added by ekorber, 2004/06/12
     *
     * @return true iff node2 is a definite nondecendent of node1
     */
    @Override
    public boolean defNonDescendent(Node node1, Node node2) {
        return !(possibleAncestor(node1, node2));
    }

    @Override
    public boolean isDConnectedTo(Node x, Node y, List<Node> z) {
        return GraphUtils.isDConnectedTo(x, y, z, this);
    }

    public boolean isDConnectedTo(List<Node> x, List<Node> y, List<Node> z) {
        return GraphUtils.isDConnectedTo(x, y, z, this);
    }


    @Override
    public List<Node> getSepset(Node x, Node y) {
        return GraphUtils.getSepset(x, y, this);
    }

    public boolean isDSeparatedFrom(List<Node> x, List<Node> y, List<Node> z) {
        return !isDConnectedTo(x, y, z);
    }

    /**
     * True if this graph has been stamped as a cpdag. The search algorithm
     * should do this.
     */
    @Override
    public boolean isCPDAG() {
        return this.cpdag;
    }

    @Override
    public void setCPDAG(boolean cpdag) {
        this.cpdag = cpdag;
    }

    /**
     * True if this graph has been "stamped" as a PAG_of_the_true_DAG. The
     * search algorithm should do this.
     */
    @Override
    public boolean isPag() {
        return this.pag;
    }

    @Override
    public void setPag(boolean pag) {
        this.pag = pag;
    }

    /**
     * Determines whether one n ode is d-separated from another. According to
     * Spirtes, Richardson and Meek, two nodes are d- connected given some
     * conditioning set Z if there is an acyclic undirected path U between them,
     * such that every collider on U is an ancestor of some element in Z and
     * every non-collider on U is not in Z. Two elements are d-separated just in
     * case they are not d-connected. A collider is a node which two edges hold
     * in common for which the endpoints leading into the node are both arrow
     * endpoints.
     *
     * @param node1 the first node.
     * @param node2 the second node.
     * @param z     the conditioning set.
     * @return true if node1 is d-separated from node2 given set t, false if
     * not.
     * @see #isDConnectedTo
     */
    @Override
    public boolean isDSeparatedFrom(Node node1, Node node2, List<Node> z) {
        return !isDConnectedTo(node1, node2, z);
    }

    //added by ekorber, June 2004
    @Override
    public boolean possDConnectedTo(Node node1, Node node2,
                                    List<Node> condNodes) {
        LinkedList<Node> allNodes = new LinkedList<>(getNodes());
        int sz = allNodes.size();
        int[][] edgeStage = new int[sz][sz];
        int stage = 1;

        int n1x = allNodes.indexOf(node1);
        int n2x = allNodes.indexOf(node2);

        edgeStage[n1x][n1x] = 1;
        edgeStage[n2x][n2x] = 1;

        List<int[]> currEdges;
        List<int[]> nextEdges = new LinkedList<>();

        int[] temp1 = new int[2];
        temp1[0] = n1x;
        temp1[1] = n1x;
        nextEdges.add(temp1);

        int[] temp2 = new int[2];
        temp2[0] = n2x;
        temp2[1] = n2x;
        nextEdges.add(temp2);

        while (true) {
            currEdges = nextEdges;
            nextEdges = new LinkedList<>();
            for (int[] edge : currEdges) {
                Node center = allNodes.get(edge[1]);
                List<Node> adj = new LinkedList<>(getAdjacentNodes(center));

                for (Node anAdj : adj) {
                    // check if we've hit this edge before
                    int testIndex = allNodes.indexOf(anAdj);
                    if (edgeStage[edge[1]][testIndex] != 0) {
                        continue;
                    }

                    // if the edge pair violates possible d-connection,
                    // then go to the next adjacent node.
                    Node X = allNodes.get(edge[0]);
                    Node Y = allNodes.get(edge[1]);
                    Node Z = allNodes.get(testIndex);

                    if (!((isDefNoncollider(X, Y, Z)
                            && !(condNodes.contains(Y))) || (isDefCollider(X, Y, Z)
                            && possibleAncestorSet(Y, condNodes)))) {
                        continue;
                    }

                    // if it gets here, then it's legal, so:
                    // (i) if this is the one we want, we're done
                    if (anAdj.equals(node2)) {
                        return true;
                    }

                    // (ii) if we need to keep going,
                    // add the edge to the nextEdges list
                    int[] nextEdge = new int[2];
                    nextEdge[0] = edge[1];
                    nextEdge[1] = testIndex;
                    nextEdges.add(nextEdge);

                    // (iii) set the edgeStage array
                    edgeStage[edge[1]][testIndex] = stage;
                    edgeStage[testIndex][edge[1]] = stage;
                }
            }

            // find out if there's any reason to move to the next stage
            if (nextEdges.isEmpty()) {
                break;
            }

            stage++;
        }

        return false;
    }

    /**
     * Determines whether an inducing path exists between node1 and node2, given
     * a set O of observed nodes and a set sem of conditioned nodes.
     *
     * @param node1 the first node.
     * @param node2 the second node.
     * @return true if an inducing path exists, false if not.
     */
    @Override
    public boolean existsInducingPath(Node node1, Node node2) {
        return node1.equals(node2) || GraphUtils.existsDirectedPathFromTo(node2, node1, this);
    }

    /**
     * Determines whether one node is a parent of another.
     *
     * @param node1 the first node.
     * @param node2 the second node.
     * @return true if node1 is a parent of node2, false if not.
     * @see #isChildOf
     * @see #getParents
     * @see #getChildren
     */
    @Override
    public boolean isParentOf(Node node1, Node node2) {
        for (Edge edge : getEdges(node1)) {
            Node sub = traverseDirected(node1, (edge));

            if (sub != null && sub.equals(node2)) {
                return true;
            }
        }

        return false;
    }
    
    /**
     * For A -&gt; B, given A, returns B; otherwise returns null.
     */
    public static Node traverseDirected(Node node, Edge edge) {
        if (node.equals(edge.getNode1())) {
            if ((edge.getEndpoint1() == Endpoint.TAIL) &&
                    (edge.getEndpoint2() == Endpoint.ARROW)) {
                return edge.getNode2();
            }
        } else if (node.equals(edge.getNode2())) {
            if ((edge.getEndpoint2() == Endpoint.TAIL) &&
                    (edge.getEndpoint1() == Endpoint.ARROW)) {
                return edge.getNode1();
            }
        }

        return null;
    }
    
    /**
     * For A --* B or A o-* B, given A, returns B. For A &lt;-* B, returns null.
     * Added by ekorber, 2004/06/12.
     */
    public static Node traverseSemiDirected(Node node, Edge edge) {
        if (node.equals(edge.getNode1())) {
            if ((edge.getEndpoint1() == Endpoint.TAIL || edge.getEndpoint1() == Endpoint.CIRCLE)) {
                return edge.getNode2();
            }
        } else if (node.equals(edge.getNode2())) {
            if ((edge.getEndpoint2() == Endpoint.TAIL || edge.getEndpoint2() == Endpoint.CIRCLE)) {
                return edge.getNode1();
            }
        }
        return null;
    }

    /**
     * Determines whether one node is a proper ancestor of another.
     */
    @Override
    public boolean isProperAncestorOf(Node node1, Node node2) {
        return (!node1.equals(node2)) && isAncestorOf(node1, node2);
    }

    /**
     * Determines whether one node is a proper decendent of another
     */
    @Override
    public boolean isProperDescendentOf(Node node1, Node node2) {
        return (!node1.equals(node2)) && isDescendentOf(node1, node2);
    }

    /**
     * Transfers nodes and edges from one graph to another. One way this is used
     * is to change graph types. One constructs a new graph based on the old
     * graph, and this method is called to transfer the nodes and edges of the
     * old graph to the new graph.
     *
     * @param graph the graph from which nodes and edges are to be pilfered.
     * @throws IllegalArgumentException This exception is thrown if adding some
     *                                  node or edge violates one of the basicConstraints of this graph.
     */
    @Override
    public void transferNodesAndEdges(Graph graph)
            throws IllegalArgumentException {
        if (graph == null) {
            throw new NullPointerException("No graph was provided.");
        }

//        System.out.println("TANSFER BEFORE " + graph.getEdges());
        for (Node node : graph.getNodes()) {
            if (!addNode(node)) {
                throw new IllegalArgumentException();
            }
        }

        for (Edge edge : graph.getEdges()) {
            if (!addEdge(edge)) {
                throw new IllegalArgumentException();
            }
        }

        this.ancestors = null;
//        System.out.println("TANSFER AFTER " + getEdges());
    }

    @Override
    public void transferAttributes(Graph graph)
            throws IllegalArgumentException {
        if (graph == null) {
            throw new NullPointerException("No graph was provided.");
        }

        this.attributes.putAll(graph.getAllAttributes());
    }

    /**
     * Determines whether a node in a graph is exogenous.
     */
    @Override
    public boolean isExogenous(Node node) {
        return getIndegree(node) == 0;
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
     * Removes the edge connecting the two given nodes.
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
     * @return the endpoint along the edge from node to node2 at the node2 end.
     */
    @Override
    public Endpoint getEndpoint(Node node1, Node node2) {
        List<Edge> edges = getEdges(node2);

        for (Edge edge : edges) {
            if (getDistalNode(node2,edge).equals(node1)) {
                return getProximalEndpoint(node2,edge);
            }
        }

        return null;
    }

    /**
     * If there is currently an edge from node1 to node2, sets the endpoint at
     * node2 to the given endpoint; if there is no such edge, adds an edge --#
     * where # is the given endpoint. Setting an endpoint to null, provided
     * there is exactly one edge connecting the given nodes, removes the edge.
     * (If there is more than one edge, an exception is thrown.)
     *
     * @throws IllegalArgumentException if the edge with the revised endpoint
     *                                  cannot be added to the graph.
     */
    @Override
    public boolean setEndpoint(Node from, Node to, Endpoint endPoint)
            throws IllegalArgumentException {
        if (!isAdjacentTo(from, to)) return false;

        Edge edge = getEdge(from, to);
        this.ancestors = null;

        removeEdge(edge);

        Edge newEdge = new Edge(from, to,
                getProximalEndpoint(from,edge), endPoint);
        addEdge(newEdge);
        return true;
    }

    /**
     * Nodes adjacent to the given node with the given proximal endpoint.
     */
    @Override
    public List<Node> getNodesInTo(Node node, Endpoint endpoint) {
        List<Node> nodes = new ArrayList<>(4);
        List<Edge> edges = getEdges(node);

        for (Edge edge : edges) {
            if (getProximalEndpoint(node,edge) == endpoint) {
                nodes.add(getDistalNode(node,edge));
            }
        }

        return nodes;
    }

    /**
     * Nodes adjacent to the given node with the given distal endpoint.
     */
    @Override
    public List<Node> getNodesOutTo(Node node, Endpoint endpoint) {
        List<Node> nodes = new ArrayList<>(4);
        List<Edge> edges = getEdges(node);

        for (Edge edge : edges) {
            if (getDistalEndpoint(node,edge) == endpoint) {
                nodes.add(getDistalNode(node,edge));
            }
        }

        return nodes;
    }

    /**
     * @return a matrix of endpoints for the nodes in this graph, with nodes in
     * the same order as getNodes().
     */
    @Override
    public Endpoint[][] getEndpointMatrix() {
        Node[] arrNodes = (Node[])this.nodes.toArray();
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

            this.edgesSet.put(edge,edge);
            
            // Now the two nodes are neighbours
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
     * Adds a PropertyChangeListener to the graph.
     *
     * @param l the property change listener.
     */
    @Override
    public void addPropertyChangeListener(PropertyChangeListener l) {
        getPcs().addPropertyChangeListener(l);
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
        
        if (!this.nodes.add(node)) {
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
     * @return the list of edges in the graph. No particular ordering of the
     * edges in the list is guaranteed.
     */
    @Override
    public Set<Edge> getEdges() {
        return new HashSet<>(this.edgesSet.keySet());
    }

    /**
     * Determines if the graph contains a particular edge.
     */
    @Override
    public boolean containsEdge(Edge edge) {
        return this.edgesSet.containsKey(edge);
    }

    /**
     * Determines whether the graph contains a particular node.
     */
    @Override
    public boolean containsNode(Node node) {
        return this.nodes.contains(node);
    }

    /**
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

    @Override
    public int hashCode() {
        int hashCode = 0;

        for (Edge edge : getEdges()) {
            hashCode += edge.hashCode();
        }

        return (new HashSet<>(this.nodes)).hashCode() + hashCode;
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

        if (o instanceof EdgeListGraph) {
            EdgeListGraph _o = (EdgeListGraph) o;
            boolean nodesEqual = new HashSet<>(_o.nodes).equals(new HashSet<>(this.nodes));
            boolean edgesEqual = new HashMap<>(_o.edgesSet).equals(new HashMap<>(this.edgesSet));
            return (nodesEqual && edgesEqual);
        } else {
            Graph graph = (Graph) o;
            return new HashSet<>(graph.getNodeNames()).equals(new HashSet<>(getNodeNames()))
                    && new HashSet<>(graph.getEdges()).equals(new HashSet<>(getEdges()));

        }
    }

    /**
     * Resets the graph so that it is fully connects it using #-# edges, where #
     * is the given endpoint.
     */
    @Override
    public void fullyConnect(Endpoint endpoint) {
        this.edgesSet.clear();
        this.edgeLists.clear();

        for (Node node : this.nodes) {
            this.edgeLists.put(node, new HashSet<>());
        }
        
        Node[] arrNodes = (Node[])this.nodes.toArray();
        int size = arrNodes.length;

        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                Edge edge = new Edge(arrNodes[i], arrNodes[j], endpoint, endpoint);
                addEdge(edge);
            }
        }
    }

    @Override
    public void reorientAllWith(Endpoint endpoint) {
        for (Edge edge : this.edgesSet.keySet()) {
            Node a = edge.getNode1();
            Node b = edge.getNode2();

            setEndpoint(a, b, endpoint);
            setEndpoint(b, a, endpoint);
        }
    }

    /**
     * @return the node with the given name, or null if no such node exists.
     */
    @Override
    public Node getNode(String name) {
        return this.namesHash.get(name);
    }

    /**
     * @return the number of nodes in the graph.
     */
    @Override
    public int getNumNodes() {
        return this.nodes.size();
    }

    /**
     * @return the number of edges in the (entire) graph.
     */
    @Override
    public int getNumEdges() {
        return this.edgesSet.size();
    }

    /**
     * @return the number of edges connected to a particular node in the graph.
     */
    @Override
    public int getNumEdges(Node node) {
        Set<Edge> list = this.edgeLists.get(node);
        return (list == null) ? 0 : list.size();
    }

    @Override
    public List<Node> getNodes() {
        return new ArrayList<>(this.nodes);
    }

    public Set<Node> getNodesSet() {
        return this.nodes;
    }

    @Override
    public void setNodes(List<Node> nodes) {
        if (nodes.size() != this.nodes.size()) {
            throw new IllegalArgumentException("Sorry, there is a mismatch in the number of variables "
                    + "you are trying to set.");
        }

        this.nodes.clear();
        this.nodes.addAll(nodes);
    }
    
    public void setNodes(HashSet<Node> nodesHash) {
        if (nodesHash.size() != this.nodes.size()) {
            throw new IllegalArgumentException("Sorry, there is a mismatch in the number of variables "
                    + "you are trying to set.");
        }

        this.nodes.clear();
        this.nodes.addAll(nodes);
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

        Iterator<Node> it2 = this.nodes.iterator();

        while (it2.hasNext()) {
            Node node = it2.next();
            it2.remove();
            this.namesHash.remove(node.getName());
            getPcs().firePropertyChange("nodeRemoved", node, null);
        }

        this.edgeLists.clear();
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
            if (!this.edgesSet.containsKey(edge)) {
                return false;
            }

            Set<Edge> edgeList1 = this.edgeLists.get(edge.getNode1());
            Set<Edge> edgeList2 = this.edgeLists.get(edge.getNode2());

            edgeList1 = new HashSet<>(edgeList1);
            edgeList2 = new HashSet<>(edgeList2);
            
            // Si no existe el enlace inverso, dejan de ser vecinos
            if (!edgesSet.containsKey(edge.reverse())){
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
     * Removes all edges connecting node A to node B.
     *
     * @param node1 the first node.,
     * @param node2 the second node.
     * @return true if edges were removed between A and B, false if not.
     */
    @Override
    public boolean removeEdges(Node node1, Node node2) {
        return removeEdges(getEdges(node1, node2));
    }

    /**
     * Removes a node from the graph.
     */
    @Override
    public boolean removeNode(Node node) {
        if (!this.nodes.remove(node)) {
            return false;
        }

        boolean changed = false;
        Set<Edge> edgeList1 = this.edgeLists.get(node);    //list of edges connected to that node
        for (Edge edge : edgeList1) {
            this.edgesSet.remove(edge);
        }

        for (Iterator<Edge> i = edgeList1.iterator(); i.hasNext(); ) {
            Edge edge = (i.next());
            Node node2 = getDistalNode(node,edge);

            if (!node2.equals(node)) {
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
     * Traverses the edge in an undirected fashion--given one node along the
     * edge, returns the node at the opposite end of the edge.
     */
    public final Node getDistalNode(Node node, Edge edge) {
        Node node1 = edge.getNode1();
        Node node2 = edge.getNode2();
        
        if (node1.equals(node)) {
            return node2;
        }

        if (node2.equals(node)) {
            return node1;
        }

        return null;
    }
    
    /**
     * @return the endpoint nearest to the given node.
     * @throws IllegalArgumentException if the given node is not along the edge.
     */
    public final Endpoint getProximalEndpoint(Node node, Edge edge) {
        Node node1 = edge.getNode1();
        Node node2 = edge.getNode2();
        
        if (node1.equals(node)) {
            return edge.getEndpoint1();
        } else if (node2.equals(node)) {
            return edge.getEndpoint2();
        }

        return null;
    }

    /**
     * @return the endpoint furthest from the given node.
     * @throws IllegalArgumentException if the given node is not along the edge.
     */
    public final Endpoint getDistalEndpoint(Node node, Edge edge) {
        Node node1 = edge.getNode1();
        Node node2 = edge.getNode2();
        
        if (node1.equals(node)) {
            return edge.getEndpoint2();
        } else if (node2.equals(node)) {
            return edge.getEndpoint1();
        }

        return null;
    }

    /**
     * Removes any relevant node objects found in this collection.
     *
     * @param newNodes the collection of nodes to remove.
     * @return true if nodes from the collection were removed, false if not.
     */
    @Override
    public boolean removeNodes(List<Node> newNodes) {
        boolean changed = false;

        for (Node node : newNodes) {
            boolean _changed = removeNode(node);
            changed = changed || _changed;
        }

        return changed;
    }

    /**
     * @return a string representation of the graph.
     */
    @Override
    public String toString() {
        return GraphUtils.graphToText(this);
    }

    @Override
    public Graph subgraph(List<Node> nodes) {
        Graph graph = new EdgeListGraph(nodes);
        Set<Edge> edges = getEdges();

        for (Edge edge : edges) {
            if (nodes.contains(edge.getNode1())
                    && nodes.contains(edge.getNode2())) {
                graph.addEdge(edge);
            }
        }

        setPag(graph.isPag());

        return graph;
    }

    /**
     * @return the edges connecting node1 and node2.
     */
    @Override
    public List<Edge> getEdges(Node node1, Node node2) {
        Set<Edge> edges = this.edgeLists.get(node1);
        if (edges == null) {
            return new ArrayList<>();
        }

        List<Edge> _edges = new ArrayList<>();

        for (Edge edge : edges) {
            if (getDistalNode(node1,edge).equals(node2)) {
                _edges.add(edge);
            }
        }

        return _edges;
    }

    @Override
    public Set<Triple> getAmbiguousTriples() {
        return new HashSet<>(this.ambiguousTriples);
    }

    @Override
    public void setAmbiguousTriples(Set<Triple> triples) {
        this.ambiguousTriples.clear();

        for (Triple triple : triples) {
            addAmbiguousTriple(triple.getX(), triple.getY(), triple.getZ());
        }
    }

    @Override
    public Set<Triple> getUnderLines() {
        return new HashSet<>(this.underLineTriples);
    }

    @Override
    public Set<Triple> getDottedUnderlines() {
//        removeTriplesNotInGraph();
        return new HashSet<>(this.dottedUnderLineTriples);
    }

    /**
     * States whether r-s-r is an underline triple or not.
     */
    @Override
    public boolean isAmbiguousTriple(Node x, Node y, Node z) {
        return this.ambiguousTriples.contains(new Triple(x, y, z));
    }

    /**
     * States whether r-s-r is an underline triple or not.
     */
    @Override
    public boolean isUnderlineTriple(Node x, Node y, Node z) {
        return this.underLineTriples.contains(new Triple(x, y, z));
    }

    /**
     * States whether r-s-r is an underline triple or not.
     */
    @Override
    public boolean isDottedUnderlineTriple(Node x, Node y, Node z) {
        return this.dottedUnderLineTriples.contains(new Triple(x, y, z));
    }

    @Override
    public void addAmbiguousTriple(Node x, Node y, Node z) {
        this.ambiguousTriples.add(new Triple(x, y, z));
    }

    @Override
    public void addUnderlineTriple(Node x, Node y, Node z) {
        Triple triple = new Triple(x, y, z);

        if (!triple.alongPathIn(this)) {
            return;
//            throw new IllegalArgumentException("<" + x + ", " + y + ", " + z + "> must lie along a path in the graph.");
        }

        this.underLineTriples.add(new Triple(x, y, z));
    }

    @Override
    public void addDottedUnderlineTriple(Node x, Node y, Node z) {
        Triple triple = new Triple(x, y, z);

        if (!triple.alongPathIn(this)) {
            return;
//            throw new IllegalArgumentException("<" + x + ", " + y + ", " + z + "> must lie along a path in the graph.");
        }

        this.dottedUnderLineTriples.add(triple);
    }

    @Override
    public void removeAmbiguousTriple(Node x, Node y, Node z) {
        this.ambiguousTriples.remove(new Triple(x, y, z));
    }

    @Override
    public void removeUnderlineTriple(Node x, Node y, Node z) {
        this.underLineTriples.remove(new Triple(x, y, z));
    }

    @Override
    public void removeDottedUnderlineTriple(Node x, Node y, Node z) {
        this.dottedUnderLineTriples.remove(new Triple(x, y, z));
    }

    @Override
    public void setUnderLineTriples(Set<Triple> triples) {
        this.underLineTriples.clear();

        for (Triple triple : triples) {
            addUnderlineTriple(triple.getX(), triple.getY(), triple.getZ());
        }
    }

    @Override
    public void setDottedUnderLineTriples(Set<Triple> triples) {
        this.dottedUnderLineTriples.clear();

        for (Triple triple : triples) {
            addDottedUnderlineTriple(triple.getX(), triple.getY(), triple.getZ());
        }
    }

    @Override
    public List<String> getNodeNames() {
        List<String> names = new ArrayList<>();

        for (Node node : getNodes()) {
            names.add(node.getName());
        }

        return names;
    }

    //===============================PRIVATE METHODS======================//
    @Override
    public void removeTriplesNotInGraph() {
//        if (!stuffRemovedSinceLastTripleAccess) return;

        for (Triple triple : new HashSet<>(this.ambiguousTriples)) {
            if (!containsNode(triple.getX()) || !containsNode(triple.getY()) || !containsNode(triple.getZ())) {
                this.ambiguousTriples.remove(triple);
                continue;
            }

            if (!isAdjacentTo(triple.getX(), triple.getY()) || !isAdjacentTo(triple.getY(), triple.getZ())) {
                this.ambiguousTriples.remove(triple);
            }
        }

        for (Triple triple : new HashSet<>(this.underLineTriples)) {
            if (!containsNode(triple.getX()) || !containsNode(triple.getY()) || !containsNode(triple.getZ())) {
                this.underLineTriples.remove(triple);
                continue;
            }

            if (!isAdjacentTo(triple.getX(), triple.getY()) || !isAdjacentTo(triple.getY(), triple.getZ())) {
                this.underLineTriples.remove(triple);
            }
        }

        for (Triple triple : new HashSet<>(this.dottedUnderLineTriples)) {
            if (!containsNode(triple.getX()) || !containsNode(triple.getY()) || !containsNode(triple.getZ())) {
                this.dottedUnderLineTriples.remove(triple);
                continue;
            }

            if (!isAdjacentTo(triple.getX(), triple.getY()) || !isAdjacentTo(triple.getY(), triple.getZ())) {
                this.dottedUnderLineTriples.remove(triple);
            }
        }

        this.stuffRemovedSinceLastTripleAccess = false;
    }

    private void collectAncestorsVisit(Node node, Set<Node> ancestors) {
        if (ancestors.contains(node)) {
            return;
        }

        ancestors.add(node);
        List<Node> parents = getParents(node);

        if (!parents.isEmpty()) {
            for (Node parent : parents) {
                collectAncestorsVisit(parent, ancestors);
            }
        }
    }

    private void collectDescendantsVisit(Node node, Set<Node> descendants) {
        descendants.add(node);
        List<Node> children = getChildren(node);

        if (!children.isEmpty()) {
            for (Node child : children) {
                doChildClosureVisit(child, descendants);
            }
        }
    }

    /**
     * closure under the child relation
     */
    private void doChildClosureVisit(Node node, Set<Node> closure) {
        if (!closure.contains(node)) {
            closure.add(node);

            for (Edge edge1 : getEdges(node)) {
                Node sub = traverseDirected(node, edge1);

                if (sub == null) {
                    continue;
                }

                doChildClosureVisit(sub, closure);
            }
        }
    }

    /**
     * @return this object.
     */
    protected PropertyChangeSupport getPcs() {
        if (this.pcs == null) {
            this.pcs = new PropertyChangeSupport(this);
        }
        return this.pcs;
    }

    /**
     * @return true iff there is a directed path from node1 to node2.
     */
    boolean existsUndirectedPathVisit(Node node1, Node node2, Set<Node> path) {
        path.add(node1);

        for (Edge edge : getEdges(node1)) {
            Node child = Edges.traverse(node1, edge);

            if (child == null) {
                continue;
            }

            if (child.equals(node2)) {
                return true;
            }

            if (path.contains(child)) {
                continue;
            }

            if (existsUndirectedPathVisit(child, node2, path)) {
                return true;
            }
        }

        path.remove(node1);
        return false;
    }

    /**
     * @return true iff there is a semi-directed path from node1 to node2
     */
    private boolean existsSemiDirectedPathVisit(Node node1, Set<Node> nodes2,
                                                LinkedList<Node> path) {
        path.addLast(node1);

        for (Edge edge : getEdges(node1)) {
            Node child = traverseSemiDirected(node1, edge);

            if (child == null) {
                continue;
            }

            if (nodes2.contains(child)) {
                return true;
            }

            if (path.contains(child)) {
                continue;
            }

            if (existsSemiDirectedPathVisit(child, nodes2, path)) {
                return true;
            }
        }

        path.removeLast();
        return false;
    }

    @Override
    public List<Node> getCausalOrdering() {
        return GraphUtils.getCausalOrdering(this, this.getNodes());
    }

    @Override
    public void setHighlighted(Edge edge, boolean highlighted) {
        this.highlightedEdges.add(edge);
    }

    @Override
    public boolean isHighlighted(Edge edge) {
        return this.highlightedEdges != null && this.highlightedEdges.contains(edge);
    }

    @Override
    public boolean isParameterizable(Node node) {
        return true;
    }

    @Override
    public boolean isTimeLagModel() {
        return false;
    }

    @Override
    public TimeLagGraph getTimeLagGraph() {
        return null;
    }

    /**
     * Adds semantic checks to the default deserialization method. This method
     * must have the standard signature for a readObject method, and the body of
     * the method must begin with "s.defaultReadObject();". Other than that, any
     * semantic checks can be specified and do not need to stay the same from
     * version to version. A readObject method of this form may be added to any
     * class, even if Tetrad sessions were previously saved out using a version
     * of the class that didn't include it. (That's what the
     * "s.defaultReadObject();" is for. See J. Bloch, Effective Java, for help.
     */
    private void readObject(ObjectInputStream s)
            throws IOException, ClassNotFoundException {
        s.defaultReadObject();

        if (this.nodes == null) {
            throw new NullPointerException();
        }

        if (this.edgesSet == null) {
            throw new NullPointerException();
        }

        if (this.edgeLists == null) {
            throw new NullPointerException();
        }

        if (this.ambiguousTriples == null) {
            this.ambiguousTriples = new HashSet<>();
        }

        if (this.highlightedEdges == null) {
            this.highlightedEdges = new HashSet<>();
        }

        if (this.underLineTriples == null) {
            this.underLineTriples = new HashSet<>();
        }

        if (this.dottedUnderLineTriples == null) {
            this.dottedUnderLineTriples = new HashSet<>();
        }
    }

    public void changeName(String name, String newName) {
        Node node = this.namesHash.get(name);
        this.namesHash.remove(name);
        node.setName(newName);
        this.namesHash.put(newName, node);
    }

    /**
     * @return the names of the triple classifications. Coordinates with
     * <code>getTriplesList</code>
     */
    @Override
    public List<String> getTriplesClassificationTypes() {
        List<String> names = new ArrayList<>();
        names.add("Underlines");
        names.add("Dotted Underlines");
        names.add("Ambiguous Triples");
        return names;
    }

    /**
     * @return the list of triples corresponding to
     * <code>getTripleClassificationNames</code> for the given node.
     */
    @Override
    public List<List<Triple>> getTriplesLists(Node node) {
        List<List<Triple>> triplesList = new ArrayList<>();
        triplesList.add(GraphUtils.getUnderlinedTriplesFromGraph(node, this));
        triplesList.add(GraphUtils.getDottedUnderlinedTriplesFromGraph(node, this));
        triplesList.add(GraphUtils.getAmbiguousTriplesFromGraph(node, this));
        return triplesList;
    }

    public void setStuffRemovedSinceLastTripleAccess(
            boolean stuffRemovedSinceLastTripleAccess) {
        this.stuffRemovedSinceLastTripleAccess = stuffRemovedSinceLastTripleAccess;
    }

    @Override
    public Map<String, Object> getAllAttributes() {
        return this.attributes;
    }

    @Override
    public Object getAttribute(String key) {
        return this.attributes.get(key);
    }

    @Override
    public void removeAttribute(String key) {
        this.attributes.remove(key);
    }

    @Override
    public void addAttribute(String key, Object value) {
        this.attributes.put(key, value);
    }
}
