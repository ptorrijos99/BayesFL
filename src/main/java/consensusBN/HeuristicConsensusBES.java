package consensusBN;

import java.util.*;

import edu.cmu.tetrad.graph.*;
import org.albacete.simd.threads.BESThread;
import org.albacete.simd.threads.GESThread;
import org.albacete.simd.utils.Utils;

import static org.albacete.simd.utils.Utils.pdagToDag;

public class HeuristicConsensusBES {
	
	ArrayList<Node> alpha;
	Dag outputDag = null;
	ArrayList<Dag> setOfdags;
	ArrayList<Dag> setOfOutDags;
	Dag union;
	Map<String, List<Set<EdgeFordFulkerson>>> localCache = new HashMap<>();
	int maxSize;  // 10
	int maxTW;

	
	public HeuristicConsensusBES(ArrayList<Dag> dags, int maxSize, int maxTW){
		this.setOfdags = new ArrayList<>();
		for (Dag d : dags) {
			this.setOfdags.add(new Dag(d));
		}

		this.alpha = AlphaOrder.alphaOrder(dags);
		this.setOfOutDags = new ArrayList<>();
		for (Dag d : dags) {
			this.setOfOutDags.add(BetaToAlpha.transformToAlpha(d, this.alpha));
		}
		this.maxSize = maxSize;
		this.union = ConsensusUnion.fusionUnion(this.setOfOutDags);
		this.maxTW = maxTW;
	}

	public Dag fusion(){
		Graph graph = new EdgeListGraph(this.union);

		while (Utils.getTreeWidth(graph) > maxTW) {
			Node x = null, y = null;
			double bestScore = Double.POSITIVE_INFINITY;
			HashSet<Node> bestSubSet = new HashSet<>();
			List<Set<EdgeFordFulkerson>> bestMinCut = new ArrayList<>();

			Set<Edge> edges1 = graph.getEdges();
			List<Edge> edges = new ArrayList<>();
			for (Edge edge : edges1) {
				Node _x = edge.getNode1();
				Node _y = edge.getNode2();

				if (Edges.isUndirectedEdge(edge)) {
					edges.add(Edges.directedEdge(_x, _y));
					edges.add(Edges.directedEdge(_y, _x));
				} else {
					edges.add(edge);
				}
			}

			for (Edge edge : edges) {
				Node _x = edge.getNode1();
				Node _y = edge.getNode2();

				List<Node> hNeighbors = GESThread.getHNeighbors(_x, _y, graph);
				List<HashSet<Node>> hSubsets = BESThread.generatePowerSet(hNeighbors);

				for (HashSet<Node> hSubset : hSubsets) {
					if(hSubset.size() > maxSize) continue;  // TODO: Meter también un random para comprobar solo algunos

					// TODO_: Hacer test al principio
					Set<Node> naYXH = GESThread.findNaYX(_x, _y, graph);
					naYXH.removeAll(hSubset);
					if (!GraphUtils.isClique(naYXH, graph)) continue;

					// deleteEval(_x, _y, hSubset, graph);
					Set<Node> set = new HashSet<>(GESThread.findNaYX(_x, _y, graph));
					set.removeAll(hSubset);
					set.addAll(graph.getParents(_y));
					set.remove(_x);

					// scoreGraphChangeDelete(_y, _x, set)
					String key = _y.getName() + _x.getName() + set;
					List<Set<EdgeFordFulkerson>> minCut = this.localCache.get(key);
					double evalScore = 0.0;

					if(minCut == null) {
						minCut = new ArrayList<>();
						LinkedList<Node> conditioning = new LinkedList<>(set);

						for (Dag g : this.setOfdags) {
							// TODO_: minCut. Cambiar por el mínimo número de enlaces que hay que borrar para que estén d-separados
							Graph aux = constructConditionedGraph(g, _y, _x, conditioning);

							// Ejecuta el algoritmo de Ford-Fulkerson para encontrar el flujo máximo
							Map<Node, List<EdgeFordFulkerson>> adjList = fordFulkerson(aux, _y, _x);

							// Obtén el conjunto mínimo de enlaces a eliminar para d-separar
							Set<EdgeFordFulkerson> minCutEdges = findMinCut(adjList, _y); // Ajusta según sea necesario
							minCut.add(minCutEdges);

							// Suma el tamaño del conjunto mínimo de corte al evalScore
							evalScore += minCutEdges.size();
						}
						evalScore = evalScore / (double) this.setOfdags.size();
						this.localCache.put(key, minCut);
					} else {
						for (Set<EdgeFordFulkerson> minCutEdges : minCut) {
							evalScore += minCutEdges.size();
						}
						evalScore = evalScore / (double) this.setOfdags.size();
					}

					// TODO_: en nuestro caso queremos que sea lo menor posible, porque aquí mide d-separaciones
					if (evalScore < bestScore) {
						bestScore = evalScore;
						bestSubSet = hSubset;
						bestMinCut = minCut;
						x = _x;
						y = _y;
					}
				}
			}

			//System.out.println("DELETE " + graph.getEdge(x, y) + bestSubSet + " (" +bestScore + ")");
			GESThread.delete(x, y, bestSubSet, graph);

			// TODO_: Borrar los enlaces de los grafos de entrada
			for (int i = 0; i < this.setOfdags.size(); i++) {
				Dag g = this.setOfdags.get(i);
				Set<EdgeFordFulkerson> minCutEdges = bestMinCut.get(i);
				for (EdgeFordFulkerson edge : minCutEdges) {
					// TODO: Comprobar si es necesario borrar la arista en ambas direcciones
					g.removeEdge(edge.from, edge.to);
					g.removeEdge(edge.to, edge.from);
				}
			}
		}

		pdagToDag(graph);
		this.outputDag = new Dag(graph);
		return this.outputDag;
	}

	// Representación de aristas con capacidad
    class EdgeFordFulkerson {
		Node from;
		Node to;
		int capacity; // Capacidad de la arista
		int residual; // Capacidad residual

		public EdgeFordFulkerson(Node from, Node to, int capacity) {
			this.from = from;
			this.to = to;
			this.capacity = capacity;
			this.residual = capacity; // Inicialmente, residual es igual a la capacidad
		}

		@Override
		public String toString() {
			return from + " -> " + to + " (" + residual + "/" + capacity + ")";
		}
	}

	// Algoritmo Ford-Fulkerson utilizando listas de adyacencia
	Map<Node, List<EdgeFordFulkerson>> fordFulkerson(Graph g, Node source, Node sink) {
		// Inicialización: Lista de adyacencia para capacidades y capacidades residuales
		Map<Node, List<EdgeFordFulkerson>> adjList = new HashMap<>();

		// Construimos la lista de adyacencia desde el grafo original
		for (Edge e : g.getEdges()) {
			Node u = e.getNode1();
			Node v = e.getNode2();

			// Agregamos las aristas a la lista de adyacencia
			adjList.computeIfAbsent(u, k -> new ArrayList<>()).add(new EdgeFordFulkerson(u, v, 1)); // Capacidad 1
			adjList.computeIfAbsent(v, k -> new ArrayList<>()).add(new EdgeFordFulkerson(v, u, 1)); // Capacidad 1, dirección opuesta
		}

		// Aplicamos Ford-Fulkerson para calcular flujo máximo
		maxFlowFordFulkerson(adjList, source, sink);

		return adjList;
	}

	// Algoritmo Ford-Fulkerson para encontrar el flujo máximo
	int maxFlowFordFulkerson(Map<Node, List<EdgeFordFulkerson>> adjList, Node source, Node sink) {
		int maxFlow = 0;

		// Búsqueda de caminos aumentantes (usaremos BFS)
		while (true) {
			// Encontrar camino aumentante usando BFS
			Map<Node, EdgeFordFulkerson> parentMap = bfs(adjList, source, sink);
			if (parentMap == null) break;  // No hay más caminos aumentantes

			// Encuentra la capacidad mínima a lo largo del camino aumentante
			int flow = Integer.MAX_VALUE;
			for (Node v = sink; !v.equals(source); v = parentMap.get(v).from) {
				flow = Math.min(flow, parentMap.get(v).residual);
			}

			// Actualiza capacidades residuales
			for (Node v = sink; !v.equals(source); v = parentMap.get(v).from) {
				EdgeFordFulkerson e = parentMap.get(v);
				e.residual -= flow;  // Reducir capacidad residual
				// Encontrar la arista inversa y aumentar la capacidad residual de vuelta
				for (EdgeFordFulkerson reverseEdge : adjList.get(e.to)) {
					if (reverseEdge.to.equals(e.from)) {
						reverseEdge.residual += flow;
						break;
					}
				}
			}

			maxFlow += flow;  // Suma el flujo encontrado
		}

		return maxFlow;
	}

	// BFS para buscar caminos aumentantes en la red residual
	Map<Node, EdgeFordFulkerson> bfs(Map<Node, List<EdgeFordFulkerson>> adjList, Node source, Node sink) {
		Queue<Node> queue = new LinkedList<>();
		queue.add(source);

		// Usamos un mapa para rastrear el camino de predecesores
		Map<Node, EdgeFordFulkerson> parentMap = new HashMap<>();
		parentMap.put(source, null);

		while (!queue.isEmpty()) {
			Node u = queue.poll();
			List<EdgeFordFulkerson> edges = adjList.get(u);

			if (edges == null) continue;  // Si no hay aristas salientes, continuamos

			for (EdgeFordFulkerson e : edges) {
				Node v = e.to;
				// Si el nodo no ha sido visitado y tiene capacidad residual positiva
				if (!parentMap.containsKey(v) && e.residual > 0) {
					parentMap.put(v, e); // Registramos el predecesor
					if (v.equals(sink)) return parentMap; // Si llegamos al sumidero, retornamos el camino
					queue.add(v);
				}
			}
		}

		return null;  // No se encontró camino aumentante
	}

	// Encuentra el conjunto mínimo de enlaces a eliminar para que estén d-separados
	Set<EdgeFordFulkerson> findMinCut(Map<Node, List<EdgeFordFulkerson>> adjList, Node source) {
		// Paso 1: Encuentra los nodos alcanzables desde la fuente (nodos del lado de la fuente del corte)
		Set<Node> reachableFromSource = bfsReachableNodes(adjList, source);

		// Paso 2: Encuentra las aristas que cruzan el corte mínimo
		Set<EdgeFordFulkerson> minCutEdges = new HashSet<>();

		for (Node u : reachableFromSource) {
			List<EdgeFordFulkerson> edges = adjList.get(u);

			if (edges == null) continue;  // Si no hay aristas salientes, continuamos

			// Para cada arista que sale de 'u'
			for (EdgeFordFulkerson e : edges) {
				Node v = e.to;
				// Si 'v' no es alcanzable desde la fuente y la capacidad residual es cero, es parte del min-cut
				if (!reachableFromSource.contains(v) && e.residual == 0) {
					minCutEdges.add(e);  // Esta arista forma parte del min-cut
				}
			}
		}

		return minCutEdges;  // Devuelve el conjunto de aristas que forman el min-cut
	}

	// BFS para encontrar los nodos alcanzables desde la fuente en la red residual
	Set<Node> bfsReachableNodes(Map<Node, List<EdgeFordFulkerson>> adjList, Node source) {
		Set<Node> reachable = new HashSet<>();
		Queue<Node> queue = new LinkedList<>();
		queue.add(source);
		reachable.add(source);

		while (!queue.isEmpty()) {
			Node u = queue.poll();
			List<EdgeFordFulkerson> edges = adjList.get(u);

			if (edges == null) continue;  // Si no hay aristas salientes, continuamos

			// Explorar todas las aristas salientes de 'u'
			for (EdgeFordFulkerson e : edges) {
				Node v = e.to;
				// Si la capacidad residual es positiva y 'v' no ha sido visitado
				if (e.residual > 0 && !reachable.contains(v)) {
					reachable.add(v);
					queue.add(v);
				}
			}
		}

		return reachable;  // Conjunto de nodos alcanzables desde la fuente
	}


	Graph constructConditionedGraph(Dag g, Node x, Node y, LinkedList<Node> cond){
		// 1. Inicialización: Listas abiertas (nodos a explorar) y cerradas (nodos ya visitados)
		LinkedList<Node> open = new LinkedList<>();
		HashMap<String,Node> close = new HashMap<>();

		// Se añaden x, y y los nodos del conjunto condicional a la lista de exploración
		open.add(x);
		open.add(y);
		open.addAll(cond);

		// 2. Exploración inicial del grafo
		while (!open.isEmpty()){
			Node a = open.getFirst();      // Tomamos el primer nodo a explorar
			open.remove(a);                // Lo eliminamos de la lista de abiertos
			close.put(a.toString(),a);     // Lo marcamos como visitado

			// Obtenemos los padres del nodo actual
			List<Node> pa =g.getParents(a);
			for(Node p : pa){
				// Si el padre no ha sido visitado, lo añadimos a la lista de exploración
				if(close.get(p.toString()) == null){
					if(!open.contains(p)) open.addLast(p);
				}
			}
		}

		// 3. Construcción de un grafo auxiliar
		Graph aux = new EdgeListGraph();
		for (Node node : g.getNodes()) aux.addNode(node);  // Copiamos los nodos del grafo original

		Node nodeT, nodeH;
		for (Edge e : g.getEdges()){
			if(!e.isDirected()) continue;  // Solo consideramos aristas dirigidas

			nodeT = e.getNode1();
			nodeH = e.getNode2();

			// Si ambos nodos de la arista están en el conjunto cerrado (alcanzables), añadimos la arista
			if((close.get(nodeH.toString())!=null)&&(close.get(nodeT.toString())!=null)){
				Edge newEdge = new Edge(e.getNode1(),e.getNode2(),e.getEndpoint1(),e.getEndpoint2());
				aux.addEdge(newEdge);
			}
		}

		// 4. Revisión de relaciones entre padres en el grafo auxiliar
		close = new HashMap<>();
		for(Edge e: aux.getEdges()){
			if(e.isDirected()){
				Node h;

				// Identificamos el nodo hijo de la arista
				if(e.getEndpoint1()==Endpoint.ARROW){
					h = e.getNode1();
				} else h = e.getNode2();

				if(close.get(h.toString())==null){
					close.put(h.toString(),h); // Marcamos el nodo hijo como visitado

					// Obtenemos los padres del nodo hijo
					List<Node> pa = aux.getParents(h);
					if(pa.size()>1){ // Si tiene más de un padre
						// Conectamos entre sí a los padres que no están conectados
						for(int i = 0 ; i< pa.size() - 1; i++) {
							for (int j = i + 1; j < pa.size(); j++) {
								Node p1 = pa.get(i);
								Node p2 = pa.get(j);
								boolean found = false;
								// Verificamos si ya están conectados
								for (Edge edge : aux.getEdges()) {
									if (edge.getNode1().equals(p1) && (edge.getNode2().equals(p2))) {
										found = true;
										break;
									}
									if (edge.getNode2().equals(p1) && (edge.getNode1().equals(p2))) {
										found = true;
										break;
									}
								}
								// Si no están conectados, añadimos una arista no dirigida
								if (!found) aux.addUndirectedEdge(p1, p2);
							}
						}
					}
				}
			}
		}

		// 5. Transformación de aristas dirigidas a no dirigidas
		for(Edge e: aux.getEdges()){
			if(e.isDirected()){
				e.setEndpoint1(Endpoint.TAIL);
				e.setEndpoint2(Endpoint.TAIL);
			}
		}

		// 6. Remover nodos condicionales del grafo auxiliar
		aux.removeNodes(cond);

		return aux;
	}

	boolean dSeparated(Dag g, Node x, Node y, LinkedList<Node> cond){
		// TODO_: Hasta aquí hace lo de los ancestros
		Graph aux = constructConditionedGraph(g, x, y, cond);

		// a partir de aquí buscamos si existe algún camino posible entre x e y
		// tenemos que cambiarlo para buscar el mínimo corte para que no exista ese camino
		// en el subconjunto que devuelve mínimo va a haber uno del grafo original (que no se haya generado al moralizar)

		// 7. Segundo recorrido BFS para buscar un camino entre x e y
		LinkedList<Node> open = new LinkedList<>();
		HashMap<String,Node> close = new HashMap<>();
		open.add(x);  // Comenzamos la búsqueda desde x

		while (!open.isEmpty()){
			Node a = open.getFirst();
			if(a.equals(y)) return false;  // Si encontramos y, retornamos false (no están d-separados)
			open.remove(a);
			close.put(a.toString(),a);     // Marcamos el nodo como visitado

			// Obtenemos los nodos adyacentes (vecinos)
			List<Node> pa =aux.getAdjacentNodes(a);
			for(Node p : pa){
				// Si el nodo adyacente no ha sido visitado, lo añadimos a la lista de exploración
				if(close.get(p.toString()) == null){
					if(!open.contains(p)) open.addLast(p);
				}
			}
		}

		// Si no se encontró un camino entre x e y, están d-separados
		return true;
	}

    
    public Dag getFusion(){
    	return this.outputDag;
    }

}
