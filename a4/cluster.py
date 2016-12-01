"""
cluster.py
"""

import networkx as nx
import pickle
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
import math

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    visited = defaultdict(bool)
    q = deque()
    end_of_level = "*"
    init_node = "init"

    # Check input
    if graph is None:
        return node2distances, node2num_paths, node2parents
    if root == "" or graph.has_node(root) == False:
        return node2distances, node2num_paths, node2parents
    if max_depth < 0:
        return node2distances, node2num_paths, node2parents

    visited[root] = True
    node2distances[root] = 0
    node2num_paths[root] = 1
    depth = 1
    max_depth = min(max_depth, math.inf)
    q.appendleft(root)
    q.appendleft(end_of_level)
    while depth <= max_depth:
        while True and len(q) != 0:
            node = q.pop()
            if node == end_of_level:
                break
            for neighbor in graph.neighbors(node):
                if not visited[neighbor]:
                    q.appendleft(neighbor)
                    node2distances[neighbor] = depth
                    visited[neighbor] = True
                if node2distances[neighbor] >= depth:
                    node2parents[neighbor].append(node)
                    node2num_paths[neighbor] += 1
        depth += 1
        if len(q) == 0:
            break
        q.appendleft(end_of_level)
    return node2distances, node2num_paths, node2parents

def complexity_of_bfs(V, E, K):
    """
    If V is the number of vertices in a graph, E is the number of
    edges, and K is the max_depth of our approximate breadth-first
    search algorithm, then what is the *worst-case* run-time of
    this algorithm? As usual in complexity analysis, you can ignore
    any constant factors. E.g., if you think the answer is 2V * E + 3log(K),
    you would return V * E + math.log(K)
    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    """
    if V <= 0:
        return 0
    complexity = (V + E)
    return complexity


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    result = defaultdict(lambda: 0.0)
    nodes = defaultdict(lambda: 0.0)
    if root == "":
        return result
    if (len(node2distances) == 0) or (len(node2num_paths) == 0) or (len(node2parents) == 0):
        return result
    nodes_list = sorted(node2distances.keys(),key= node2distances.get, reverse=True)
    for node in nodes_list:
        nodes[node] += 1
    for node in nodes_list:
        parents = node2parents[node]
        for parent in parents:
            edge = tuple(sorted((node, parent)))
            result[edge] = ((nodes[node]) * (node2num_paths[parent] / node2num_paths[node]))
            nodes[parent] += result[edge]
    return result


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    betweeness = Counter()
    if graph is None:
        return betweeness
    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        betweeness.update(bottom_up(node, node2distances, node2num_paths, node2parents))
    for edge,_ in betweeness.items():
        betweeness[edge] /= 2
    return betweeness


def is_approximation_always_right():
    """
    Look at the doctests for approximate betweenness. In this example, the
    edge with the highest betweenness was ('B', 'D') for both cases (when
    max_depth=5 and max_depth=2).

    Consider an arbitrary graph G. For all max_depth > 1, will it always be
    the case that the edge with the highest betweenness will be the same
    using either approximate_betweenness verses the exact computation?
    Answer this question below.

    In this function, you just need to return either the string 'yes' or 'no'
    (no need to do any actual computations here).
    >>> s = is_approximation_always_right()
    >>> type(s)
    <class 'str'>
    """
    answer = "no"
    return answer


def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    components = []
    if graph is None or max_depth < 0:
        return components
    component_graph = graph.copy()
    components = [component for component in nx.connected_component_subgraphs(component_graph)]
    if len(components) > 1:
        return components
    betweenness = approximate_betweenness(component_graph, max_depth)
    if len(betweenness) == 0:
        return components
    edges = sorted(betweenness.items(), key= lambda score: (-score[1],score[0][0],score[0][1]))
    for edge in edges:
        if len(components) > 1:
            break
        component_graph.remove_edge(*edge[0])
        components = [component for component in nx.connected_component_subgraphs(component_graph)]
    return components



def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    result_graph = nx.Graph()
    if graph is None:
        return result_graph
    for node in graph.nodes():
        if graph.degree(node) >= min_degree:
            result_graph.add_node(node)
    for node in result_graph.nodes():
        for neighbor in graph.neighbors(node):
            if result_graph.has_node(neighbor):
                result_graph.add_edge(node, neighbor)
    return result_graph


def check_nodes_exist_in_graph(graph, nodes):
    if graph is None or len(nodes) == 0:
        return False
    return len(set(graph.nodes()).intersection(nodes)) == len(nodes)


def girvan_newman(G, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html

    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if G.order() == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        print(indent + 'removing ' + str(edge_to_remove))
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    result = [c.nodes() for c in components]
    print(indent + 'components=' + str(result))
    for c in components:
        result.extend(girvan_newman(c, depth + 1))

    return result

def load_pickle_file(filename):
    with open(filename, "rb") as handle:
         while True:
            try:
                 yield pickle.load(handle)
            except EOFError:
                break

def read_user_data(filename):
    return load_pickle_file(filename)

def jaccard_similarity(user1, user2):
    if len(user1)==0 and len(user2)==0:
        return 0
    user1 = set(user1)
    user2 = set(user2)
    num = len(user1.intersection(user2))
    deno = len(user1.union(user2))
    return num / deno


def create_graph(user_friends, num_users=100, jaccard_threshold=0.0001):
    graph = nx.Graph()
    for index, user1 in enumerate(user_friends):
        if index > num_users:
            break
        user_iter = 0
        graph.add_node(user1[0])
        while user_iter < len(user_friends):
            user2 = user_friends[user_iter]
            graph.add_node(user2[0])
            if jaccard_similarity(user1[1], user2[1]) >= jaccard_threshold:
                graph.add_edge(user1[0], user2[0])
            user_iter += 1
    return graph


def create_subgraph(graph, num_nodes):
    if len(graph.nodes()) <= num_nodes:
        return graph
    return graph.subgraph(graph.nodes()[:num_nodes])


def get_communities(graph):
    return partition_girvan_newman(graph, max_depth=50)


def draw_network(graph, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """

    nx.draw_networkx(graph)
    plt.axis('off')
    plt.savefig(filename)

def write_summary(filename, user_friends, communities):
    number_of_users_collected = len(user_friends)
    number_of_communities_discovered = len(communities)
    average_number_of_users_per_community = 0
    if number_of_communities_discovered != 0:
        average_number_of_users_per_community = number_of_users_collected / number_of_communities_discovered
    with open(filename, "wb") as handle:
        pickle.dump(number_of_users_collected, handle)
        pickle.dump(number_of_communities_discovered, handle)
        pickle.dump(average_number_of_users_per_community, handle)


def main():
    user_friends = list(read_user_data("users_friends"))
    graph = create_graph(user_friends, len(user_friends))
    #subgraph = create_subgraph(graph, 100)
    draw_network(graph, "network.png")
    communities = get_communities(graph)
    print(communities)
    for index, community in enumerate(communities):
        print(community.order())
        draw_network(community, "community%d.png" % index)
    write_summary("cluster_summary", user_friends, communities)

if __name__ == '__main__':
    main()