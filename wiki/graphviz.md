# Graph Visualization

The attribute reference for the dot grammar (eg graph.dot files) used by grapviz are defined here:
* [http://www.graphviz.org/doc/info/attrs.html](http://www.graphviz.org/doc/info/attrs.html)		


Example of making a graph using dot:
    cat graph.dot
	digraph {
    /* declare the node & style them */
    "Node 1" [shape=diamond, penwidth=3, style=filled, fillcolor="#FCD975"];
    "Node 2" [style=filled,fillcolor="#9ACEEB" ];
    "Node 3" [shape=diamond, style=filled, fillcolor="#FCD975" ];
    "Node 4" [style=filled, fillcolor="#9ACEEB" ]

    /* declare the edges & style them */
    "Node 1" -> "Node 2" [dir=none, weight=1, penwidth=3] ;
    "Node 1" -> "Node 3" [dir=none, color="#9ACEEB"] ;
    "Node 1" -> "Node 4" [arrowsize=.5, weight=2.]
}

    dot  -Tpng  graph.dot > graph.png

	[graph.png]*(graph.png)


[1]: a dot guide: http://graphviz.org/Documentation/dotguide.pdf
