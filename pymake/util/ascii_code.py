
""" Pygraphviz colorsceme
	ref: http://www.graphviz.org/doc/info/colors.html
"""

#SVG_colors = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure","beige", "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse","chocolate", "coral", "cornflowerblue", "cornsilk", "crimson","cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray","darkgreen", "darkgrey", "darkkhaki", "darkmagenta", "darkolivegreen","darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen","darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet","deeppink", "deepskyblue", "dimgray", "dimgrey", "dodgerblue","firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro","ghostwhite", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred","indigo", "ivory", "khaki", "lavender", "lavenderblush","lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan","lightgoldenrodyellow", "lightgray", "lightgreen", "lightgrey", "lightpink","lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey","lightsteelblue", "lightyellow", "lime", "limegreen", "linen","magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid","mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise","mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin","navajowhite", "navy", "oldlace", "olive", "olivedrab","orange", "orangered", "orchid", "palegoldenrod", "palegreen","paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown","seagreen", "seashell", "sienna", "silver", "skyblue","slateblue", "slategray", "slategrey", "snow", "springgreen","steelblue", "tan", "teal", "thistle", "tomato","turquoise", "violet", "wheat", "white", "whitesmoke","yellow", "yellowgreen"]


X11_colors = ["aliceblue" , "antiquewhite4" , "aquamarine4" , "azure4", "bisque3" , "blue1"  	, "brown"  , "burlywood" , "cadetblue" , "chartreuse", "chocolate" , "coral"	, "cornflowerblue" 	, "cornsilk4" 	, "cyan3" 	 	, "darkgoldenrod3" 	, "darkolivegreen1" , "darkorange1" 	, "darkorchid1" 	, "darkseagreen" 	, "darkslateblue" 	, "darkslategray4" 	, "deeppink1" 	, "deepskyblue1" 	, "dimgrey" 	, "dodgerblue4" 	, "firebrick4" , "gold" 	, "goldenrod" 	, "gray" ]


""" Matplotlib
	ref: http://matplotlib.org/examples/color/named_colors.html
"""
from matplotlib import colors
plt_colors = [(k,v) for k,v in colors.cnames.items()]
