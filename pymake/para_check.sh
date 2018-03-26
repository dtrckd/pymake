ps -aux | grep "ssh.*parallel"
bz2.decompress(base64.b64decode(code))


####

dulac    56796  0.0  0.0  50724  3084 pts/10   S    12:34   0:00 ssh -l adulac tiger -- exec perl -e @GNU_Parallel\=split/_/,\"use_IPC::Open3\;_use_MIME::Base64\"\;eval\"@GNU_Parallel\"\;\$SIG\{CHLD\}\=\"IGNORE\"\;my\$zip\=\(grep\{-x\$_\}\"/usr/local/bin/bzip2\"\)\[0\]\|\|\"bzip2\"\;open3\(\$in,\$out,\"\>\&STDERR\",\$zip,\"-dc\"\)\;if\(my\$perlpid\=fork\)\{close\$in\;\$eval\=join\"\",\<\$out\>\;close\$out\;\}else\{close\$out\;print\$in\(decode_base64\(join\"\",@ARGV\)\)\;close\$in\;exit\;\}wait\;eval\$eval\; QlpoOTFBWSZTWWpbS0EAAbifgHV////u5/8ev//f/kACJhcoAhNTKJ/qQ9KA8hsinqBtT0mRoMRo/VAeozUxqDjJk0MRiaMAjATCAMBNNGmRoBhqYTSNSfomk0NGnqYg0D1ANAAAAAGSaQaNEYmSfqT0JigPSHqZG1AaPUbSAzTS/T1HVp+YfvSnWxsksbG2pJTMzMzMzN03IuPXBFGz31NaT3loqPo2b9vAOP4srTRAAMkjW3+ucJv1iRcVC1qc7w47Idcwv21vB+wp0KZHCaTZ7IV/FVxD7jq5VlNW3Te9WmF3GTywR2UrsaMoTB8LROxWeyd6GgvZaZxJWaDWr0wooixGoW3jcTjDVOMnICcRzObNuUSGi+RYIQnTC02TCItyRjf3rguu12aRnSZ251XSibW7QeQe98FDlIAzXlRQAqtPXmcGQPkCQEyBKDCnlSzY2E1AreXK+ORRHfRzg9EztBcRQYUaMwBOA+Y58X27OrdLuYBQDt6pYwi/d26lozmecB4GEhnY4b/5EDiVBYG79hQR1zgPQLZJLstSpp7XYgwGnzaN7MdwrQR+gzjSNLwjRWkSkQKcQVBCFbXcBOEtoahwBYTYUKB6doT0B5bseZYwk1vxbUUL1Rn+EXSame/CNtzQhmMXBJzUL7mLcXlu2gce0xGFAeBnzGuI0BALFntmsERD7RBvWgHIpNqm3JAMTKVVagqiQIh/ZP6wHlsMnAN+BeHIVGMpoChualzZsJbjLDxgBA5XZLwXRDRkmrtUU1GvKnQ3mSU5kBdLhKIE8NRgheGABDrK59QwBDyFCw6agwTiMAaPJz4BSkL2PkbZOOacGSTlOUah001+d2dVoN5bEBBUHBAKEvk3G7WkXQ024NekU4/Zx0wudUW7WnxF3JFOFCQaltLQQA\=\=

