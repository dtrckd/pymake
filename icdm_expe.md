
* Performance of immsb is better for bursty. (consistent with homophily un peu..to confirm)
* Performance of ibp is better for non-bursty (consistent with homophily...to confirm)

Prediction task is conducted for models according two scheme
* mask_sub1: where the testing set is randomly chosen with a proportional quantity for links versus non-links. 20% of the links is considered for the mask.
* mask_all: where the testing set is choosed randomly on on 20 % of the size of the networks.

The impact of homophilly/burstiness on predictive task is visible on  mask_sub1 for local precsion and on global precision for prediction mak over all the networks.


Wich one is better than the other ? rappel ?

* The differnece of two models according to the propertie reduce with the number of latent class ? dimensionality reduction advantage...

* links between homophily/burstiness and the number of class.


generator4
Similarity | Hobs | Hexp                        
community   0.92393442623  0.262846846847                        
euclide_old     0.745803278689  0.42441041041                        
euclide_abs     0.660885245902  0.474674674675                        
euclide_dist     0.300360655738  0.19429029029         
generator10
Similarity | Hobs | Hexp                        
community   0.872181818182  0.252194194194                        
euclide_old     0.416545454545  0.42441041041                        
euclide_abs     0.439090909091  0.474674674675                        
euclide_dist     0.178727272727  0.19429029029         
generator12
Similarity | Hobs | Hexp                        
community   0.4292  0.251327327327                        
euclide_old     0.4136  0.42441041041                        
euclide_abs     0.4288  0.474674674675                        
euclide_dist     0.198  0.19429029029         
generator7
Similarity | Hobs | Hexp                        
community   0.66178915863  0.258144144144                        
euclide_old     0.608912537413  0.42441041041                        
euclide_abs     0.561024276688  0.474674674675                        
euclide_dist     0.274027269704  0.19429029029 

###### Graph4                                                                                                                                                 [8/4134]
        Building: None minutes
        Nodes: 1000
        Links: (61000.0,)
        Degree mean: 61.0
        Degree var: 2490.456
        Diameter: 5
        Clustering Coefficient: 0.61549924713
        Density: 0.0620620620621
        Communities: 4
        Relations: 2.0
        Directed: False
        

###### Graph10
        Building: None minutes
        Nodes: 1000
        Links: (11000.0,)
        Degree mean: 11.0
        Degree var: 99.62
        Diameter: 8
        Clustering Coefficient: 0.495707248204
        Density: 0.012012012012
        Communities: 4
        Relations: 2.0
        Directed: False
        

###### Graph12
        Building: None minutes
        Nodes: 1000
        Links: (5000.0,)
        Degree mean: 5.0
        Degree var: 54.734
        Diameter: 10
        Clustering Coefficient: 0.0800805641164
        Density: 0.00600600600601
        Communities: 4
        Relations: 2.0
        Directed: False

        

###### Graph7
        Building: None minutes
        Nodes: 1000
        Links: (6014.0,)
        Degree mean: 6.014
        Degree var: 68.183804
        Diameter: 8
        Clustering Coefficient: 0.0641057824548
        Density: 0.00702102102102
        Communities: 4
        Relations: 2.0
        Directed: False
        

        
