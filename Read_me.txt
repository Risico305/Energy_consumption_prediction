Best Model:

* In my opinion, with the BEST_FIT feature set ETR known as Extra Tree Regressor had performed better than others on R2 score
 in most cases (More information in the code and code compilation)

* I have performed both feature Ablation Tests and taken correlation statistics to find the best fit features.
Both of them resulted in almost same result.

* Models like Neural Networks (e.g., Multi-layer Perceptron) models usually require relatively larger data to train and
 will most probably fail on this data. I also experimented with Recurrent Networks like Gated Recurrent Unit (GRU)
 (Architecture specification and model parameters are depicted in the code) despite higher R square measure the model is my opinion
 is not doing great due to the less data, thus, less correlations it was able to learn and resulted in irrelgular R2
 which at many occasion adjusted to negative on the test R2 score.
 Moreover, R2 is not the perfect indicator of how well your model is performing in some situations
 and we also need to consider residuals as well. Thus, this needs more testing.

* Have not done the parameter optimization that may lead to better result and hve not plotted ablation test results
 on graph for visual representations due to the less time I had due to my other projects and work!!





*** By executing the code:
Training regimes of individual alorithms, ablation test results eliciting important features, and important statistics will come up!