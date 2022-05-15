# Semantic Segmentation
Experiments with semantic segmentation on the tas500 dataset

Two main network types are tested, an iterative network which tries to (iteratively) improve a semantic segmentation and the hourglass network which was successfully applied to the problem of pose estimation.

# Network types

## Iterative

The iterative network type has an iterative network which is applied to its own output and possibly an initial network which generates an initial semantic segmentation.

| Network       | Initial network | Copy input | Residual connections | Result |
| ------------- | :-------------: | :--------: | :------------------: | ------ |
| IterativeV1   | x               | x          |                      |        |
| IterativeV2   |                 | x          |                      |        |
| IterativeV3   | x               | x          | x                    |        |
| IterativeV4   |                 | x          | x                    |  Good  |
| IterativeV5   | x               |            | x                    |        |
| IterativeV6   |                 |            | x                    |  Good  |

Initial network: An initial network produces a semantic segementation and then the iterative network is applied N times.

Copy input: The input to the initial network (or the iterative network if there is no initial network) is copied and fed into the network at each iteration so the iterative network doesn't need to extract all the information from the input at once.

Residual connections: Residual connections from input to output of network.

## Hourglass

The hourglass network types are inspired by (small variations of) the network from *Stacked Hourglass Networks for Human Pose Estimation* by Newell et. al.

| Network               | Copy input | Copy-ex | Residual connections | Results |
| --------------------- | :--------: | :-----: | :------------------: | :-----: |
| Hourglass             |            |         |                      |         |
| Hourglass_iter        | x          |         |                      |         |
| Hourglass_iter_ex     |            | x       |                      |         |
| res_Hourglass         |            |         | x                    | Good    |
| res_Hourglass_iter    | x          |         | x                    |         |
| res_Hourglass_iter_ex |            | x       | x                    |         |

Copy input: The input to the first network is copied and fed into the hourglass module at each iteration so each module doesn't need to extract all the information from the input at once.

Copy-ex: The extractor for copy-input shares the same weights for all hourglass modules, copy-ex has new weights for each extractor for each module

Residual connections: Residual connections from input to output of network.

# Results

The iterative network structures seems promising while having a very small footprint in memory (1/5th the size of the same hourglass network), further testing is required on different problems, preferably with more data.

The hourglass network could most likely be successfully applied to semantic segmentation but the model becomes very big so it cannot produce a large ouput.

Residuals connections seem to be very important both for learning and reaching a high accuracy.
