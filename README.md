# Semantic_Segmentation
Experiments with semantic segmentation on the tas500 dataset

Two main network types are tested, an iterative network which tries to (iteratively) improve a semantic segmentation and the hourglass network which was successfully applied to the problem of pose estimation.

Network models which are trained are an iterative network (network is applied several times to its own output) and an hourglass network. The iterative networks have the following structure

| Network       | Initial network | Copy input | Residual connections |
| ------------- | --------------- | ---------- | -------------------- |
| IterativeV1   | x               | x          |                      |
| IterativeV2   |                 | x          |                      |
| IterativeV3   | x               | x          | x                    |
| IterativeV4   |                 | x          | x                    |
| IterativeV5   | x               |            | x                    |
| IterativeV6   |                 |            | x                    |

Initial network: An initial network produces a semantic segementation and then the iterative network is applied N times

Copy input: The input to the initial network (or the iterative network if there is no initial network) is copied and fed into the network at each iteration so the iterative network doesn't need to extract all the information from the input at once

Residual connections: Residual connections from input to output of network
