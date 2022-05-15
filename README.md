# Semantic Segmentation
Experiments with semantic segmentation on the tas500 dataset

Two main network types are tested, an iterative network which tries to (iteratively) improve a semantic segmentation and the hourglass network which was successfully applied to the problem of pose estimation.

# Network types

## Initial network

To make training times faster a distillation network was trained which made the input picture smaller and a segmentation of this smaller picture was the training label.

The distillation networks are

| Network | Size of ouput compared to input |
| :-----: | :-----------------------------: |
| x1      | 1                               |
| x2      | 1/4                             |
| x4      | 1/16                            |
| x8      | 1/64                            |

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

The hourglass network could most likely be successfully applied to semantic segmentation but the model becomes very big so it cannot produce a large ouput. Residuals connections seem to be very important both for learning and reaching a high accuracy.

All the values that are reported are results on a deterministic (same for all experiments) validation set. More data is most likely needed to test the true potential of the models.

The results for the network types trained for 100 epochs with ADAM in pytorch with learning rate 1e-3 with the distillation network x8 are

| Network               | Pixel Accuracy (%) | mIoU (%) |
| --------------------- | ------------------ | -------- |
| IterativeV1           | 94.2               | 63.7     |
| IterativeV2           | 91.7               | 58.7     |
| IterativeV3           | 93.8               | 53.0     |
| IterativeV4           | 95.7               | **71.6** |
| IterativeV5           | 97.9               | 59.2     |
| IterativeV6           | 95.7               | **74.6** |
| Hourglass             | 91.9               | 59.6     |
| Hourglass_iter        | 97.1               | 51.2     |
| Hourglass_iter_ex     | 97.6               | 59.4     |
| res_Hourglass         | 97.9               | **85.2** |
| res_Hourglass_iter    | 97.7               | 58.7     |
| res_Hourglass_iter_ex | 88.9               | 56.7     |

the networks which seemes to perform well were then trained with the same parameters for 250 epochs

| Network               | Pixel Accuracy (%) | mIoU (%) |
| --------------------- | ------------------ | -------- |
| IterativeV1           | 95.2               | 70.0     |
| IterativeV4           | 96.5               | 79.3     |
| IterativeV6           | 96.1               | 65.3     |
| res_Hourglass         | 98.0               | 97.8     |

and the iterativeV6 network type was also trained with the same parameters but with x2 as the distillation network

| Network               | Pixel Accuracy (%) | mIoU (%) |
| --------------------- | ------------------ | -------- |
| IterativeV6           | 96.8               | 88.0     |

The accuracy could be better but improvement does seem to happen at each iteration which can be seen on the plot of mIoU in the results folder
