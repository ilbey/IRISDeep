# IRISDeep
This project implemented an identity recognition system based on convolutional neural networks (CNN), and evaluated it with the provided dataset.

1. Identity recognition based on iris biometric data

The main purpose of the identity recognition systems is to determine the identity of the person in a query image. The recognition is done by evaluating the semantic contents of the query image. In a typical practical iris recognition system complete processing chain encompassing initial image segmentation, normalization, feature encoding, and the matching steps used to required. However, performance of these systems can be degraded very easily with the failure of the segmentation stage.

In order to overcome this difficulty iris recognition can be formulated as a learning problem to match an image representation with the identity of person in the eye image. Hence, in this assignment, you are required to construct a CNN with rectified linear unit (ReLU) as a nonlinearity function between layers and train it using the provided images. While training the network you are required to use the softmax (cross-entropy loss) function to minimize the difference between the actual identity and the estimated one.

2. Database

The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilized for this project. Four eye images (two left and two right) were acquired in two different sessions with a resolution of 640*480 pixels from 200 subjects. Since the left and right eye of an individual is completely different. You will consider them as different individuals. Hence, in this case, the database will contain 400 subjects each with 4 eye images.

For more detailed information, please check the pdf project file => [IRISDeep.pdf](https://github.com/ilbey/IRISDeep/files/9104651/IRISDeep.pdf)

For Project Results and Report, please check the pdf file => [IRISDeep_Report.pdf](https://github.com/ilbey/IRISDeep/files/9106403/IRISDeep_Report.pdf)
