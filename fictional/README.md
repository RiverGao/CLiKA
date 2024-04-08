**Dataset Description**

  This supplementary material contains the data used in the CD, aimed at assisting readers in better understanding and verifying the experiments.

  It comprises two folders: the training part includes training sets in nine languages. The test set includes the testing Translate and testing knowledge sections. The Translate section contains all matching results, whereas the knowledge part includes reference outputs and tags, such as: [[continent:Mythosia]][[city:Solarianna Forest]][[type:0]]. Here, the continent is used to check for correct results, city is used for statistics in preliminary experiments, and type is categorized into 0 and 1. When training in a specific language, such as ZH, it is necessary to test the Translate section, and types 0 and 1 for EN and type 0 for ZH. Training is considered complete only if the accuracy of the above sections exceeds 90%. Subsequently, ZH's type 1 is used as a test set to assess the model's CD capabilities.




