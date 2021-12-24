# Stack Overflow Question Quality Classification Using Deep Learning Techniques
## Abstract
Community Question Answering (CQA) forums like Stack Overflow play an important role to support developers of all experience levels. Thus, it is essential to establish an automatic quality control metric to filter high-quality questions better than current manual moderation methods.  
In this paper, we apply different natural language processing and deep learning techniques to classify high-quality questions based on linguistic features and assigned tags. Using random forests, we evaluate question features most influential to the quality of the posts. In accordance with our findings, we conclude that an approach that combines deep learning and natural language processing methods serves as an accurate solution to the automated quality classification problem for Stack Overflow. We found that bi-directional LSTM and CNN had higher accuracies than BERT although BERT had higher precision and recall. Furthermore, we found that when evaluating the dataset using sentiment analysis, Neural Network Classifcation had an accuracy of about 46\% while our Random Forest Classifier had an accuracy of about 51\% and found tags to be the most influential feature to predicting post quality.

## Research Paper:
The research paper is included within this GitHub Repository as "Stack Overflow Question Quality Classification". Click [here](https://github.com/cindylay/cs159-final-proj/blob/main/Stack%20Overflow%20Question%20Quality%20Classification.pdf) to access the PDF.

# Repository Structure
```
.
├── keypair <- ここにaws上のEC2インスタンスに接続するための.pemを置く（gitignore）
│   └── Revorf-sbm-server.pem
├── sbm_project_template <- アロステリックパス解析のテンプレート
│   ├── AllostericPathFinder <- ABC社のQUBO作成用・SBM出力結果評価用プログラム
│   │   ├── Dockerfile
│   │   ├── Makefile
│   │   └── src
│   ├── SBM_result_visualization
│   │   └── script.py
│   ├── data <- ここに.pdbファイルとconfig.iniを置く
│   │   └── config.ini
│   ├── src <- ここにソースファイルおよびノートブックを置く
│   ├── docker-compose.yml
│   ├── .env
│   └── README.md <- Projectの概要等を書く
├── kimura <- 木村作業用ディレクトリ
│   └── test_4obe
├── taniya <- 谷家作業用ディレクトリ
└── README.md

```
